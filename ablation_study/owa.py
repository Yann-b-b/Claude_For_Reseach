#!/usr/bin/env python3
"""
Run baseline models on a fixed 70/15/15 split (seed=42) and report test accuracy.

Baselines implemented:
1) Majority class
2) Random (p = training prevalence)
3) Logistic regression on peptide length
4) Logistic regression on peptide AAC (20-d) + length
5) RandomForest on peptide AAC + length
6) Peptide-only BiLSTM (PyTorch)
7) DNA-only CNN (PyTorch)

Usage:
  python ablation_study/run_baselines.py
"""

import sys
from pathlib import Path
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from antimicrobial_predictor import load_real_dataset, SequenceEncoder


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_70_15_15(dna: List[str], pep: List[str], y: List[int], seed: int = 42):
    dna_train, dna_temp, pep_train, pep_temp, y_train, y_temp = train_test_split(
        dna, pep, y, test_size=0.30, random_state=seed, stratify=y
    )
    dna_val, dna_test, pep_val, pep_test, y_val, y_test = train_test_split(
        dna_temp, pep_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return (dna_train, pep_train, y_train), (dna_val, pep_val, y_val), (dna_test, pep_test, y_test)


# --- Feature Engineering for classical ML ---
AA_LIST = list("ARNDCQEGHILKMFPSTWYV")
AA_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def peptide_aac(peptide: str) -> np.ndarray:
    v = np.zeros(len(AA_LIST), dtype=np.float32)
    if peptide:
        for ch in peptide:
            if ch in AA_IDX:
                v[AA_IDX[ch]] += 1.0
        total = v.sum()
        if total > 0:
            v /= total
    return v


def build_peptide_features(peps: List[str]) -> np.ndarray:
    feats = []
    for p in peps:
        p = (p or "").upper()
        aac = peptide_aac(p)
        length = len(p)
        feats.append(np.concatenate([aac, np.array([length], dtype=np.float32)]))
    return np.vstack(feats)


# --- PyTorch peptide-only model ---
class PeptideLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int = 21, emb_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, L)
        e = self.emb(x)
        o, _ = self.lstm(e)
        mask = (x != 0).float().unsqueeze(-1)
        o = (o * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(o).squeeze(-1)


# --- PyTorch DNA-only model ---
class DNACNNClassifier(nn.Module):
    def __init__(self, vocab_size: int = 5, emb_dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(emb_dim, 32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        # x: (B, L)
        e = self.emb(x).transpose(1, 2)  # (B, C, L)
        h = torch.relu(self.conv1(e))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return self.fc(h).squeeze(-1)


class SeqOnlyDataset(Dataset):
    def __init__(self, seqs: List[str], encoder: SequenceEncoder, is_peptide: bool, max_len_dna: int = 50, max_len_pep: int = 100):
        self.seqs = seqs
        self.encoder = encoder
        self.is_peptide = is_peptide
        self.max_len_dna = max_len_dna
        self.max_len_pep = max_len_pep

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        if self.is_peptide:
            x = torch.tensor(self.encoder.encode_protein(s, max_length=self.max_len_pep), dtype=torch.long)
        else:
            x = torch.tensor(self.encoder.encode_dna(s, max_length=self.max_len_dna), dtype=torch.long)
        return x


def train_torch_model(model: nn.Module, x_train: List[str], y_train: List[int], x_val: List[str], y_val: List[int],
                      is_peptide: bool, epochs: int = 2, batch_size: int = 256, lr: float = 1e-3,
                      progress_name: str = "model", num_workers: int = 2) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    enc = SequenceEncoder()
    ds_tr = SeqOnlyDataset(x_train, enc, is_peptide)
    ds_va = SeqOnlyDataset(x_val, enc, is_peptide)
    pin = torch.cuda.is_available()
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for ep in range(epochs):
        model.train()
        idx = 0
        iterator = tqdm(dl_tr, desc=f"{progress_name} Epoch {ep+1}/{epochs}") if _HAS_TQDM else dl_tr
        for xb in iterator:
            bsz = xb.size(0)
            yb = y_train_t[idx:idx+bsz].to(device)
            xb = xb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            idx += bsz

        # quick val progress
        model.eval()
        preds = []
        with torch.no_grad():
            for xb in dl_va:
                xb = xb.to(device)
                preds.extend(model(xb).cpu().numpy())
        val_acc = accuracy_score(y_val, (np.array(preds) >= 0.5).astype(int))
        print(f"{progress_name} val acc: {val_acc:.4f}")

    # Validation accuracy (optional)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in dl_va:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.cpu().numpy())
    acc = accuracy_score(y_val, (np.array(preds) >= 0.5).astype(int))
    return acc, model


def test_torch_model(model: nn.Module, x_test: List[str], y_test: List[int], is_peptide: bool) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    enc = SequenceEncoder()
    ds_te = SeqOnlyDataset(x_test, enc, is_peptide)
    dl_te = DataLoader(ds_te, batch_size=128, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb in dl_te:
            xb = xb.to(device)
            out = model(xb)
            preds.extend(out.cpu().numpy())
    return accuracy_score(y_test, (np.array(preds) >= 0.5).astype(int))


def compute_metrics(y_true, scores):
    y_pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float('nan')
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'auc_score': float(auc),
    }


def main():
    set_seed(42)
    dna, pep, y = load_real_dataset()
    (dna_tr, pep_tr, y_tr), (dna_va, pep_va, y_va), (dna_te, pep_te, y_te) = split_70_15_15(dna, pep, y, seed=42)

    # --- Baseline 1: Majority class ---
    results = {}

    majority = int(round(np.mean(y_tr)))
    scores = np.full_like(np.array(y_te, dtype=float), majority, dtype=float)
    results['majority'] = compute_metrics(y_te, scores)
    print(f"Majority: {results['majority']}")

    # --- Baseline 2: Random p=prevalence ---
    p_prev = float(np.mean(y_tr))
    rng = np.random.RandomState(42)
    scores = rng.binomial(1, p_prev, size=len(y_te)).astype(float)
    results['random_prev'] = compute_metrics(y_te, scores)
    print(f"Random (p=prev): {results['random_prev']}")

    # --- Baseline 3: Length-only logistic ---
    Xlen_tr = np.array([[len(p or "")] for p in pep_tr], dtype=np.float32)
    Xlen_te = np.array([[len(p or "")] for p in pep_te], dtype=np.float32)
    lr_len = LogisticRegression(max_iter=1000)
    lr_len.fit(Xlen_tr, y_tr)
    scores = lr_len.predict_proba(Xlen_te)[:,1]
    results['logreg_length'] = compute_metrics(y_te, scores)
    print(f"LogReg (length): {results['logreg_length']}")

    # --- Baseline 4: AAC + length logistic ---
    X_tr = build_peptide_features(pep_tr)
    X_te = build_peptide_features(pep_te)
    lr_aac = LogisticRegression(max_iter=2000)
    lr_aac.fit(X_tr, y_tr)
    scores = lr_aac.predict_proba(X_te)[:,1]
    results['logreg_aac_len'] = compute_metrics(y_te, scores)
    print(f"LogReg (AAC+len): {results['logreg_aac_len']}")

    # --- Baseline 5: RF on AAC + length ---
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    scores = rf.predict_proba(X_te)[:,1]
    results['rf_aac_len'] = compute_metrics(y_te, scores)
    print(f"RandomForest (AAC+len): {results['rf_aac_len']}")

    # --- Baseline 6: MLP on AAC + length ---
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    mlp.fit(X_tr, y_tr)
    scores = mlp.predict_proba(X_te)[:,1]
    results['mlp_aac_len'] = compute_metrics(y_te, scores)
    print(f"MLP (AAC+len): {results['mlp_aac_len']}")

    # --- Baseline 7: XGBoost on AAC + length (if available) ---
    if _HAS_XGB:
        xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb.fit(X_tr, y_tr)
        scores = xgb.predict_proba(X_te)[:,1]
        results['xgb_aac_len'] = compute_metrics(y_te, scores)
        print(f"XGBoost (AAC+len): {results['xgb_aac_len']}")
    else:
        print("XGBoost not installed; skipping xgb baseline")

    # --- Baseline 6: Peptide-only BiLSTM (Torch) ---
    pep_model = PeptideLSTMClassifier()
    # downsample training for speed
    MAX_TRAIN_SAMPLES = 5000
    sel_idx = np.random.RandomState(42).choice(len(pep_tr), size=min(MAX_TRAIN_SAMPLES, len(pep_tr)), replace=False)
    pep_tr_small = [pep_tr[i] for i in sel_idx]
    y_tr_small = [y_tr[i] for i in sel_idx]
    val_acc_pep, pep_model = train_torch_model(pep_model, pep_tr_small, y_tr_small, pep_va, y_va, is_peptide=True, epochs=2, batch_size=256, progress_name="Peptide-LSTM", num_workers=2)
    # test metrics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = SequenceEncoder()
    ds_te = SeqOnlyDataset(pep_te, enc, is_peptide=True)
    dl_te = DataLoader(ds_te, batch_size=128, shuffle=False)
    pep_model.eval(); preds=[]
    with torch.no_grad():
        for xb in dl_te:
            xb = xb.to(device)
            preds.extend(pep_model(xb).cpu().numpy())
    results['pep_bilstm'] = compute_metrics(y_te, np.array(preds))
    print(f"Peptide-only BiLSTM: {results['pep_bilstm']}")

    # --- Baseline 7: DNA-only CNN (Torch) ---
    dna_model = DNACNNClassifier()
    sel_idx = np.random.RandomState(42).choice(len(dna_tr), size=min(MAX_TRAIN_SAMPLES, len(dna_tr)), replace=False)
    dna_tr_small = [dna_tr[i] for i in sel_idx]
    y_tr_small_dna = [y_tr[i] for i in sel_idx]
    val_acc_dna, dna_model = train_torch_model(dna_model, dna_tr_small, y_tr_small_dna, dna_va, y_va, is_peptide=False, epochs=2, batch_size=256, progress_name="DNA-CNN", num_workers=2)
    ds_te_dna = SeqOnlyDataset(dna_te, enc, is_peptide=False)
    dl_te_dna = DataLoader(ds_te_dna, batch_size=128, shuffle=False)
    dna_model.eval(); preds=[]
    with torch.no_grad():
        for xb in dl_te_dna:
            xb = xb.to(device)
            preds.extend(dna_model(xb).cpu().numpy())
    results['dna_cnn'] = compute_metrics(y_te, np.array(preds))
    print(f"DNA-only CNN: {results['dna_cnn']}")

    # Save combined metrics table
    out_dir = PROJECT_ROOT/Path('plots')
    out_dir.mkdir(exist_ok=True)
    rows = []
    for name, m in results.items():
        rows.append({
            'model': name,
            'accuracy': m['accuracy'],
            'precision': m['precision'],
            'recall': m['recall'],
            'f1_score': m['f1_score'],
            'auc': m['auc_score'],
        })
    df = pd.DataFrame(rows)
    # CSV
    csv_path = out_dir/'baseline_metrics.csv'
    df.to_csv(csv_path, index=False)
    # JSON
    json_path = out_dir/'baseline_metrics.json'
    df.to_json(json_path, orient='records', indent=2)
    # PNG table
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 0.6 + 0.4*len(df)))
        plt.axis('off')
        tbl = plt.table(cellText=df.round(4).values,
                        colLabels=df.columns.tolist(),
                        loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        png_path = out_dir/'baseline_metrics.png'
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches='tight')
        plt.close()
        print('Wrote', csv_path, json_path, png_path)
    except Exception as e:
        print('Wrote', csv_path, json_path)
        print('PNG table generation failed:', e)

    # Brute-force idea notes removed from console output per request


if __name__ == "__main__":
    main()


