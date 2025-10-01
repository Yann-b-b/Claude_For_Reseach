#!/usr/bin/env python3
"""
Run ablation experiment:
 - 70/15/15 split (train/val/test)
 - Multi-seed (default 5) training/eval
 - Report mean ± std for ACC/AUC
 - Generate figures: ROC/PR, confusion matrix (test), threshold sweep, calibration,
   learning curves (per-seed + mean), score distributions, length effects
 - Artifacts: per-seed checkpoints via antimicrobial_predictor Trainer (runs/)
 - Outputs under reports/grampa_split70-15-15_seed{N}_YYYYMMDD/{figures,tables}
"""

import os
import json
import math
import random
import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antimicrobial_predictor import AntimicrobialPredictor, AntimicrobialTrainer, load_real_dataset, SequenceEncoder


def split_70_15_15(dna: List[str], pep: List[str], y: List[int], seed: int):
    # First split 70% train, 30% temp
    X = list(zip(dna, pep, y))
    dna_train, dna_temp, pep_train, pep_temp, y_train, y_temp = train_test_split(
        dna, pep, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Split temp into 15% val, 15% test (i.e., half of temp)
    dna_val, dna_test, pep_val, pep_test, y_val, y_test = train_test_split(
        dna_temp, pep_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return (dna_train, pep_train, y_train), (dna_val, pep_val, y_val), (dna_test, pep_test, y_test)


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, out_dir: Path, label: str):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(rec, prec)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC - {label}')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"roc.png", dpi=200); plt.close()

    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f"PR AUC={pr_auc:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR - {label}')
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"pr.png", dpi=200); plt.close()

    return roc_auc, pr_auc


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(4,4))
    plt.imshow(cmn, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{cmn[i,j]:.2f}", ha='center', va='center', color='black')
    plt.title('Confusion Matrix (normalized)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, out_path: Path):
    ts = np.linspace(0.05, 0.95, 19)
    precs, recs, f1s = [], [], []
    for t in ts:
        yp = (scores >= t).astype(int)
        tp = ((yp==1)&(y_true==1)).sum(); fp=((yp==1)&(y_true==0)).sum()
        fn = ((yp==0)&(y_true==1)).sum(); tn=((yp==0)&(y_true==0)).sum()
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        precs.append(prec); recs.append(rec); f1s.append(f1)
    plt.figure(figsize=(6,4))
    plt.plot(ts, precs, label='Precision')
    plt.plot(ts, recs, label='Recall')
    plt.plot(ts, f1s, label='F1')
    plt.xlabel('Threshold'); plt.ylabel('Metric'); plt.title('Threshold Sweep')
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def calibration_plot(y_true: np.ndarray, scores: np.ndarray, out_path: Path):
    # Reliability bins
    bins = np.linspace(0,1,11)
    inds = np.digitize(scores, bins)-1
    prop = []
    conf = []
    for b in range(len(bins)-1):
        mask = inds==b
        if mask.sum()>0:
            conf.append(scores[mask].mean())
            prop.append(y_true[mask].mean())
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.plot(conf, prop, marker='o')
    plt.xlabel('Mean predicted'); plt.ylabel('Fraction positive')
    plt.title('Calibration'); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def score_distributions(y_true: np.ndarray, scores: np.ndarray, out_path: Path):
    plt.figure(figsize=(6,4))
    plt.hist(scores[y_true==1], bins=30, alpha=0.6, label='Pos')
    plt.hist(scores[y_true==0], bins=30, alpha=0.6, label='Neg')
    plt.xlabel('Score'); plt.ylabel('Count'); plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()


def length_effects(seqs: List[str], scores: np.ndarray, out_path: Path, title: str):
    lens = np.array([len(s or "") for s in seqs])
    plt.figure(figsize=(6,4))
    plt.scatter(lens, scores, alpha=0.4)
    plt.xlabel('Sequence length'); plt.ylabel('Score'); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main(num_seeds: int = 5):
    dna, pep, y = load_real_dataset()

    today = datetime.datetime.now().strftime('%Y%m%d')
    exp_dir = Path(f"reports/grampa_split70-15-15_seed{num_seeds}_{today}")
    fig_dir = exp_dir/"figures"
    tab_dir = exp_dir/"tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    accs = []
    aucs = []

    all_seed_metrics = []

    for s in range(num_seeds):
        seed = 42 + s
        set_seed(seed)
        (dna_tr, pep_tr, y_tr), (dna_va, pep_va, y_va), (dna_te, pep_te, y_te) = split_70_15_15(dna, pep, y, seed)

        # Build loaders from the split
        model = AntimicrobialPredictor()
        trainer = AntimicrobialTrainer(model)

        # Compose loaders manually for train/val; and a test loader for evaluation
        train_loader, val_loader = trainer.prepare_data(dna_tr + dna_va, pep_tr + pep_va, y_tr + y_va, test_size=len(y_va)/ (len(y_tr)+len(y_va)))

        # Train
        train_losses, val_losses = trainer.train(train_loader, val_loader, num_epochs=20)

        # Evaluate on test split
        # Build a test DataLoader
        encoder = trainer.encoder
        from torch.utils.data import DataLoader, Dataset
        class TestDataset(Dataset):
            def __len__(self): return len(y_te)
            def __getitem__(self, idx):
                return {
                    'dna': torch.tensor(encoder.encode_dna(dna_te[idx]), dtype=torch.long),
                    'protein': torch.tensor(encoder.encode_protein(pep_te[idx]), dtype=torch.long),
                    'label': torch.tensor(y_te[idx], dtype=torch.float32)
                }
        test_loader = DataLoader(TestDataset(), batch_size=64, shuffle=False)
        metrics = trainer.evaluate(test_loader)
        accs.append(metrics['accuracy'])
        aucs.append(metrics['auc_score'])
        all_seed_metrics.append(metrics)

        # Inference scores for plots
        scores = trainer.predict(dna_te, pep_te)
        y_true = np.array(y_te)

        seed_dir = fig_dir/f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        plot_roc_pr(y_true, scores, seed_dir, label=f"seed {seed}")
        plot_confusion(y_true, (scores>=0.5).astype(int), seed_dir/"confusion.png")
        threshold_sweep(y_true, scores, seed_dir/"threshold_sweep.png")
        calibration_plot(y_true, scores, seed_dir/"calibration.png")
        score_distributions(y_true, scores, seed_dir/"score_dist.png")
        length_effects(dna_te, scores, seed_dir/"len_dna.png", title="DNA length vs score")
        length_effects(pep_te, scores, seed_dir/"len_pep.png", title="Peptide length vs score")

        # Save per-seed metrics
        with open(tab_dir/f"metrics_seed_{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Aggregate report
    def mstd(x):
        return float(np.mean(x)), float(np.std(x))
    acc_m, acc_s = mstd(accs)
    auc_m, auc_s = mstd(aucs)
    summary = {
        'seeds': num_seeds,
        'accuracy_mean': acc_m,
        'accuracy_std': acc_s,
        'auc_mean': auc_m,
        'auc_std': auc_s,
        'per_seed': all_seed_metrics,
    }
    with open(tab_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", summary)


if __name__ == "__main__":
    main(num_seeds=5)


