#!/usr/bin/env python3
"""
Generic baseline runner for binary classification datasets (rewritten so the
PyTorch MLP works cleanly inside an sklearn Pipeline).

Key changes vs. your original:
- Implemented a sklearn-compatible PyTorchMLPClassifier (inherits BaseEstimator,
  ClassifierMixin; exposes get_params/set_params; defines predict, predict_proba,
  classes_ after fit).
- Robust handling of sparse inputs (TF-IDF) by densifying only for the PyTorch
  model; other models still operate on sparse matrices.
- Trimmed training logs and ensured deterministic seeding.
- Kept your model zoo (SVMs, RF, XGBoost/GB, kNN, Dummy, Keras if available).

CLI:
  --data_path <csv|json>
  --out_path  <json>
  --seed      <int> (default 42)

Automatically:
  - Detect text/categorical/numeric columns
  - TF-IDF for text; Ordinal for categorical; Standardize numeric
  - Stratified 70/15/15 split (train/val/test)
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_PYTORCH = True
except Exception:
    HAS_PYTORCH = False

# TensorFlow (optional, can be disabled to avoid import-time issues)
DISABLE_TF = os.environ.get("BASELINES_DISABLE_TF", "").strip().lower() in {"1", "true", "yes"}
if not DISABLE_TF:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        HAS_TF = True
    except Exception:
        HAS_TF = False
else:
    HAS_TF = False


# ------------------------------- IO -----------------------------------------

def load_df(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix.lower() == ".json":
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("JSON must be an array of records (list of objects)")
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(p)

    # Normalize label column to numeric 0/1 and canonical name 'label'
    label_candidates = ["label", "Label", "antimicrobial_activity", "y", "target", "class"]
    found = None
    for c in label_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        for c in df.columns:
            ser = df[c]
            vals = ser.dropna()
            if len(vals) > 0 and vals.isin(
                [0, 1, True, False, "0", "1", "yes", "no", "Yes", "No", "true", "false", "True", "False"]
            ).all():
                found = c
                break
    if found is None:
        raise SystemExit(
            "Could not find a label column in the data. Provide one of: " + ", ".join(label_candidates)
        )

    ser = df[found]
    if ser.dtype == bool:
        df[found] = ser.astype(int)
    else:
        ser_str = ser.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "positive": 1,
            "negative": 0,
        }
        try:
            df[found] = ser_str.map(mapping).fillna(ser).astype(int)
        except Exception:
            df[found] = pd.to_numeric(ser, errors="raise").astype(int)

    if found != "label":
        df = df.rename(columns={found: "label"})

    if "micoorganism_sequence" in df.columns and "microorganism_sequence" not in df.columns:
        df = df.rename(columns={"micoorganism_sequence": "microorganism_sequence"})

    return df


def detect_columns(df: pd.DataFrame, label_hint: List[str]) -> Tuple[list, list, list, str]:
    label_col = None
    for cand in label_hint:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        for c in df.columns:
            if c.lower() == "label":
                label_col = c
                break
    if label_col is None:
        raise SystemExit("Could not find a label column. Provide 'label' or one of: " + ", ".join(label_hint))

    feature_cols = [c for c in df.columns if c != label_col]
    obj_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in obj_cols]

    text_cols, cat_cols = [], []
    nrows = len(df)
    for c in obj_cols:
        uniq = df[c].astype(str).nunique(dropna=False)
        avglen = df[c].astype(str).map(len).mean()
        if uniq / max(1, nrows) > 0.2 or avglen > 20:
            text_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols, text_cols, label_col


def build_preprocessor(num_cols: list, cat_cols: list, text_cols: list) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat", enc, cat_cols))
    for c in text_cols:
        transformers.append((
            f"tfidf_{c}",
            TfidfVectorizer(ngram_range=(1, 3), min_df=2),
            c,
        ))
    # sparse_threshold=0.3 keeps outputs sparse unless dense share >70%
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


# ----------------------------- PyTorch MLP -----------------------------------

class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible MLP binary classifier using PyTorch.

    Works with sparse or dense numpy arrays. If X is sparse, it is densified
    internally (only for the PyTorch model). Runs on CPU by default to avoid
    backend issues on macOS Metal.
    """

    def __init__(
        self,
        input_dim=None,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        lr=1e-3,
        epochs=5,
        batch_size=16,
        seed=42,
        verbose=False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.model_ = None
        self.classes_ = np.array([0, 1], dtype=int)
        self.device_ = torch.device("cpu") if HAS_PYTORCH else None

    # --- sklearn API ---
    def fit(self, X, y):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch not available")

        y = np.asarray(y).astype(np.float32)
        X_dense = self._to_dense_float32(X)

        # Seed everything for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.input_dim is None:
            self.input_dim = int(X_dense.shape[1])

        self.model_ = self._build_model(self.input_dim)

        X_tensor = torch.from_numpy(X_dense).to(self.device_)
        y_tensor = torch.from_numpy(y.reshape(-1, 1)).to(self.device_)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            if self.verbose:
                print(f"[mlp_pytorch] epoch {epoch+1}/{self.epochs} loss={epoch_loss/len(dataset):.4f}")

        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")
        X_dense = self._to_dense_float32(X)
        X_tensor = torch.from_numpy(X_dense).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_tensor).cpu().numpy().reshape(-1)
        preds = np.clip(preds, 0.0, 1.0)
        return np.vstack([1.0 - preds, preds]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    # --- internals ---
    def _build_model(self, input_dim: int) -> nn.Module:
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        return nn.Sequential(*layers).to(self.device_)

    @staticmethod
    def _to_dense_float32(X) -> np.ndarray:
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32, order="C")


# -------------------------- Keras (optional) ---------------------------------

class KerasMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim=None,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        lr=1e-3,
        epochs=50,
        batch_size=32,
        seed=42,
        verbose=False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.model_ = None
        self.classes_ = np.array([0, 1], dtype=int)

    def _build_model(self, input_dim):
        tf.random.set_seed(self.seed)
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for h in self.hidden_dims:
            model.add(layers.Dense(h, activation="relu"))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def fit(self, X, y):
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")
        X_arr = X if isinstance(X, np.ndarray) else X.toarray()
        if self.input_dim is None:
            self.input_dim = X_arr.shape[1]
        self.model_ = self._build_model(self.input_dim)
        self.model_.fit(X_arr, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=1 if self.verbose else 0, validation_split=0.1)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not trained")
        X_arr = X if isinstance(X, np.ndarray) else X.toarray()
        preds = self.model_.predict(X_arr, verbose=0).reshape(-1)
        preds = np.clip(preds, 0.0, 1.0)
        return np.vstack([1.0 - preds, preds]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# ------------------------------- Zoo -----------------------------------------

def get_models(seed: int):
    models = {
        "logreg": LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, C=0.5, random_state=seed),
        "svm_linear": SVC(kernel="linear", probability=False, random_state=seed),
        "svm_rbf": SVC(kernel="rbf", probability=False, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        models["gb"] = GradientBoostingClassifier(random_state=seed)

    if HAS_PYTORCH:
        models["mlp_pytorch"] = PyTorchMLPClassifier(
            hidden_dims=(256, 128, 64), dropout=0.3, lr=1e-3, epochs=5, batch_size=16, seed=seed, verbose=False
        )

    if HAS_TF:
        models["mlp_keras"] = KerasMLPClassifier(
            hidden_dims=(256, 128, 64), dropout=0.3, lr=1e-3, epochs=50, batch_size=32, seed=seed, verbose=False
        )

    return models


# ------------------------------ Metrics --------------------------------------

def evaluate(y_true, scores) -> dict:
    y_pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
    }


# ------------------------------- Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, type=str)
    ap.add_argument("--out_path", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import sklearn
    print("[baseline] running:", __file__)
    print("[baseline] sklearn:", sklearn.__version__)
    print(f"[baseline] PyTorch available: {HAS_PYTORCH}")
    print(f"[baseline] TensorFlow available: {HAS_TF}")
    print(f"[baseline] XGBoost available: {HAS_XGB}")

    df = load_df(args.data_path)
    print("[baseline] loaded records:", len(df))
    label_hints = ["label", "antimicrobial_activity", "y", "target", "class"]
    num_cols, cat_cols, text_cols, label_col = detect_columns(df, label_hints)

    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col])

    # 70/15/15 split with fixed seed and stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.17647058823529413, random_state=args.seed, stratify=y_temp
    )

    print("[baseline] split sizes:", {"train": len(y_train), "val": len(y_val), "test": len(y_test)})

    pre = build_preprocessor(num_cols, cat_cols, text_cols)
    models = get_models(args.seed)

    results = {}

    for name, clf in models.items():
        print(f"[baseline] ▶ Training {name} ...")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        # unified scorer (avoids Pipeline.predict_proba for 3rd-party quirks)
        def _scores(model, X_):
            pre_ = model.named_steps["pre"]
            clf_ = model.named_steps["clf"]
            X_tr = pre_.transform(X_)
            # Deep models
            if isinstance(clf_, (PyTorchMLPClassifier, KerasMLPClassifier)):
                return clf_.predict_proba(X_tr)[:, 1]
            if hasattr(clf_, "predict_proba"):
                try:
                    return clf_.predict_proba(X_tr)[:, 1]
                except Exception:
                    pass
            if hasattr(clf_, "decision_function"):
                s = clf_.decision_function(X_tr)
                s_min, s_max = s.min(), s.max()
                return (s - s_min) / (s_max - s_min + 1e-9)
            return clf_.predict(X_tr).astype(float)

        scores_val = _scores(pipe, X_val)
        scores_test = _scores(pipe, X_test)
        results[name] = {"val": evaluate(y_val, scores_val), "test": evaluate(y_test, scores_test)}
        print(f"[baseline] ✓ Finished {name}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("[baseline] Writing results to:", out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
