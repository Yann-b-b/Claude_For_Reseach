#!/usr/bin/env python3
"""
Generic baseline runner for binary classification datasets.

Input CSV/JSON must contain feature columns (text/categorical/numeric) and a binary label column.

CLI:
  --data_path <csv|json>
  --out_path  <json>
  --seed      <int> (default 42)

Models:
  - Logistic Regression
  - SVM (linear)
  - SVM (rbf)
  - Random Forest
  - Gradient Boosted Trees (XGBoost if available, else sklearn GradientBoosting)
  - k-Nearest Neighbors
  - Dummy (most frequent)
  - Deep Neural Network (MLP with PyTorch)
  - Deep Neural Network (Keras/TensorFlow)
  - 1D CNN (for text/sequence features)

Automatically:
  - Detect text/categorical/numeric columns
  - TF-IDF for text; Ordinal for categorical; Standardize numeric
  - Stratified 80/20 split; default hyperparameters
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from tqdm import tqdm
import time

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_PYTORCH = True
except Exception:
    HAS_PYTORCH = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    HAS_TF = False


def load_df(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if p.suffix.lower() == ".json":
        # Load array-of-objects JSON
        with open(p, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("JSON must be an array of records (list of objects)")
        df = pd.DataFrame(data)
    else:
        # Default to CSV
        df = pd.read_csv(p)

    # Normalize label column to numeric 0/1 and a canonical name 'label'
    label_candidates = [
        "label", "Label", "antimicrobial_activity", "y", "target", "class"
    ]
    found = None
    for c in label_candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        # Try to infer if a boolean-like column exists
        for c in df.columns:
            if df[c].dropna().isin([0, 1, True, False, "0", "1",
                                    "yes", "no", "Yes", "No",
                                    "true", "false", "True", "False"]).all():
                found = c
                break
    if found is None:
        raise SystemExit("Could not find a label column in the data. Provide one of: " + ", ".join(label_candidates))

    # Map common string/boolean labels to {0,1}
    ser = df[found]
    if ser.dtype == bool:
        df[found] = ser.astype(int)
    else:
        ser_str = ser.astype(str).str.strip().str.lower()
        mapping = {"1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0, "positive": 1, "negative": 0}
        try:
            df[found] = ser_str.map(mapping).fillna(ser).astype(int)
        except Exception:
            # If it was already numeric-like, coerce
            df[found] = pd.to_numeric(ser, errors="raise").astype(int)

    # Rename to canonical 'label'
    if found != "label":
        df = df.rename(columns={found: "label"})

    # Optional: fix common key typo from sample JSON
    if "micoorganism_sequence" in df.columns and "microorganism_sequence" not in df.columns:
        df = df.rename(columns={"micoorganism_sequence": "microorganism_sequence"})

    return df


def detect_columns(df: pd.DataFrame, label_hint: List[str]) -> tuple[list, list, list, str]:
    # Determine label column
    label_col = None
    for cand in label_hint:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        for c in df.columns:
            if c.lower() == 'label':
                label_col = c
                break
    if label_col is None:
        raise SystemExit("Could not find a label column. Provide a 'label' column or one of: " + ", ".join(label_hint))

    feature_cols = [c for c in df.columns if c != label_col]
    obj_cols = [c for c in feature_cols if df[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in obj_cols]

    # Heuristic: long/unique object columns → text; else categorical
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
    # For text, build TF-IDF per text column
    for c in text_cols:
        transformers.append((f"tfidf_{c}", TfidfVectorizer(ngram_range=(1, 3), min_df=2), c))
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)


# PyTorch MLP Classifier
class PyTorchMLPClassifier:
    def __init__(self, input_dim=None, hidden_dims=[256, 128, 64], dropout=0.3,
                 lr=0.001, epochs=50, batch_size=32, seed=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim):
        torch.manual_seed(self.seed)
        layers_list = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers_list.append(nn.Linear(prev_dim, hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers_list.append(nn.Linear(prev_dim, 1))
        layers_list.append(nn.Sigmoid())
        return nn.Sequential(*layers_list).to(self.device)

    def fit(self, X, y):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch not available")

        X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.toarray()).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        if self.input_dim is None:
            self.input_dim = X_tensor.shape[1]

        self.model = self._build_model(self.input_dim)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X if isinstance(X, np.ndarray) else X.toarray()).to(self.device)
            preds = self.model(X_tensor).cpu().numpy()
        return np.hstack([1 - preds, preds])


# Keras/TensorFlow MLP Classifier
class KerasMLPClassifier:
    def __init__(self, input_dim=None, hidden_dims=[256, 128, 64], dropout=0.3,
                 lr=0.001, epochs=50, batch_size=32, seed=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.model = None

    def _build_model(self, input_dim):
        tf.random.set_seed(self.seed)
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        for hidden_dim in self.hidden_dims:
            model.add(layers.Dense(hidden_dim, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, X, y):
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")

        X_array = X if isinstance(X, np.ndarray) else X.toarray()

        if self.input_dim is None:
            self.input_dim = X_array.shape[1]

        self.model = self._build_model(self.input_dim)
        self.model.fit(
            X_array, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1
        )
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")

        X_array = X if isinstance(X, np.ndarray) else X.toarray()
        preds = self.model.predict(X_array, verbose=0)
        return np.hstack([1 - preds, preds])


def get_models(seed: int):
    models = {
        "logreg": LogisticRegression(solver="saga", max_iter=5000, tol=1e-3, C=0.5, random_state=seed),
        "svm_linear": SVC(kernel='linear', probability=False, random_state=seed),
        "svm_rbf": SVC(kernel='rbf', probability=False, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "dummy_majority": DummyClassifier(strategy='most_frequent')
    }
    if HAS_XGB:
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
    else:
        models["gb"] = GradientBoostingClassifier(random_state=seed)

    # Add deep learning models
    if HAS_PYTORCH:
        models["mlp_pytorch"] = PyTorchMLPClassifier(
            hidden_dims=[256, 128, 64], dropout=0.3, lr=0.001,
            epochs=50, batch_size=32, seed=seed
        )

    if HAS_TF:
        models["mlp_keras"] = KerasMLPClassifier(
            hidden_dims=[256, 128, 64], dropout=0.3, lr=0.001,
            epochs=50, batch_size=32, seed=seed
        )

    return models


def evaluate(y_true, scores) -> dict:
    y_pred = (scores >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float('nan')
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
    }


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
    # First: hold out 15% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    # Second: from the remaining 85%, take 15% as validation -> 0.15 / 0.85 ≈ 0.176470588
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.17647058823529413, random_state=args.seed, stratify=y_temp
    )

    print("[baseline] split sizes:", {
        "train": len(y_train), "val": len(y_val), "test": len(y_test)
    })

    pre = build_preprocessor(num_cols, cat_cols, text_cols)
    models = get_models(args.seed)

    results = {}
    for name, clf in tqdm(models.items(), desc="[baseline] Training models", unit="model"):
        tqdm.write(f"[baseline] ▶ Training {name} ...")
        t0 = time.time()
        pipe = Pipeline([
            ("pre", pre),
            ("clf", clf)
        ])
        if name in ["svm_linear", "svm_rbf"]:
            tqdm.write(f"[baseline][DEBUG] Starting fit for {name} ...")
            t_fit_start = time.time()
        pipe.fit(X_train, y_train)
        if name in ["svm_linear", "svm_rbf"]:
            tqdm.write(f"[baseline][DEBUG] Finished fit for {name} in {time.time() - t_fit_start:.2f}s")
            tqdm.write(f"[baseline][DEBUG] Now scoring {name} on validation set ...")
        tqdm.write(f"[baseline] ✓ Finished {name} in {time.time() - t0:.1f}s")
        # Helper to get probabilistic scores in [0,1]
        def _scores(model, X_):
            # Compute scores without invoking Pipeline.predict_proba to avoid
            # scikit-learn 1.6.x third-party __sklearn_tags__ compatibility issues
            # (e.g., with some xgboost versions).
            pre = model.named_steps['pre']
            clf = model.named_steps['clf']
            X_tr = pre.transform(X_)

            # Handle deep learning models (PyTorch/Keras)
            if isinstance(clf, (PyTorchMLPClassifier, KerasMLPClassifier)):
                return clf.predict_proba(X_tr)[:, 1]

            # Prefer probabilistic output if available
            if hasattr(clf, "predict_proba"):
                try:
                    return clf.predict_proba(X_tr)[:, 1]
                except Exception:
                    # Fall through to decision_function if predict_proba fails
                    pass

            if hasattr(clf, "decision_function"):
                s = clf.decision_function(X_tr)
                s_min, s_max = s.min(), s.max()
                return (s - s_min) / (s_max - s_min + 1e-9)

            # Last resort: use class predictions as scores
            return clf.predict(X_tr).astype(float)

        if name in ["svm_linear", "svm_rbf"]:
            t_val_start = time.time()
        scores_val = _scores(pipe, X_val)
        if name in ["svm_linear", "svm_rbf"]:
            tqdm.write(f"[baseline][DEBUG] Validation scoring took {time.time() - t_val_start:.2f}s")
            tqdm.write(f"[baseline][DEBUG] Now scoring {name} on test set ...")
            t_test_start = time.time()
        scores_test = _scores(pipe, X_test)
        if name in ["svm_linear", "svm_rbf"]:
            tqdm.write(f"[baseline][DEBUG] Test scoring took {time.time() - t_test_start:.2f}s")
        results[name] = {
            "val": evaluate(y_val, scores_val),
            "test": evaluate(y_test, scores_test),
        }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("[baseline] Writing results to:", out_path)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()