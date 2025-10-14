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
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

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

# Allow opt-out of TensorFlow baselines to avoid import-time hangs with Abseil locks.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s'
)
logger = logging.getLogger('baseline')


def load_df(data_path: str) -> pd.DataFrame:
    """Load dataframe with error handling."""
    p = Path(data_path)

    if not p.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    try:
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
    except Exception as e:
        raise SystemExit(f"Failed to load data from {data_path}: {e}")

    if df.empty:
        raise SystemExit("Loaded dataframe is empty")

    if len(df) < 10:
        logger.warning(f"Dataset is very small ({len(df)} samples). Results may be unreliable.")

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
        mapped = ser_str.map(mapping)

        # Check for unmapped values
        if mapped.isna().any():
            unmapped_count = mapped.isna().sum()
            # Try numeric conversion for unmapped values
            try:
                df[found] = pd.to_numeric(ser, errors="coerce").astype(int)
                if df[found].isna().any():
                    raise ValueError(f"{unmapped_count} labels could not be mapped to 0/1")
            except Exception:
                raise SystemExit(f"Could not convert all labels to binary (0/1). Found {unmapped_count} unmapped values.")
        else:
            df[found] = mapped.astype(int)

    # Rename to canonical 'label'
    if found != "label":
        df = df.rename(columns={found: "label"})

    return df


def detect_columns(df: pd.DataFrame, label_hint: List[str]) -> Tuple[List[str], List[str], List[str], str]:
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


def build_preprocessor(num_cols: List[str], cat_cols: List[str], text_cols: List[str], n_samples: int) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append(("cat", enc, cat_cols))
    # For text, build TF-IDF per text column
    # Adjust min_df based on dataset size to avoid dropping too many features
    min_df = max(1, min(2, n_samples // 100))
    for c in text_cols:
        transformers.append((
            f"tfidf_{c}",
            TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=min_df,
                max_features=5000
            ),
            c
        ))
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)


# PyTorch MLP Classifier
class PyTorchMLPClassifier:
    def __init__(self, input_dim=None, hidden_dims=[256, 128, 64], dropout=0.3,
                 lr=0.001, epochs=50, batch_size=32, seed=42, patience=5, device=None):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience
        self.model = None
        self.logger = logging.getLogger('mlp_pytorch')

        # Auto-detect best device: CUDA > CPU (skip MPS due to potential hangs)
        # Set PYTORCH_ENABLE_MPS_FALLBACK=1 environment variable to enable MPS
        use_mps = os.environ.get("PYTORCH_ENABLE_MPS", "0") == "1"

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info("Using CUDA device")
        elif use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                self.device = torch.device('mps')
                self.logger.info("Using MPS (Apple Silicon) device")
                # Test MPS with a small operation
                test_tensor = torch.zeros(1, device=self.device)
                del test_tensor
            except Exception as e:
                self.logger.warning(f"MPS initialization failed: {e}, falling back to CPU")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU device")

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
        # No sigmoid here - will be handled by BCEWithLogitsLoss / in predict_proba
        layers_list.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers_list).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        if not HAS_PYTORCH:
            raise RuntimeError("PyTorch not available")

        self.logger.info("Starting fit: preprocessing arrays for training")
        if isinstance(X, np.ndarray):
            X_dense = np.asarray(X, dtype=np.float32)
        else:
            X_dense = np.asarray(X.toarray(), dtype=np.float32)
        y_dense = np.asarray(y, dtype=np.float32)

        positive = int(np.sum(y_dense >= 0.5))
        negative = int(y_dense.size - positive)
        self.logger.info(f"Input shapes → X: {X_dense.shape}, y: {y_dense.shape}")
        self.logger.info(f"Class balance → pos: {positive}, neg: {negative}")

        # Calculate class weights for imbalanced datasets
        if positive > 0 and negative > 0:
            pos_weight = negative / positive
            self.logger.info(f"Using pos_weight={pos_weight:.2f} for class imbalance")
        else:
            pos_weight = 1.0

        X_tensor = torch.tensor(X_dense, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_dense, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Prepare validation data if provided
        if X_val is not None and y_val is not None:
            if isinstance(X_val, np.ndarray):
                X_val_dense = np.asarray(X_val, dtype=np.float32)
            else:
                X_val_dense = np.asarray(X_val.toarray(), dtype=np.float32)
            y_val_dense = np.asarray(y_val, dtype=np.float32)
            X_val_tensor = torch.tensor(X_val_dense, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val_dense, dtype=torch.float32, device=self.device).unsqueeze(1)

        if self.input_dim is None:
            self.input_dim = X_tensor.shape[1]

        self.model = self._build_model(self.input_dim)
        self.logger.info(f"Device: {self.device}, input_dim: {self.input_dim}, "
                        f"epochs: {self.epochs}, batch_size: {self.batch_size}")

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        self.model.train()

        for epoch in range(self.epochs):
            # Log progress every epoch
            if epoch == 0 or (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_loss = 0.0
            epoch_correct = 0
            samples_seen = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                # Remove sigmoid from model output, use BCEWithLogitsLoss
                outputs = self.model(batch_X)
                # Temporarily change last layer output
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                batch_size_actual = batch_y.size(0)
                epoch_loss += loss.item() * batch_size_actual
                samples_seen += batch_size_actual

                # For accuracy, apply sigmoid
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                epoch_correct += (preds == batch_y).float().sum().item()

            epoch_loss_avg = epoch_loss / max(1, len(dataset))
            epoch_acc = epoch_correct / max(1, len(dataset))

            # Validation
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                    val_preds = (torch.sigmoid(val_outputs) >= 0.5).float()
                    val_acc = (val_preds == y_val_tensor).float().mean().item()
                self.model.train()

                # Log only periodically to avoid spam
                if epoch == 0 or (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                    self.logger.info(f"  train_loss: {epoch_loss_avg:.4f}, train_acc: {epoch_acc:.4f}, "
                                   f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    # Restore best model
                    self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
                    break
            else:
                if epoch == 0 or (epoch + 1) % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1:
                    self.logger.info(f"  train_loss: {epoch_loss_avg:.4f}, train_acc: {epoch_acc:.4f}")

        self.logger.info("Training complete")

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_dense = np.asarray(X, dtype=np.float32)
            else:
                X_dense = np.asarray(X.toarray(), dtype=np.float32)
            X_tensor = torch.tensor(X_dense, dtype=torch.float32, device=self.device)
            logits = self.model(X_tensor)
            preds = torch.sigmoid(logits).cpu().numpy()
        return np.hstack([1 - preds, preds])


# Keras/TensorFlow MLP Classifier
class KerasMLPClassifier:
    def __init__(self, input_dim=None, hidden_dims=[256, 128, 64], dropout=0.3,
                 lr=0.001, epochs=50, batch_size=32, seed=42, patience=5):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience
        self.model = None
        self.logger = logging.getLogger('mlp_keras')

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

    def fit(self, X, y, X_val=None, y_val=None):
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")

        X_array = X if isinstance(X, np.ndarray) else X.toarray()

        # Calculate class weights for imbalanced datasets
        positive = int(np.sum(y >= 0.5))
        negative = int(len(y) - positive)
        if positive > 0 and negative > 0:
            class_weight = {0: 1.0, 1: negative / positive}
            self.logger.info(f"Using class_weight for class 1: {class_weight[1]:.2f}")
        else:
            class_weight = None

        if self.input_dim is None:
            self.input_dim = X_array.shape[1]

        self.model = self._build_model(self.input_dim)

        class _LogCallback(keras.callbacks.Callback):
            def __init__(self, logger, epochs: int):
                super().__init__()
                self._logger = logger
                self._epochs = epochs

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                # Log only periodically
                if epoch == 0 or (epoch + 1) % max(1, self._epochs // 10) == 0 or epoch + 1 == self._epochs:
                    metrics = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])
                    self._logger.info(f"Epoch {epoch + 1}/{self._epochs} - {metrics}")

        callbacks = [_LogCallback(self.logger, self.epochs)]

        # Add early stopping with validation data
        if X_val is not None and y_val is not None:
            X_val_array = X_val if isinstance(X_val, np.ndarray) else X_val.toarray()
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=0
            ))
            validation_data = (X_val_array, y_val)
        else:
            validation_data = None

        self.logger.info(f"input_dim={self.input_dim}, epochs={self.epochs}, batch_size={self.batch_size}")
        self.model.fit(
            X_array, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_data=validation_data,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained")

        X_array = X if isinstance(X, np.ndarray) else X.toarray()
        preds = self.model.predict(X_array, verbose=0)
        return np.hstack([1 - preds, preds])


def get_models(seed: int):
    # Use class_weight='balanced' for sklearn models that support it
    models = {
        "logreg": LogisticRegression(
            solver="saga", max_iter=5000, tol=1e-3, C=0.5,
            class_weight='balanced', random_state=seed
        ),
        "svm_linear": SVC(
            kernel='linear', probability=False, class_weight='balanced', random_state=seed
        ),
        "svm_rbf": SVC(
            kernel='rbf', probability=False, class_weight='balanced', random_state=seed
        ),
        "rf": RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=seed, n_jobs=-1
        ),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "dummy_majority": DummyClassifier(strategy='most_frequent')
    }
    if HAS_XGB:
        # XGBoost will use scale_pos_weight parameter, set in training
        models["xgb"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, eval_metric='logloss'
        )
    else:
        models["gb"] = GradientBoostingClassifier(random_state=seed)

    # Add deep learning models with consistent hyperparameters
    if HAS_PYTORCH:
        models["mlp_pytorch"] = PyTorchMLPClassifier(
            hidden_dims=[256, 128, 64], dropout=0.3, lr=0.0001,
            epochs=50, batch_size=32, seed=seed, patience=20
        )

    if HAS_TF:
        models["mlp_keras"] = KerasMLPClassifier(
            hidden_dims=[256, 128, 64], dropout=0.3, lr=0.0001,
            epochs=50, batch_size=32, seed=seed, patience=20
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
    ap.add_argument("--skip-dl", action="store_true", help="Skip deep learning models (faster)")
    ap.add_argument("--models", type=str, default=None, help="Comma-separated list of models to run (e.g., logreg,rf,xgb)")
    args = ap.parse_args()

    import sklearn
    logger.info(f"Running: {__file__}")
    logger.info(f"sklearn: {sklearn.__version__}")
    logger.info(f"PyTorch available: {HAS_PYTORCH}")
    logger.info(f"TensorFlow available: {HAS_TF}")
    logger.info(f"XGBoost available: {HAS_XGB}")

    df = load_df(args.data_path)
    logger.info(f"Loaded records: {len(df)}")
    label_hints = ["label", "antimicrobial_activity", "y", "target", "class"]
    num_cols, cat_cols, text_cols, label_col = detect_columns(df, label_hints)

    logger.info(f"Feature columns: {len(num_cols)} numeric, {len(cat_cols)} categorical, {len(text_cols)} text")

    y = df[label_col].astype(int).to_numpy()
    X = df.drop(columns=[label_col])

    # 70/15/15 split with fixed seed and stratification
    # First: hold out 15% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=args.seed, stratify=y
    )
    # Second: from the remaining 85%, take ~15% as validation -> 0.15 / 0.85 ≈ 0.176
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=args.seed, stratify=y_temp
    )

    logger.info(f"Split sizes - train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    logger.info(f"Train class distribution: {np.sum(y_train)}/{len(y_train)} positive")

    pre = build_preprocessor(num_cols, cat_cols, text_cols, len(y_train))
    all_models = get_models(args.seed)

    # Filter models based on arguments
    if args.skip_dl:
        models = {k: v for k, v in all_models.items() if k not in ['mlp_pytorch', 'mlp_keras']}
        logger.info("Skipping deep learning models")
    elif args.models:
        selected = [m.strip() for m in args.models.split(',')]
        models = {k: v for k, v in all_models.items() if k in selected}
        logger.info(f"Running only: {', '.join(models.keys())}")
    else:
        models = all_models

    logger.info(f"Training {len(models)} models")

    results = {}
    for name, clf in models.items():
        logger.info(f"▶ Training {name} ...")
        t0 = time.time()

        # Preprocess data once
        X_train_pre = pre.fit_transform(X_train)
        X_val_pre = pre.transform(X_val)
        X_test_pre = pre.transform(X_test)

        # For deep learning models, pass validation data
        if isinstance(clf, (PyTorchMLPClassifier, KerasMLPClassifier)):
            clf.fit(X_train_pre, y_train, X_val=X_val_pre, y_val=y_val)
        else:
            # Handle XGBoost scale_pos_weight
            if name == "xgb" and HAS_XGB:
                positive = int(np.sum(y_train))
                negative = int(len(y_train) - positive)
                if positive > 0:
                    scale_pos_weight = negative / positive
                    clf.set_params(scale_pos_weight=scale_pos_weight)
            clf.fit(X_train_pre, y_train)

        logger.info(f"✓ Finished {name} in {time.time() - t0:.1f}s")

        # Helper to get probabilistic scores in [0,1]
        def _scores(clf_model, X_transformed):
            # Handle deep learning models (PyTorch/Keras)
            if isinstance(clf_model, (PyTorchMLPClassifier, KerasMLPClassifier)):
                return clf_model.predict_proba(X_transformed)[:, 1]

            # Prefer probabilistic output if available
            if hasattr(clf_model, "predict_proba"):
                try:
                    return clf_model.predict_proba(X_transformed)[:, 1]
                except Exception:
                    # Fall through to decision_function if predict_proba fails
                    pass

            if hasattr(clf_model, "decision_function"):
                s = clf_model.decision_function(X_transformed)
                s_min, s_max = s.min(), s.max()
                return (s - s_min) / (s_max - s_min + 1e-9)

            # Last resort: use class predictions as scores
            return clf_model.predict(X_transformed).astype(float)

        scores_val = _scores(clf, X_val_pre)
        scores_test = _scores(clf, X_test_pre)

        results[name] = {
            "val": evaluate(y_val, scores_val),
            "test": evaluate(y_test, scores_test),
        }
        logger.info(f"{name} - val_f1: {results[name]['val']['f1_score']:.4f}, "
                   f"test_f1: {results[name]['test']['f1_score']:.4f}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to: {out_path}")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
