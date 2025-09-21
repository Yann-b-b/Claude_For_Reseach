### Antimicrobial Peptide–Bacteria Predictor (Claude Code Output)

#### Overview
Predicts whether a peptide kills a bacterium from two inputs: bacterial DNA sequence and peptide (protein) sequence.

#### Model Architecture
- **DNA branch (CNN)**: Embedding(5→64) → Conv1d 64→128 (k=7) → ReLU → Conv1d 128→256 (k=5) → ReLU → AdaptiveMaxPool1d(1) → 256-d.
- **Protein branch (BiLSTM + Attention)**: Embedding(21→128) → 2-layer bidirectional LSTM (hidden 512 total, dropout 0.3) → 8-head self-attention → masked global average pool → 512-d.
- **Fusion + Classifier**: Concat(256+512=768) → MLP 768→512→256 (ReLU + dropout) → 256→128→64→1 → Sigmoid.

#### Data
- Uses a **synthetic dataset** (for pipeline demonstration only):
  - Random DNA (≤1000 bp) and peptide sequences (≤200 aa).
  - Labels from peptide cationic fraction (R/K/H) + Gaussian noise; thresholded to 0/1.
  - Encoded with integer vocabularies and padding; stratified 80/20 split.

#### Training Pipeline
- Optimizer: Adam (lr 1e-3, weight_decay 1e-5); Loss: **BCELoss**.
- Regularization: dropout (0.3), gradient clipping (max_norm 1.0).
- Scheduler: ReduceLROnPlateau on val loss; Early stopping (patience 10).
- Metrics: accuracy, precision, recall, F1, ROC-AUC; best model saved to `best_model.pth`.

#### Results (synthetic validation)
- Early stopped at epoch 17.
- accuracy: 0.4900, precision: 0.4286, recall: 0.0600, F1: 0.1053, ROC-AUC: 0.5166.
- Interpretation: ~chance-level; synthetic/noisy labels are not biologically meaningful.

#### How to Run
```bash
pip install -r requirements.txt
python antimicrobial_predictor.py
```

#### Notes
- This setup validates the end-to-end pipeline. For real conclusions, replace the synthetic generator with a curated dataset of peptide–bacteria outcomes (e.g., DBAASP/APD3/DRAMP) and use leakage-aware splits.
