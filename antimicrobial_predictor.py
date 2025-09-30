import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from tqdm import tqdm

class SequenceEncoder:
    """Encodes DNA and protein sequences into numerical representations"""

    def __init__(self):
        # DNA nucleotide mapping
        self.dna_vocab = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 0}  # N for unknown

        # Amino acid mapping (20 standard amino acids + unknown)
        self.aa_vocab = {
            'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8,
            'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
            'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'X': 0  # X for unknown
        }

    def encode_dna(self, sequence: str, max_length: int = 50) -> np.ndarray:
        """Encode DNA sequence to numerical array"""
        sequence = sequence.upper()
        encoded = [self.dna_vocab.get(base, 0) for base in sequence]

        # Pad or truncate to max_length
        if len(encoded) < max_length:
            encoded.extend([0] * (max_length - len(encoded)))
        else:
            encoded = encoded[:max_length]

        return np.array(encoded, dtype=np.int64)

    def encode_protein(self, sequence: str, max_length: int = 100) -> np.ndarray:
        """Encode protein sequence to numerical array"""
        sequence = sequence.upper()
        encoded = [self.aa_vocab.get(aa, 0) for aa in sequence]

        # Pad or truncate to max_length
        if len(encoded) < max_length:
            encoded.extend([0] * (max_length - len(encoded)))
        else:
            encoded = encoded[:max_length]

        return np.array(encoded, dtype=np.int64)

class AntimicrobialDataset(Dataset):
    """Dataset class for antimicrobial prediction"""

    def __init__(self, dna_sequences: List[str], protein_sequences: List[str],
                 labels: List[int], encoder: SequenceEncoder):
        self.dna_sequences = dna_sequences
        self.protein_sequences = protein_sequences
        self.labels = labels
        self.encoder = encoder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dna_encoded = self.encoder.encode_dna(self.dna_sequences[idx])
        protein_encoded = self.encoder.encode_protein(self.protein_sequences[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'dna': torch.tensor(dna_encoded, dtype=torch.long),
            'protein': torch.tensor(protein_encoded, dtype=torch.long),
            'label': label
        }

class AntimicrobialPredictor(nn.Module):
    """Deep learning model for predicting antimicrobial activity"""

    def __init__(self, dna_vocab_size: int = 5, protein_vocab_size: int = 21,
                 dna_embedding_dim: int = 64, protein_embedding_dim: int = 128,
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.3):
        super(AntimicrobialPredictor, self).__init__()

        # Embedding layers
        self.dna_embedding = nn.Embedding(dna_vocab_size, dna_embedding_dim, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, protein_embedding_dim, padding_idx=0)

        # CNN layers for DNA
        self.dna_conv1 = nn.Conv1d(dna_embedding_dim, 128, kernel_size=7, padding=3)
        self.dna_conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.dna_pool = nn.AdaptiveMaxPool1d(1)

        # LSTM layers for protein
        self.protein_lstm = nn.LSTM(protein_embedding_dim, hidden_dim//2,
                                   num_layers=num_layers, batch_first=True,
                                   bidirectional=True, dropout=dropout if num_layers > 1 else 0)

        # Attention mechanism for protein sequences
        self.protein_attention = nn.MultiheadAttention(hidden_dim, num_heads=8,
                                                      dropout=dropout, batch_first=True)

        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(256 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, dna_seq, protein_seq):
        # DNA processing with CNN
        dna_emb = self.dna_embedding(dna_seq)  # (batch, seq_len, embed_dim)
        dna_emb = dna_emb.transpose(1, 2)  # (batch, embed_dim, seq_len)

        dna_conv1 = F.relu(self.dna_conv1(dna_emb))
        dna_conv2 = F.relu(self.dna_conv2(dna_conv1))
        dna_features = self.dna_pool(dna_conv2).squeeze(-1)  # (batch, 256)

        # Protein processing with LSTM and attention
        protein_emb = self.protein_embedding(protein_seq)  # (batch, seq_len, embed_dim)

        lstm_out, (hidden, _) = self.protein_lstm(protein_emb)  # (batch, seq_len, hidden_dim)

        # Self-attention on LSTM output
        attn_out, _ = self.protein_attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        mask = (protein_seq != 0).unsqueeze(-1).float()
        masked_attn = attn_out * mask
        protein_features = masked_attn.sum(1) / mask.sum(1).clamp(min=1)  # (batch, hidden_dim)

        # Feature fusion
        combined_features = torch.cat([dna_features, protein_features], dim=1)
        fused_features = self.fusion_layer(combined_features)

        # Classification
        output = self.classifier(fused_features)
        return output.squeeze(-1)

class AntimicrobialTrainer:
    """Training pipeline for the antimicrobial predictor"""

    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.encoder = SequenceEncoder()

    def prepare_data(self, dna_sequences: List[str], protein_sequences: List[str],
                    labels: List[int], test_size: float = 0.2, batch_size: int = 32):
        """Prepare data loaders for training and validation"""

        # Split data
        dna_train, dna_val, protein_train, protein_val, y_train, y_val = train_test_split(
            dna_sequences, protein_sequences, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Create datasets
        train_dataset = AntimicrobialDataset(dna_train, protein_train, y_train, self.encoder)
        val_dataset = AntimicrobialDataset(dna_val, protein_val, y_val, self.encoder)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, learning_rate: float = 0.001,
              weight_decay: float = 1e-5, early_stopping_patience: int = 10):
        """Train the model"""

        optimizer = torch.optim.Adam(self.model.parameters(),
                                   lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                dna_seq = batch['dna'].to(self.device)
                protein_seq = batch['protein'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(dna_seq, protein_seq)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    dna_seq = batch['dna'].to(self.device)
                    protein_seq = batch['protein'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(dna_seq, protein_seq)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_acc = accuracy_score(train_labels, np.round(train_preds))
            val_acc = accuracy_score(val_labels, np.round(val_preds))
            val_auc = roc_auc_score(val_labels, val_preds)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))

        return train_losses, val_losses

    def predict(self, dna_sequences: List[str], protein_sequences: List[str],
                batch_size: int = 32) -> np.ndarray:
        """Make predictions on new data"""

        # Create dataset without labels (dummy labels)
        dummy_labels = [0] * len(dna_sequences)
        dataset = AntimicrobialDataset(dna_sequences, protein_sequences, dummy_labels, self.encoder)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                dna_seq = batch['dna'].to(self.device)
                protein_seq = batch['protein'].to(self.device)

                outputs = self.model(dna_seq, protein_seq)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance"""

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                dna_seq = batch['dna'].to(self.device)
                protein_seq = batch['protein'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(dna_seq, protein_seq)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        binary_preds = np.round(predictions)

        accuracy = accuracy_score(true_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, binary_preds, average='binary')
        auc = roc_auc_score(true_labels, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc
        }

 

def plot_training_history(train_losses: List[float], val_losses: List[float]):
    """Plot training history"""

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_real_dataset():
    """Load the real antimicrobial dataset"""
    import pandas as pd
    import os

    # Check if combined processed dataset exists
    if os.path.exists('combined_amp_training_data.csv'):
        print("Loading combined processed dataset...")
        df = pd.read_csv('combined_amp_training_data.csv')
        return df['dna_sequence'].tolist(), df['protein_sequence'].tolist(), df['antimicrobial_activity'].tolist()
    elif os.path.exists('antimicrobial_training_data.csv'):
        print("Loading processed dataset...")
        df = pd.read_csv('antimicrobial_training_data.csv')
        return df['dna_sequence'].tolist(), df['protein_sequence'].tolist(), df['antimicrobial_activity'].tolist()
    # Fallback: look inside Dataset_and_train_sequence folder
    else:
        alt_path = os.path.join('Dataset_and_train_sequence', 'antimicrobial_training_data.csv')
        if os.path.exists(alt_path):
            print("Loading processed dataset from Dataset_and_train_sequence/...")
            df = pd.read_csv(alt_path)
            return df['dna_sequence'].tolist(), df['protein_sequence'].tolist(), df['antimicrobial_activity'].tolist()
        raise SystemExit("No dataset found. Provide a CSV with columns dna_sequence, protein_sequence, antimicrobial_activity.")

if __name__ == "__main__":
    # Load real dataset or fall back to synthetic data
    print("Loading dataset...")
    dna_sequences, protein_sequences, labels = load_real_dataset()

    print(f"Loaded {len(labels)} samples")
    print(f"Positive samples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")

    # Initialize model and trainer
    model = AntimicrobialPredictor()
    trainer = AntimicrobialTrainer(model)

    # Prepare data
    print("Preparing data...")
    train_loader, val_loader = trainer.prepare_data(dna_sequences, protein_sequences, labels)

    # Train model
    print("Training model...")
    train_losses, val_losses = trainer.train(train_loader, val_loader, num_epochs=10)

    # Evaluate model
    print("Evaluating model...")
    metrics = trainer.evaluate(val_loader)

    print("Final Validation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Example prediction
    sample_dna = dna_sequences[0]
    sample_protein = protein_sequences[0]
    sample_prediction = trainer.predict([sample_dna], [sample_protein])

    print(f"\nSample prediction:")
    print(f"DNA sequence length: {len(sample_dna)}")
    print(f"Protein sequence length: {len(sample_protein)}")
    print(f"Predicted antimicrobial probability: {sample_prediction[0]:.4f}")
    print(f"Actual label: {labels[0]}")