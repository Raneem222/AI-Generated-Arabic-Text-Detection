import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ======================
# Load BERT Embeddings
# ======================

train_emb = np.load("data/bert/train_embeddings.npy")
val_emb = np.load("data/bert/val_embeddings.npy")
test_emb = np.load("data/bert/test_embeddings.npy")

# Load labels
train_df = pd.read_csv("data/processed/train_clean.csv")
val_df = pd.read_csv("data/processed/val_clean.csv")
test_df = pd.read_csv("data/processed/test_clean.csv")

y_train = (train_df["label"] == "AI").astype(int).values
y_val = (val_df["label"] == "AI").astype(int).values
y_test = (test_df["label"] == "AI").astype(int).values

# Convert to PyTorch tensors
X_train = torch.tensor(train_emb, dtype=torch.float32)
X_val = torch.tensor(val_emb, dtype=torch.float32)
X_test = torch.tensor(test_emb, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# ======================
# Neural Network Model
# ======================

class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = FFNN()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ======================
# Training Function
# ======================

def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/5, Loss = {epoch_loss:.4f}")

# ======================
# Evaluation Function
# ======================

def evaluate(model, loader, y_true):
    model.eval()
    preds = []

    with torch.no_grad():
        for X, _ in loader:
            out = model(X).squeeze()
            preds.extend(out.numpy())

    preds = np.array(preds)
    y_pred = (preds >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, preds),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

# ======================
# Run Training & Evaluation
# ======================

print("\n=== Training FFNN Model ===")
train_model()

print("\n=== Evaluating on Test Set ===")
results = evaluate(model, test_loader, y_test)

print(results)

pd.DataFrame([results]).to_csv("reports/models/ffnn_results.csv", index=False)
print("\nFFNN Model Completed Successfully!")
