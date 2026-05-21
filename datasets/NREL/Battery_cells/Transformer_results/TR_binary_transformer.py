from __future__ import annotations
import os
import re
import argparse
from pathlib import Path
from typing import Literal, Dict, Any, Callable, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precisio_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precisio_recall_fscore_support, confusion_matrix

Pipeline([Random])

def find_files(dataset_name, severity_level):
    stem = f"{dataset_name}.xlsx_reform_severity_level{severity_level}.csv_rt_detect_result_window100.csv"
    p = DATA_DIR/stem
    if p.exists():
        return p
    alt_name = dataset_name.replace('Soteria')


def normalize_dataset_name(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"[\s\-]", "_", name.lower())


def find_rt_detect_file(dataset_name: str, severity_level: int) -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    # Try exact match first
    stem = f"{dataset_name}.xlsx_reform_severity_level{severity_level}.csv_rt_detect_result_window100.csv"
    p = DATA_DIR / stem
    if p.exists():
        return p
    # Try with Sorteria->Soteria and other variants
    alt_name = dataset_name.replace("Sorteria", "Soteria").replace("Soetria", "Soteria")
    if alt_name != dataset_name:
        p2 = DATA_DIR / f"{alt_name}.xlsx_reform_severity_level{severity_level}.csv_rt_detect_result_window100.csv"
        if p2.exists():
            return p2
    # Scan files: match by normalized name substring and severity
    base_norm = normalize_dataset_name(dataset_name)
    base_dash = re.sub(r"_+", "-", dataset_name.replace(" ", ""))
    for f in DATA_DIR.glob("*rt_detect_result_window100.csv"):
        if f"severity_level{severity_level}" not in f.name:
            continue
        fnorm = normalize_dataset_name(f.stem.split(".xlsx")[0])
        if base_norm in fnorm or base_dash.lower() in f.name.lower():
            return f
    return None


def load_temperature_series(path: Path, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Load temperature column, return array of shape (max_len,) padded/truncated."""
    df = pd.read_csv(path)
    if "temperature" not in df.columns:
        return np.full(max_len, np.nan)
    temp = df["temperature"].astype(float).dropna().values
    if len(temp) == 0:
        return np.full(max_len, np.nan)
    # Use first max_len points (early behavior often discriminative)
    temp = temp[:max_len]
    if len(temp) < max_len:
        temp = np.pad(temp, (0, max_len - len(temp)), constant_values=temp[-1] if len(temp) > 0 else np.nan)
    return temp.astype(np.float32)


def load_labels_and_metadata() -> pd.DataFrame:
    """Load battery_class_results with dataset_name, Label, and severity if available."""
    for p in [BATTERY_CLASS_PATH, SENSITIVITY_CLASS_PATH]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("battery_class_results.csv not found")
    # Keep rows with valid Label
    if "Label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "Label"})
    df = df[df["Label"].isin([0, 1])].copy()
    df = df.iloc[:118]
    # Get severity from dataset_correspondence if available
    corr_path = BASE_DIR / "dataset_correspondence.xlsx"
    if corr_path.exists():
        try:
            corr = pd.read_excel(corr_path)
            if "dataset_name" in corr.columns and "severity_level" in corr.columns:
                df = df.merge(
                    corr[["dataset_name", "severity_level"]],
                    on="dataset_name",
                    how="left"
                )
        except Exception:
            pass
    # If no severity, try to infer from filename patterns in NREL_data_results
    if "severity_level" not in df.columns or df["severity_level"].isna().all():
        df["severity_level"] = 2  # default
        if DATA_DIR.exists():
            for i, row in df.iterrows():
                dn = row["dataset_name"]
                for sev in [1, 2, 3, 4, 5, 6, 7]:
                    if find_rt_detect_file(dn, sev):
                        df.at[i, "severity_level"] = sev
                        break
    return df


class BatteryDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = sequences  # (N, max_len)
        self.y = labels    # (N,)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x.unsqueeze(-1), y  # (max_len, 1), scalar


# ---------- Model ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BatteryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        self.d_model = d_model
    def forward(self, x):
        # x: (B, T, 1)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)  # (B, d_model)
        return self.cls(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def run_kfold(X: np.ndarray, y: np.ndarray) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    all_preds, all_labels = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # Handle NaN: fill with column mean
        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            mn = np.nanmean(col)
            if np.isnan(mn):
                mn = 25.0
            X_train[np.isnan(X_train[:, i]), i] = mn
            X_val[np.isnan(X_val[:, i]), i] = mn
        train_ds = BatteryDataset(X_train, y_train)
        val_ds = BatteryDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        model = BatteryTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device)
            p, l = evaluate(model, val_loader, device)
            acc = accuracy_score(l, p)
            if acc > best_acc:
                best_acc = acc
        p, l = evaluate(model, val_loader, device)
        fold_results.append({
            "fold": fold,
            "accuracy": accuracy_score(l, p),
            "precision": precision_recall_fscore_support(l, p, average="binary")[0],
            "recall": precision_recall_fscore_support(l, p, average="binary")[1],
            "f1": precision_recall_fscore_support(l, p, average="binary")[2],
        })
        all_preds.append(p)
        all_labels.append(l)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "fold_results": fold_results,
        "overall_accuracy": accuracy_score(all_labels, all_preds),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "precision_recall_f1": precision_recall_fscore_support(all_labels, all_preds, average="binary"),
    }






























