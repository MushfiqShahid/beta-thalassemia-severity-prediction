#!/usr/bin/env python3
"""
Main script for a Multi-Task Attention Transformer (MT-AT) project tailored to Beta-Thalassemia
  • Classification (Normal/Mild/Severe)
  • HbF_target regression

Usage:
  # Train:
python test2.py train   --train-data data/data2.csv   --val-data data/data2.csv   --out-dir saved_models   --epochs 40   --batch-size 32   --lr 1e-4   --seed 42

  # Interactive predict:
python test2.py predict-interactive --model-path saved_models/best_model.pt
"""

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)

FEATURE_KEYS = [
    "age", "sex", "hbb_mutation_type", "hbb_functional_score",
    "bcl11a_rs1427407_dosage", "hbs1l_myb_rs9399137_dosage",
    "klf1_variant", "hbg1_promoter_mut", "dna_methylation_hbg",
    "histone_acetylation", "mir_486_3p_level", "drug_treatment",
    "hbf_percent", "hemoglobin_g_dl", "transfusion_freq_per_year",
    "splenectomy", "ferritin_ng_ml",
]
CLASS_TARGET = "severity_label"
REG_TARGET = "HbF_target"


def derive_hbf_target_from_row(row: dict) -> float:
    if "hbf_percent" in row and isinstance(row["hbf_percent"], (int, float)):
        v = row["hbf_percent"] * 0.9
        return max(0.0, min(100.0, v))
    return 20.0


def ensure_numeric(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


class ThalDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.means = {k: 0.0 for k in FEATURE_KEYS}
        self.stds = {k: 1.0 for k in FEATURE_KEYS}
        self._compute_stats()

    def _compute_stats(self):
        arrays = {k: [] for k in FEATURE_KEYS}
        for row in self.records:
            for k in FEATURE_KEYS:
                v = row.get(k, 0.0)
                try:
                    v = float(v)
                except:
                    v = 0.0
                arrays[k].append(v)
        for k, vals in arrays.items():
            arr = np.array(vals, dtype=float)
            self.means[k] = float(arr.mean())
            self.stds[k] = float(arr.std()) if arr.std() > 1e-6 else 1.0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        x = []
        for k in FEATURE_KEYS:
            v = row.get(k, None)
            if v is None:
                if k == "hbf_percent":
                    v = derive_hbf_target_from_row(row)
                else:
                    v = 0.0
            else:
                try:
                    v = float(v)
                except:
                    v = 0.0
            x.append((v - self.means[k]) / self.stds[k])
        x = np.clip(np.array(x, dtype=np.float32), -10.0, 10.0)

        y_cls = int(row.get(CLASS_TARGET, 0))
        y_reg = row.get(REG_TARGET, None)
        if y_reg is None:
            y_reg = derive_hbf_target_from_row(row)
        else:
            try:
                y_reg = float(y_reg)
            except:
                y_reg = derive_hbf_target_from_row(row)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.long),
            torch.tensor(y_reg, dtype=torch.float32),
        )


class MTTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_layers=2, num_heads=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 3)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        for layer in self.enc_layers:
            h = layer(h)
        h = self.dropout(h.squeeze(1))
        logits = self.class_head(h)
        reg_out = self.reg_head(h).squeeze(1)
        return logits, reg_out


def compute_and_print_metrics(true_cls, pred_logits, true_reg, pred_reg):
    # Ensure all are numpy arrays on CPU
    y_true_np = np.array(true_cls)
    if isinstance(pred_logits, torch.Tensor):
        pred_probs = torch.softmax(pred_logits, dim=1).cpu().numpy()
        y_pred_cls = np.argmax(pred_probs, axis=1)
    else:
        pred_probs = pred_logits
        y_pred_cls = np.argmax(pred_probs, axis=1)

    y_true_reg_np = np.array(true_reg)
    y_pred_reg_np = np.array(pred_reg)

    acc = accuracy_score(y_true_np, y_pred_cls)
    prec_macro = precision_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    rec_macro = recall_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true_np, pred_probs, multi_class='ovr')
    except Exception:
        auc_roc = float('nan')

    mae = mean_absolute_error(y_true_reg_np, y_pred_reg_np)
    mse = mean_squared_error(y_true_reg_np, y_pred_reg_np)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_reg_np, y_pred_reg_np)
    cm = confusion_matrix(y_true_np, y_pred_cls)

    print("\n=== Prediction Performance Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall (macro): {rec_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"MAE (HbF target): {mae:.4f}")
    print(f"MSE (HbF target): {mse:.4f}")
    print(f"RMSE (HbF target): {rmse:.4f}")
    print(f"R² (HbF target): {r2:.4f}")
    print("Confusion Matrix:")
    print(cm)


def train_one_epoch(model, loader, optimizer, crit_cls, crit_reg, device, w_cls=1.0, w_reg=0.5):
    model.train()
    total_loss, batches = 0.0, 0
    for X, yc, yr in loader:
        X, yc, yr = X.to(device), yc.to(device), yr.to(device)
        optimizer.zero_grad()
        logits, pr = model(X)
        loss = w_cls * crit_cls(logits, yc) + w_reg * crit_reg(pr, yr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)


def evaluate(model, loader, device):
    model.eval()
    all_yc, all_logits, all_yr, all_pr = [], [], [], []
    with torch.no_grad():
        for X, yc, yr in loader:
            X = X.to(device)
            logits, pr = model(X)
            all_yc.append(yc.cpu())
            all_logits.append(logits.cpu())
            all_yr.append(yr.cpu())
            all_pr.append(pr.cpu())
    y_true_cls = torch.cat(all_yc)
    logits = torch.cat(all_logits)
    y_true_reg = torch.cat(all_yr)
    preds_reg = torch.cat(all_pr)
    # Return metrics dict without printing here (silent eval)
    y_true_np = y_true_cls.numpy()
    pred_probs = torch.softmax(logits, dim=1).numpy()
    y_pred_cls = np.argmax(pred_probs, axis=1)
    y_true_reg_np = y_true_reg.numpy()
    y_pred_reg_np = preds_reg.numpy()

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true_np, y_pred_cls)
    metrics["precision_macro"] = precision_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    metrics["recall_macro"] = recall_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    metrics["f1_macro"] = f1_score(y_true_np, y_pred_cls, average='macro', zero_division=0)
    try:
        metrics["auc_roc"] = roc_auc_score(y_true_np, pred_probs, multi_class='ovr')
    except:
        metrics["auc_roc"] = float("nan")
    metrics["mae"] = mean_absolute_error(y_true_reg_np, y_pred_reg_np)
    metrics["mse"] = mean_squared_error(y_true_reg_np, y_pred_reg_np)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["r2"] = r2_score(y_true_reg_np, y_pred_reg_np)
    metrics["confusion_matrix"] = confusion_matrix(y_true_np, y_pred_cls)
    return metrics


def load_json_or_csv(path):
    if path.endswith(".json") or path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        df = df.fillna(0)
        records = df.to_dict(orient="records")
        return records
    else:
        raise ValueError("Unsupported data format. Use JSON/JSONL/CSV.")


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_interactive(result, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    labels = ["Normal", "Mild", "Severe"]
    probs = [result["severity_probabilities"][l] for l in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color=["green", "orange", "red"])
    plt.ylabel("Probability")
    plt.title("Severity Probabilities")
    ppath = os.path.join(out_dir, "interactive_probs.png")
    plt.savefig(ppath)
    plt.close()
    print(f"Saved plot: {ppath}")


def get_interactive_input():
    prompts = {
        "age": ("Age (years)", 0, 120),
        "sex": ("Sex (0=F,1=M)", 0, 1),
        "hbb_mutation_type": ("HBB mutation type (0-1)", 0, 1),
        "hbb_functional_score": ("HBB functional score (0-1)", 0, 1),
        "bcl11a_rs1427407_dosage": ("BCL11A dosage (0-2)", 0, 2),
        "hbs1l_myb_rs9399137_dosage": ("HBS1L-MYB dosage (0-2)", 0, 2),
        "klf1_variant": ("KLF1 variant (0-1)", 0, 1),
        "hbg1_promoter_mut": ("HBG1 promoter mut (0-1)", 0, 1),
        "dna_methylation_hbg": ("DNA methylation HBG (0-1)", 0, 1),
        "histone_acetylation": ("Histone acetylation (0-1)", 0, 1),
        "mir_486_3p_level": ("miR-486-3p level (0-1)", 0, 1),
        "drug_treatment": ("Drug treatment (0-1)", 0, 1),
        "hbf_percent": ("HbF percent (0-100)", 0, 100),
        "hemoglobin_g_dl": ("Hemoglobin g/dL (0-20)", 0, 20),
        "transfusion_freq_per_year": ("Transf freq/year (0-52)", 0, 52),
        "splenectomy": ("Splenectomy (0=No,1=Yes)", 0, 1),
        "ferritin_ng_ml": ("Ferritin ng/mL (0-10000)", 0, 10000),
    }
    row = {}
    for k, (prompt, mn, mx) in prompts.items():
        while True:
            try:
                val = float(input(f"{prompt} [{mn}-{mx}]: "))
                if mn <= val <= mx:
                    row[k] = val
                    break
            except:
                print("Invalid input, try again.")
    return row


def format_severity(cls, probs):
    mapping = {0: "Normal", 1: "Mild", 2: "Severe"}
    out = f"\nPredicted Severity: {mapping[cls]}\n"
    for i, p in enumerate(probs):
        out += f"  {mapping[i]}: {p * 100:.1f}%\n"
    return out


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--train-data", required=True)
    train_p.add_argument("--val-data", required=True)
    train_p.add_argument("--out-dir", required=True)
    train_p.add_argument("--epochs", type=int, default=40)
    train_p.add_argument("--batch-size", type=int, default=32)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--seed", type=int, default=42)

    pred_p = sub.add_parser("predict-interactive")
    pred_p.add_argument("--model-path", required=True)

    args = parser.parse_args()

    if args.mode == "train":
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        train_records = load_json_or_csv(args.train_data)
        val_records = load_json_or_csv(args.val_data)

        train_ds = ThalDataset(train_records)
        val_ds = ThalDataset(val_records)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MTTransformer(input_dim=len(FEATURE_KEYS)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        crit_cls = nn.CrossEntropyLoss()
        crit_reg = nn.MSELoss()

        os.makedirs(args.out_dir, exist_ok=True)
        best_score, best_path = -float("inf"), None

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, crit_cls, crit_reg, device)
            metrics = evaluate(model, val_loader, device)
            score = 0.5 * metrics["accuracy"] + 0.5 * metrics["r2"]
            if score > best_score:
                best_score = score
                best_path = os.path.join(args.out_dir, "best_model.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                    "feature_means": train_ds.means,
                    "feature_stds": train_ds.stds
                }, best_path)

            # Print loss and validation accuracy & regression R squared only (no detailed metrics here)
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"- Loss: {loss:.4f} "
                f"- Val Acc: {metrics['accuracy']:.4f} "
                f"- Val R2: {metrics['r2']:.4f}"
            )

        print(f"Training complete. Best model saved to: {best_path}")

    elif args.mode == "predict-interactive":
        def _load_ckpt(path):
            try:
                return torch.load(path, map_location="cpu")
            except Exception as e:
                try:
                    return torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    raise e

        ckpt = _load_ckpt(args.model_path)

        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                model_state = ckpt["model_state"]
            elif "state_dict" in ckpt:
                model_state = ckpt["state_dict"]
            else:
                model_state = ckpt
            metrics = ckpt.get("metrics", {})
            feature_means = ckpt.get("feature_means", {k: 0.0 for k in FEATURE_KEYS})
            feature_stds = ckpt.get("feature_stds", {k: 1.0 for k in FEATURE_KEYS})
        else:
            raise ValueError("Unsupported checkpoint format")

        print("\n=== Loaded Model Performance Metrics ===")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                try:
                    print(f"{k}: {v:.4f}")
                except Exception:
                    print(f"{k}: {v}")
        if "confusion_matrix" in metrics:
            print("Confusion Matrix:")
            print(metrics["confusion_matrix"])

        model = MTTransformer(input_dim=len(FEATURE_KEYS))
        try:
            model.load_state_dict(model_state)
        except Exception:
            model.load_state_dict(model_state, strict=False)
        model.eval()

        row = get_interactive_input()

        x_raw = [ensure_numeric(row.get(k, 0.0)) for k in FEATURE_KEYS]
        x_norm = [
            (x_raw[i] - feature_means[FEATURE_KEYS[i]]) / feature_stds[FEATURE_KEYS[i]]
            for i in range(len(FEATURE_KEYS))
        ]
        x_tensor = torch.tensor([x_norm], dtype=torch.float32)

        with torch.no_grad():
            logits, pr = model(x_tensor)

        probs = torch.softmax(logits, dim=1).numpy()[0]
        cls = int(logits.argmax(dim=1).numpy()[0])
        hbf = float(pr.numpy()[0])

        # If true labels present, compute and print detailed metrics
        if CLASS_TARGET in row and REG_TARGET in row:
            true_cls = [row[CLASS_TARGET]]
            true_reg = [row[REG_TARGET]]
            compute_and_print_metrics(true_cls, logits.cpu(), true_reg, pr.cpu())

        print(format_severity(cls, probs))
        print(f"Predicted HbF Target: {hbf:.1f}")

        result = {
            "severity_probabilities": {
                "Normal": float(probs[0]),
                "Mild": float(probs[1]),
                "Severe": float(probs[2])
            }
        }
        plot_interactive(result)


if __name__ == "__main__":
    main()
