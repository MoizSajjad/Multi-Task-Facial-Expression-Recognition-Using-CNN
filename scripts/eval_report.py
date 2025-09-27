# Generate metrics + figures for the report from a best checkpoint
# --- ensure project root is on sys.path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# ------------------------------------------


import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2, matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

from src.models import MultiTaskNet
from src.data import make_loaders, AffectDataset
from src.aug import val_tfms
from src.config import NUM_CLASSES, IMG_DIR

def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(6,5))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_roc_pr(y_true, y_proba, save_dir):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    save_dir.mkdir(parents=True, exist_ok=True)
    # One-vs-rest
    for c in range(y_proba.shape[1]):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, c])
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, c])
        ap = average_precision_score(y_bin, y_proba[:, c])

        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1], [0,1], lw=1, linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC – Class {c}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_dir / f"roc_class_{c}.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR – Class {c}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(save_dir / f"pr_class_{c}.png", dpi=200)
        plt.close()

def make_examples(df, y_true, y_pred, out_dir, n_each=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    correct_idx = np.where(y_true == y_pred)[0]
    wrong_idx   = np.where(y_true != y_pred)[0]
    pick_c = correct_idx[:n_each]
    pick_w = wrong_idx[:n_each]

    def grid(indices, title, out_path):
        cols = 3
        rows = max(1, int(np.ceil(len(indices)/cols)))
        plt.figure(figsize=(cols*3, rows*3))
        for i, idx in enumerate(indices, 1):
            p = IMG_DIR / df.iloc[idx]["filename"]
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            plt.subplot(rows, cols, i); plt.imshow(img); plt.axis("off")
            gt = int(y_true[idx]); pr = int(y_pred[idx])
            plt.title(f"GT:{gt}  PR:{pr}", fontsize=9)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    if len(pick_c): grid(pick_c, "Correct examples", out_dir / "examples_correct.png")
    if len(pick_w): grid(pick_w, "Incorrect examples", out_dir / "examples_incorrect.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, required=True, choices=["resnet18","efficientnet_b0"])
    ap.add_argument("--ckpt", type=str, required=True, help="path to best checkpoint .pt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = MultiTaskNet(args.backbone, pretrained=False).eval()
    model.load_state_dict(ckpt["model"])

    # Build a val loader (no shuffle)
    data_dir = Path("data")
    val_df = pd.read_csv(data_dir/"val.csv")
    from torch.utils.data import DataLoader
    ds = AffectDataset(val_df, transforms=val_tfms())
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred, y_proba = [], [], []
    va_true, va_pred = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            logits, va = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(1)

            y_true.append(batch["y_cls"].numpy())
            y_pred.append(pred)
            y_proba.append(probs)
            va_true.append(batch["y_va"].numpy())
            va_pred.append(va.cpu().numpy())

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); y_proba = np.concatenate(y_proba)
    va_true = np.concatenate(va_true); va_pred = np.concatenate(va_pred)

    # Metrics and plots
    cm = confusion_matrix(y_true, y_pred)
    out_dir = Path("reports/figs") / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, [str(i) for i in range(NUM_CLASSES)], out_dir / "confusion_matrix.png")
    plot_roc_pr(y_true, y_proba, out_dir)

    # Save some examples
    make_examples(val_df, y_true, y_pred, out_dir)

    # Save raw arrays for your report if needed
    np.save(out_dir/"y_true.npy", y_true)
    np.save(out_dir/"y_pred.npy", y_pred)
    np.save(out_dir/"y_proba.npy", y_proba)
    np.save(out_dir/"va_true.npy", va_true)
    np.save(out_dir/"va_pred.npy", va_pred)

    # Summary JSON
    from src.metrics_cls import multiclass_metrics
    from src.metrics_reg import va_metrics
    cls = multiclass_metrics(y_true, y_pred, y_proba, NUM_CLASSES)
    reg = va_metrics(va_true, va_pred)
    summary = {"backbone": args.backbone, **{f"cls_{k}": v for k,v in cls.items()}, **reg}
    with open(out_dir/"summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved figs + summary to:", out_dir)

if __name__ == "__main__":
    main()
