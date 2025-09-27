# Train Baseline-1: ResNet-18 multi-task (classification + VA regression)
import argparse, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd

from src.models import ResNet18Multi
from src.data import AffectDataset, make_loaders
from src.aug import train_tfms, val_tfms
from src.metrics_cls import multiclass_metrics, krippendorff_alpha_nominal
from src.metrics_reg import va_metrics
from src.config import BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, NUM_CLASSES

def set_seed(seed=42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def masked_mse(pred, target):
    # mask invalid target values (-2 per assignment)
    mask = (target != -2).float()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    diff = (pred - target) * mask
    return (diff.pow(2).sum() / mask.sum())

def evaluate(model, loader, device, ce, lam):
    model.eval()
    cls_losses, reg_losses = [], []
    y_true, y_pred, y_proba = [], [], []
    va_true, va_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y_cls = batch["y_cls"].to(device)
            y_va  = batch["y_va"].to(device)

            logits, va = model(x)
            loss_cls = ce(logits, y_cls)
            loss_reg = masked_mse(va, y_va)
            loss = loss_cls + lam * loss_reg

            cls_losses.append(loss_cls.item())
            reg_losses.append(loss_reg.item())

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(1)
            y_true.append(batch["y_cls"].cpu().numpy())
            y_pred.append(pred)
            y_proba.append(probs)
            va_true.append(batch["y_va"].cpu().numpy())
            va_pred.append(va.cpu().numpy())

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred); y_proba = np.concatenate(y_proba)
    va_true = np.concatenate(va_true); va_pred = np.concatenate(va_pred)

    cls = multiclass_metrics(y_true, y_pred, y_proba, NUM_CLASSES)
    cls["kripp_alpha"] = float(krippendorff_alpha_nominal(y_true, y_pred, NUM_CLASSES))
    reg = va_metrics(va_true, va_pred)

    return {
        "loss_cls": float(np.mean(cls_losses)),
        "loss_reg": float(np.mean(reg_losses)),
        **{f"cls_{k}": v for k,v in cls.items()},
        **reg
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--lambda-va", type=float, default=0.5, help="weight for VA regression loss")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data")
    train_df = pd.read_csv(data_dir/"train.csv")
    val_df   = pd.read_csv(data_dir/"val.csv")

    # loaders
    from src.data import make_loaders
    train_loader, val_loader = make_loaders(
        train_df, val_df,
        batch_size=args.batch_size,
        transforms_train=train_tfms(),
        transforms_val=val_tfms()
    )

    model = ResNet18Multi(num_classes=NUM_CLASSES, pretrained=args.pretrained).to(device)
    ce = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best_acc, best_path = -1.0, None

    out_ckpt = Path("outputs/checkpoints"); out_ckpt.mkdir(parents=True, exist_ok=True)
    out_logs = Path("outputs/logs"); out_logs.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y_cls = batch["y_cls"].to(device)
            y_va  = batch["y_va"].to(device)

            logits, va = model(x)
            loss = ce(logits, y_cls) + args.lambda_va * masked_mse(va, y_va)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        sched.step()
        metrics = evaluate(model, val_loader, device, ce, args.lambda_va)
        history.append({"epoch": epoch, **metrics})
        print({k: round(v,4) if isinstance(v, (int,float)) else v for k,v in metrics.items() if "conf_mat" not in k})

        # save best by val acc
        if metrics["cls_acc"] > best_acc:
            best_acc = metrics["cls_acc"]
            best_path = out_ckpt / "resnet18_best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, best_path)

        # persist logs each epoch
        with open(out_logs/"resnet18_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best acc: {best_acc:.4f}. Checkpoint: {best_path}")

if __name__ == "__main__":
    main()
