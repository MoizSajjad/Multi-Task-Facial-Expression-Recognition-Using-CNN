# --- ensure project root is on sys.path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# ------------------------------------------

import json
import matplotlib.pyplot as plt

def _load_hist(path: Path):
    if not path.exists():
        print(f"[skip] history not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None

def _plot_series(xs, series_map, title, save_path):
    plt.figure()
    for label, ys in series_map.items():
        # guard against missing keys in earlier epochs
        ys = [y if isinstance(y, (int, float)) else None for y in ys]
        plt.plot(xs, ys, label=label)
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[ok] saved {save_path}")

def plot_backbone(backbone: str):
    hist_path = Path("outputs/logs") / f"{backbone}_history.json"
    out_dir   = Path("reports/figs") / backbone
    hist = _load_hist(hist_path)
    if not hist:
        return

    xs = [h.get("epoch", i+1) for i, h in enumerate(hist)]
    def col(k): return [h.get(k, None) for h in hist]

    # Losses
    _plot_series(xs, {
        "loss_cls": col("loss_cls"),
        "loss_reg": col("loss_reg"),
    }, f"{backbone}: Losses", out_dir / "losses.png")

    # Classification metrics
    _plot_series(xs, {
        "acc":        col("cls_acc"),
        "f1_macro":   col("cls_f1_macro"),
        "kappa":      col("cls_kappa"),
    }, f"{backbone}: Classification", out_dir / "cls_metrics.png")

    # Valence/Arousal RMSE
    _plot_series(xs, {
        "val_rmse": col("val_rmse"),
        "aro_rmse": col("aro_rmse"),
    }, f"{backbone}: VA RMSE", out_dir / "va_rmse.png")

    # Valence/Arousal CCC
    _plot_series(xs, {
        "val_ccc": col("val_ccc"),
        "aro_ccc": col("aro_ccc"),
    }, f"{backbone}: VA CCC", out_dir / "va_ccc.png")

def main():
    # Plot for both common baselines if present
    for bb in ["resnet18", "efficientnet_b0"]:
        plot_backbone(bb)

if __name__ == "__main__":
    main()
