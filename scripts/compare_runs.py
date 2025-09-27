# --- ensure project root is on sys.path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# ------------------------------------------

import json
from pathlib import Path

METRICS = [
    # classification
    "cls_acc", "cls_f1_macro", "cls_kappa", "cls_kripp_alpha",
    "cls_roc_auc_macro", "cls_pr_auc_macro",
    # VA
    "val_rmse", "aro_rmse", "val_ccc", "aro_ccc",
    "val_pearson", "aro_pearson", "val_sagr", "aro_sagr",
]

def load_summary(backbone: str):
    p = Path("reports/figs") / backbone / "summary.json"
    if not p.exists():
        print(f"[skip] summary not found: {p}")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] failed to read {p}: {e}")
        return None

def main():
    names = ["resnet18", "efficientnet_b0"]
    sums = {bb: load_summary(bb) for bb in names}
    if not all(sums.values()):
        print("[hint] Run eval first, e.g.:")
        print("  python scripts\\eval_report.py --backbone resnet18 --ckpt outputs\\checkpoints\\resnet18_best.pt")
        print("  python scripts\\eval_report.py --backbone efficientnet_b0 --ckpt outputs\\checkpoints\\efficientnet_b0_best.pt")
        return

    # Print table
    print("Metric".ljust(24), names[0].rjust(14), names[1].rjust(14))
    rows = []
    for m in METRICS:
        v0 = sums[names[0]].get(m, float("nan"))
        v1 = sums[names[1]].get(m, float("nan"))
        try:
            s0 = f"{v0:.4f}"
            s1 = f"{v1:.4f}"
        except Exception:
            s0 = str(v0); s1 = str(v1)
        print(m.ljust(24), s0.rjust(14), s1.rjust(14))
        rows.append((m, s0, s1))

    # Save CSV + Markdown for your report
    out_dir = Path("reports/figs"); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"compare.csv").write_text(
        "metric,resnet18,efficientnet_b0\n" + "\n".join([",".join(r) for r in rows]),
        encoding="utf-8"
    )
    md_lines = ["| Metric | ResNet-18 | EfficientNet-B0 |", "|---|---:|---:|"] + \
               [f"| {m} | {a} | {b} |" for m,a,b in rows]
    (out_dir/"compare.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ok] wrote {out_dir/'compare.csv'} and {out_dir/'compare.md'}")

if __name__ == "__main__":
    main()
