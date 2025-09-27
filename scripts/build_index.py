# scripts/build_index.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_scalar_number(p: Path, as_int=False):
    arr = np.load(str(p), allow_pickle=True)
    # handle np scalar, 0-d array, or string
    if hasattr(arr, "item"):
        arr = arr.item()
    try:
        val = int(float(arr)) if as_int else float(arr)
    except Exception:
        # fallback: cast string like b'3' or ' 3 '
        s = str(arr).strip().replace("\x00", "")
        val = int(float(s)) if as_int else float(s)
    return val

def load_landmarks(p: Path):
    try:
        arr = np.load(str(p), allow_pickle=True)
        arr = np.array(arr, dtype=float)
        if arr.ndim == 1 and arr.size == 136:
            arr = arr.reshape(68, 2)
        return arr.tolist()
    except Exception:
        return None

def resolve_image_for_base(img_dir: Path, base: str):
    # try exact filename with common extensions
    for ext in IMG_EXTS:
        cand = img_dir / f"{base}{ext}"
        if cand.exists():
            return cand.name
    # try any file whose stem matches (case-insensitive)
    for p in img_dir.iterdir():
        if p.is_file() and p.stem.lower() == base.lower() and p.suffix.lower() in IMG_EXTS:
            return p.name
    return None

def build_from_perfile(ann_dir: Path, img_dir: Path, out_dir: Path):
    exp_files = list(ann_dir.glob("*_exp.npy"))
    val_files = list(ann_dir.glob("*_val.npy"))
    aro_files = list(ann_dir.glob("*_aro.npy"))
    lnd_files = list(ann_dir.glob("*_lnd.npy"))  # optional

    if not (exp_files and val_files and aro_files):
        raise RuntimeError("Per-file layout detected but required *_exp.npy, *_val.npy, *_aro.npy not all present.")

    def map_by_base(files, suffix):
        m = {}
        slen = len(suffix)
        for f in files:
            name = f.name
            if name.endswith(suffix):
                base = name[:-slen]
                m[base] = f
        return m

    exp_map = map_by_base(exp_files, "_exp.npy")
    val_map = map_by_base(val_files, "_val.npy")
    aro_map = map_by_base(aro_files, "_aro.npy")
    lnd_map = map_by_base(lnd_files, "_lnd.npy") if lnd_files else {}

    # Prefer iterating over images to ensure they exist
    img_bases = set(p.stem for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not img_bases:
        # fallback: intersect bases present in all required maps
        img_bases = set(exp_map).intersection(val_map).intersection(aro_map)

    rows = []
    missing_img, missing_ann = 0, 0

    for base in sorted(img_bases, key=lambda x: (len(x), x)):
        # resolve image filename
        img_name = resolve_image_for_base(img_dir, base)
        if img_name is None:
            missing_img += 1
            continue

        e = exp_map.get(base); v = val_map.get(base); a = aro_map.get(base)
        if not (e and v and a):
            missing_ann += 1
            continue

        try:
            exp = load_scalar_number(e, as_int=True)
            val = load_scalar_number(v, as_int=False)
            aro = load_scalar_number(a, as_int=False)
        except Exception:
            # skip if any scalar malformed
            continue

        lmk = None
        lf = lnd_map.get(base)
        if lf:
            lmk = load_landmarks(lf)

        rows.append({
            "filename": img_name,
            "expression": int(exp),
            "valence": float(val),
            "arousal": float(aro),
            "landmarks": lmk,
            "box": None
        })

    if not rows:
        raise RuntimeError("No rows built. Check that image names match the *_exp.npy base names.")

    df = pd.DataFrame(rows)

    # Basic sanity: keep only files that actually exist (paranoia)
    df["exists"] = df["filename"].apply(lambda s: (img_dir / s).exists())
    df = df[df["exists"]].drop(columns=["exists"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"index.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    try:
        df.to_parquet(out_dir/"index.parquet", index=False)
    except Exception:
        pass

    print(f"[done] Rows: {len(df)}  | dropped missing_img: {missing_img}, missing_ann: {missing_ann}")
    print("head:\n", df.head())
    print(f"Saved → {out_dir/'index.csv'}", "and index.parquet" if (out_dir/'index.parquet').exists() else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(Path(__file__).resolve().parents[1]/"data"/"annotations"))
    ap.add_argument("--imgs", default=str(Path(__file__).resolve().parents[1]/"data"/"images"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]/"data"))
    args = ap.parse_args()

    ann_dir = Path(args.ann); img_dir = Path(args.imgs); out_dir = Path(args.out)
    print("Annotations:", ann_dir)
    print("Images:     ", img_dir)
    print("Output to:  ", out_dir)

    build_from_perfile(ann_dir, img_dir, out_dir)

if __name__ == "__main__":
    main()
