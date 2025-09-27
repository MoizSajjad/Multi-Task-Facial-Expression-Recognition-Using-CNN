from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def make_splits(val_size=0.2, seed=42, data_dir="data"):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "index.csv")

    # stratify by expression (8 classes)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    y = df["expression"].values
    tr_idx, va_idx = next(sss.split(df, y))

    df.iloc[tr_idx].to_csv(data_dir / "train.csv", index=False)
    df.iloc[va_idx].to_csv(data_dir / "val.csv", index=False)

    print(f"Saved {data_dir/'train.csv'} ({len(tr_idx)} rows)")
    print(f"Saved {data_dir/'val.csv'}   ({len(va_idx)} rows)")

if __name__ == "__main__":
    make_splits()
