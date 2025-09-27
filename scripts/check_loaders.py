# --- ensure project root is on sys.path ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
# ------------------------------------------

import pandas as pd
from src.data import make_loaders
from src.aug import train_tfms, val_tfms
from src.config import BATCH_SIZE

def main():
    DATA = Path("data")
    train_df = pd.read_csv(DATA / "train.csv")
    val_df   = pd.read_csv(DATA / "val.csv")

    # On Windows, avoid multiprocessing in quick sanity checks
    train_loader, val_loader = make_loaders(
        train_df, val_df,
        batch_size=BATCH_SIZE,
        num_workers=0,                 # <-- key change
        transforms_train=train_tfms(),
        transforms_val=val_tfms()
    )

    batch = next(iter(train_loader))
    print("image:", batch["image"].shape)
    print("y_cls:", batch["y_cls"].shape)
    print("y_va :", batch["y_va"].shape)
    print("va_mask:", batch["va_mask"].shape)
    print("example filenames:", batch["fname"][:5])

if __name__ == "__main__":          # <-- key change
    main()
