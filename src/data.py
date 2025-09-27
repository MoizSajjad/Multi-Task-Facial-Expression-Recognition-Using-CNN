import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from .config import IMG_DIR, IMG_SIZE

class AffectDataset(Dataset):
    def __init__(self, index_df, transforms=None):
        self.df = index_df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = IMG_DIR / row["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            out = self.transforms(image=img)
            img = out["image"]

        # HWC->CHW and scale to [0,1]
        img = img.transpose(2, 0, 1).astype("float32") / 255.0

        y_cls = int(row["expression"])
        y_va  = np.array([row["valence"], row["arousal"]], dtype="float32")

        # mask invalid VA values (-2 => ignore)
        va_mask = (y_va != -2).astype("float32")

        return {
            "image": torch.from_numpy(img),
            "y_cls": torch.tensor(y_cls, dtype=torch.long),
            "y_va":  torch.from_numpy(y_va),
            "va_mask": torch.from_numpy(va_mask),
            "fname": row["filename"],
        }

def make_loaders(train_df, val_df, batch_size=32, num_workers=4, transforms_train=None, transforms_val=None):
    tr = AffectDataset(train_df, transforms_train)
    va = AffectDataset(val_df, transforms_val)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True),
        DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )
