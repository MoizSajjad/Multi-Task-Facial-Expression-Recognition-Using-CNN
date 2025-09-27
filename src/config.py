from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMG_DIR  = DATA_DIR / "images"
ANN_DIR  = DATA_DIR / "annotations"

SEED = 42
NUM_CLASSES = 8
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4

IDX2EMO = {0:"Neutral",1:"Happy",2:"Sad",3:"Surprise",4:"Fear",5:"Disgust",6:"Anger",7:"Contempt"}
EMO2IDX = {v:k for k,v in IDX2EMO.items()}
