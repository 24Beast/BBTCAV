# Imprting Libraries
import yaml
import argparse
from pathlib import Path
from torchvision import transforms
from utils.data import CelebAJointConcept


# Parser for args parse
parser = argparse.ArgumentParser(description="AE-TCAV Runner with YAML Config")
parser.add_argument(
    "--config", type=str, required=True, help="Path to YAML configuration file"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(f"{config=}")

# LOADING CONSTANTS
MODEL_TYPE = config["MODEL_TYPE"]
MODEL_PATH = Path(config["MODEL_PATH"])
TRAIN_MODEL = 0 if MODEL_PATH.exists() else 1
DATA_DIR = Path(config["DATA_DIR"])
CONCEPTS = config["CONCEPTS"]
B_SIZE = config["BATCH_SIZE"]
target = config["TARGET"]
H, W = config["IMG_SIZE"]

transform = transforms.Compose(
    [
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)
