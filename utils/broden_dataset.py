# Importing Libraries
import ast
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Union

# CONSTANTS
VALID_CONCEPT_TYPES = ["color", "object", "part", "material", "texture", "scene"]
CONCEPT_COUNTS = {
    "color": 11,
    "object": 584,
    "part": 234,
    "material": 32,
    "texture": 47,
    "scene": 468,
}


# Helper Class
class BroDenDataset(torch.utils.data.Dataset):

    def __init__(
        self, data_path: Union[str, Path], concepts: list[str], transform=None
    ):
        self.data_path = Path(data_path)
        self.concepts = concepts
        for concept in concepts:
            if not (concept in VALID_CONCEPT_TYPES):
                raise ValueError(
                    f"Unexpected Value: '{concept}' encountered in concepts.\n Valid concepts: {VALID_CONCEPT_TYPES}"
                )
        self.file_path = data_path / "map.csv"
        if not (self.file_path.exists()):
            raise FileNotFoundError(
                f"Could not find file {self.file_path}. If map.csv does not exist in {self.data_path} directory, use broden_label_gen.py"
            )
        self.transform = transform
        print("Loading Data")
        self.loadData()
        print("Data Ready!")

    def loadData(self):
        self.data = pd.read_csv(self.file_path)
        # Filter concepts
        self.offsets = [0]
        for concept in self.concepts:
            curr_size = CONCEPT_COUNTS[concept]
            self.offsets.append(self.offsets[-1] + curr_size)
        total_size = self.offsets[-1]
        self.concept_arr = torch.zeros((len(self.data), total_size))
        for i, concept in enumerate(self.concepts):
            start_pos = self.offsets[i]
            self.data[concept] = self.data[concept].replace("[nan]", "[]")
            pos = self.data[concept].map(
                lambda x: (
                    np.array(ast.literal_eval(x)).astype(np.int32) + start_pos - 1
                    if (len(ast.literal_eval(x)) > 0)
                    else []
                )
            )
            for num, curr_pos in enumerate(pos):
                if len(curr_pos) > 0:
                    self.concept_arr[num, curr_pos] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        file_path = self.data.iloc[idx].img_name
        img_path = self.data_path / "images" / file_path
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self.concept_arr[idx]


if __name__ == "__main__":
    DATA_PATH = Path("C:/Users/btokas/Projects/NetDissect/dataset/broden1_227")
    dataset = BroDenDataset(DATA_PATH, VALID_CONCEPT_TYPES)
