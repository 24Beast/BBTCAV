# Importing Libraries
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

# Testing

DATA_PATH = Path("C:/Users/btokas/Projects/NetDissect/dataset/broden1_227")
CONCEPT_TYPES = ["color", "object", "part", "material", "texture", "scene"]

index_file_path = DATA_PATH / "index.csv"

index_data = pd.read_csv(index_file_path)

seg_concepts = CONCEPT_TYPES[:-2]
num_concepts = CONCEPT_TYPES[-2:]

data_vals = []

num_imgs = len(index_data)
for num in range(num_imgs):
    print(f"\rWorking on item num:{num+1}/{num_imgs}", end="")
    item = index_data.iloc[num]
    data_val = {}
    data_val["img_name"] = item["image"]
    for concept in seg_concepts:
        if (type(item[concept]) == float) and (np.isnan(item[concept])):
            data_val[concept] = []
        else:
            data_val[concept] = []
            file_paths = item[concept].split(";")
            for file_path in file_paths:
                img_path = DATA_PATH / "images" / file_path
                img = Image.open(img_path)
                img = np.array(img)
                if concept == "color":
                    vals, counts = np.unique(img, return_counts=True)
                    vals = vals[counts >= 0.05 * counts.sum()]
                else:
                    vals = np.unique(img)
                vals = vals[vals != 0]
                data_val[concept].extend(vals)
    for concept in num_concepts:
        if (type(item[concept]) == float) and (np.isnan(item[concept])):
            data_val[concept] = []
        else:
            if type(item[concept]) == str:
                data_val[concept] = item[concept].split(";")
            else:
                data_val[concept] = [item[concept]]
    data_vals.append(data_val)

data = pd.DataFrame(data_vals)
data.to_csv(DATA_PATH / "map.csv")
