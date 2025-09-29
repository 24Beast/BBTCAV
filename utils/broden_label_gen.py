# Importing Libraries
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


# Helper Function
def getNumDict(DATA_PATH, concept_name):
    file_path = DATA_PATH / f"c_{concept_name}.csv"
    data = pd.read_csv(file_path)
    num_dict = {item.number: item.code for num, item in data.iterrows()}
    return num_dict


# Testing

DATA_PATH = Path("C:/Users/btokas/Projects/NetDissect/dataset/broden1_227")
CONCEPT_TYPES = ["color", "object", "part", "material", "texture", "scene"]
CONCEPT_DICTS = {concept: getNumDict(DATA_PATH, concept) for concept in CONCEPT_TYPES}

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
        curr_concept_dict = CONCEPT_DICTS[concept]
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
                vals = [
                    curr_concept_dict[item]
                    for item in vals
                    if int(item) in curr_concept_dict.keys()
                ]
                data_val[concept].extend(vals)
    for concept in num_concepts:
        curr_concept_dict = CONCEPT_DICTS[concept]
        if pd.isna(item[concept]):
            data_val[concept] = []
        else:
            if type(item[concept]) == str:
                vals = item[concept].split(";")
                vals = [
                    curr_concept_dict[int(item)]
                    for item in vals
                    if int(item) in curr_concept_dict.keys()
                ]
                data_val[concept] = vals
            else:
                data_val[concept] = (
                    [curr_concept_dict[int(item[concept])]]
                    if int(item[concept]) in curr_concept_dict.keys()
                    else []
                )
    data_vals.append(data_val)

data = pd.DataFrame(data_vals)
data.to_csv(DATA_PATH / "map.csv")
