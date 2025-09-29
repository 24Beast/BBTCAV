# Importing Libraries
import torch
import itertools
import torchvision
from pathlib import Path
from typing import Literal, Union
from broden_dataset import BroDenDataset

# CONSTANTS
FACIAL_HAIR_CLS = ["5_o_Clock_Shadow", "Goatee", "Mustache", "No_Beard", "Sideburns"]
AGE_CLS = "Young"
GENDER_CLS = "Male"
HAIR_CLS = ["Bald", "Bangs", "Wavy_Hair", "Straight_Hair"]
NAME_KEYS = {
    "Age": "Young",
    "Gender": "Male",
    "Bald": "Bald",
    "Skin": "Pale_Skin",
    "Attractive": "Attractive",
    "Fat": "Chubby",
    "Smiling": "Smiling",
}

# Type hints
CONCEPT_TYPE = Literal["Age", "Gender", "Skin", "Bald"]


# Helper Class
class CelebAConcept(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: Literal["train", "test", "valid", "all"],
        target: Literal["Attractive", "Age", "Gender"] = "Attractive",
        concept: CONCEPT_TYPE = "Gender",
        transform=None,
        download: bool = True,
        mode: Literal["concept", "target"] = "target",
        concept_num: int = 1000,  # Must be even!
    ) -> None:
        self.orig_data = torchvision.datasets.CelebA(
            data_dir, split, target_type="attr", download=download, transform=transform
        )
        self.setIndexes(target, concept)
        self.c_num = concept_num
        self.mode = mode
        if self.mode == "concept":
            self.filterConcept()

    def __len__(self):
        return len(self.orig_data)

    def setIndexes(self, target: str, concept: str) -> None:
        attr_list = self.orig_data.attr_names
        if concept in NAME_KEYS.keys():
            concept = NAME_KEYS[concept]
        else:
            raise ValueError(
                f"""Invalid value '{concept}' given for concept.
                Valid values are ['Age', 'Gender', 'Skin', 'Bald']"""
            )
        if target in NAME_KEYS.keys():
            target = NAME_KEYS[target]
        else:
            raise ValueError(
                f"""Invalid value '{target}' given for target.
                Valid values are ['Attractive','Age', 'Gender']"""
            )
        self.c_idx = attr_list.index(concept)
        self.t_idx = attr_list.index(target)

    def __getitem__(self, idx: int):
        img, attrs = self.orig_data[idx]
        concept = attrs[self.c_idx].item()
        target = attrs[self.t_idx].item()
        if self.mode == "concept":
            return img, concept
        return img, concept, target

    def filterConcept(self):
        attrs = self.orig_data.attr
        c0_t0_idx = torch.where(
            (1 - attrs[:, self.c_idx]) * (1 - attrs[:, self.t_idx]) == 1
        )[0]
        c0_t1_idx = torch.where((1 - attrs[:, self.c_idx]) * attrs[:, self.t_idx] == 1)[
            0
        ]
        c1_t0_idx = torch.where(attrs[:, self.c_idx] * (1 - attrs[:, self.t_idx]) == 1)[
            0
        ]
        c1_t1_idx = torch.where(attrs[:, self.c_idx] * attrs[:, self.t_idx] == 1)[0]
        min_counts = len(
            min([c0_t0_idx, c0_t1_idx, c1_t0_idx, c1_t1_idx], key=lambda x: len(x))
        )
        if 2 * min_counts < self.c_num:
            print(
                f"Balanced counts less than concept num :{self.c_num}, using {2*min_counts} instances for concepts."
            )
        num_counts = min(min_counts, self.c_num // 2)
        c0_t0_idx = c0_t0_idx[:num_counts]
        c0_t1_idx = c0_t1_idx[:num_counts]
        c1_t0_idx = c1_t0_idx[:num_counts]
        c1_t1_idx = c1_t1_idx[:num_counts]
        final_idx = torch.cat([c0_t0_idx, c0_t1_idx, c1_t0_idx, c1_t1_idx])
        perm = torch.randperm(len(final_idx))
        final_idx = final_idx[perm]
        self.orig_data = torch.utils.data.Subset(self.orig_data, final_idx)


class CelebAJointConcept(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: Literal["train", "test", "valid", "all"],
        target: Literal["Attractive", "Age", "Gender"] = "Attractive",
        concept: CONCEPT_TYPE | list[CONCEPT_TYPE] = ["Age", "Gender", "Skin", "Bald"],
        transform=None,
        download: bool = True,
        mode: Literal["concept", "target"] = "target",
        concept_num: int = 1000,  # Must be even!
    ) -> None:
        self.orig_data = torchvision.datasets.CelebA(
            data_dir, split, target_type="attr", download=download, transform=transform
        )
        self.setIndexes(target, concept)
        self.c_num = concept_num
        self.mode = mode
        if self.mode == "concept":
            self.filterConcept()

    def __len__(self):
        return len(self.orig_data)

    def setIndexes(self, target: str, concepts: list[str]) -> None:
        attr_list = self.orig_data.attr_names
        if type(concepts) == str:
            concepts = [concepts]
        concept_names = []
        for concept in concepts:
            if concept in NAME_KEYS.keys():
                concept_names.append(NAME_KEYS[concept])
            else:
                raise ValueError(
                    f"""Invalid value '{concept}' given for concept.
                    Valid values are ['Age', 'Gender', 'Skin', 'Bald']"""
                )
        if target in NAME_KEYS.keys():
            target = NAME_KEYS[target]
        else:
            raise ValueError(
                f"""Invalid value '{target}' given for target.
                Valid values are ['Attractive','Age', 'Gender']"""
            )
        self.c_idx = list(map(attr_list.index, concept_names))
        self.t_idx = attr_list.index(target)

    def __getitem__(self, idx: int):
        img, attrs = self.orig_data[idx]
        concept = attrs[self.c_idx]
        target = attrs[self.t_idx].item()
        if self.mode == "concept":
            return img, concept
        return img, concept, target

    def filterConcept(self):
        """Filter to balanced subset based on multiple concepts and target."""
        attrs = self.orig_data.attr
        N = len(self.c_idx)
        per_bucket = self.c_num // (2**N)  # per (concept_combo, target_value)

        all_idxs = []
        concept_combos = list(itertools.product([0, 1], repeat=N))

        for combo in concept_combos:
            for t_val in [0, 1]:
                mask = torch.ones(len(attrs), dtype=torch.bool)
                for ci, val in zip(self.c_idx, combo):
                    mask &= attrs[:, ci] == val
                mask &= attrs[:, self.t_idx] == t_val
                idx = torch.where(mask)[0]
                count = min(len(idx), per_bucket)
                if count > 0:
                    chosen = idx[torch.randperm(len(idx))[:count]]
                    all_idxs.append(chosen)

        if not all_idxs:
            raise ValueError("No samples match the filtering criteria.")

        final_idx = torch.cat(all_idxs)
        final_idx = final_idx[torch.randperm(len(final_idx))]
        self.orig_data = torch.utils.data.Subset(self.orig_data, final_idx)


# Testing
if __name__ == "__main__":
    DATA_DIR = "../Datasets/CelebA/"
    train_set = CelebAConcept(DATA_DIR, split="train")
    train_concept_set = CelebAConcept(data_dir=DATA_DIR, split="train", mode="concept")
    train_concept_joint_set = CelebAJointConcept(
        DATA_DIR, split="train", mode="concept", concept_num=2000
    )

    # Write tests!
