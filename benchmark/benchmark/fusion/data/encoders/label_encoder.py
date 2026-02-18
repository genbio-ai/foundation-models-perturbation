from sklearn.preprocessing import LabelEncoder
from typing import List
import torch


class TensorLabelEncoder:
    def __init__(self, key_added: str, required_key: str):
        self.key_added = key_added
        self.required_key = required_key
        self.le = LabelEncoder()

    def set_universe(self, universe: List):
        self.le.fit(universe)

    def encode(self, x):
        return {
            self.key_added: torch.tensor(
                self.le.transform([x]),
                dtype=torch.long,
            )
        }
