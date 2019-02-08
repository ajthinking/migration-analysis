from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import sys
from torch.autograd import Variable
import json
import csv
import numpy as np
import pandas as pd
import torch.utils.data as data
import time
import operator

from BaseDataset import BaseDataset

class BagOfWords(object):
    def __init__(
            self,
            words
    ):
        self.words = words
        self.bags = self.words.unique()
        self.tensors = self.encode()

    def encode(self):
        return list(map(
            lambda word: list(map(lambda bag_word: float(word == bag_word), self.bags)), self.words.values
        ))

    def text_to_tensor(self, text):
        return list(map(lambda bag_word: float(text == bag_word), self.bags))

    def tensor_to_text(self, tensor):
        values, indices = tensor.max(0)
        print(indices)
        print(values)
        return self.words.values[indices.item()]                    