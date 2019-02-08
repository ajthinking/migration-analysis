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

class BaseDataset(data.Dataset):
    def __init__(
            self,
            train=False,
            test=False,
            limit_rows=False,
            transform=None,
            target_transform=None,
            download=False
    ):
        data = pd.read_csv(path)

        # pre split limit
        if(limit_rows):
            data = data.sample(limit_rows, random_state=1337)

        split_point1 = int(np.floor(len(data)*0.9))
        data_train = data[0:split_point1]
        data_test = data[split_point1:]

        print(len(data_train))
        print(len(data_test))

        print(data_test)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)