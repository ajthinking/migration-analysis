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
from BagOfWords import BagOfWords

class MigrationsDataset(data.Dataset):
    def __init__(
            self,
            train=False,
            test=False,
            limit_rows=False,
            transform=None,
            target_transform=None,
            download=False
    ):
        self.path = r"/Users/anders/Code/migration-analysis/data/processed/migrations_metadata.csv"
        data = pd.read_csv(self.path)

        # pre split limit
        if(limit_rows):
            data = data.sample(limit_rows, random_state=1337)

        split_point1 = int(np.floor(len(data)*0.9))
        data_train = data[0:split_point1]
        data_test = data[split_point1:]

        self.bow_column_name = BagOfWords(data_train.column_name)
        self.bow_column_data_type = BagOfWords(data_train.column_data_type)

        if(train):
            self.x = Variable(torch.tensor(self.bow_column_name.tensors, dtype=torch.float))
            self.y = Variable(torch.tensor(self.bow_column_data_type.tensors, dtype=torch.float))
        elif(test):
            self.x = Variable(torch.tensor(self.bow_column_name.tensors_for(data_test.column_name), dtype=torch.float))
            self.y = Variable(torch.tensor(self.bow_column_data_type.tensors_for(data_test.column_data_type), dtype=torch.float))
        else:
            data = []  

    def get_datatypes(self):
        return self.datatypes

    def output_tensor_to_text(self, output_tensor):
        index, value = max(enumerate(output_tensor[0]), key=operator.itemgetter(1))
        return self.datatypes[index]

    def input_tensor_to_text(self, input_tensor):
        index, value = max(enumerate(input_tensor[0]), key=operator.itemgetter(1))
        return self.global_word_bins[index]            

    def text_to_output_tensor(self, word):
        return list(map(lambda item: float(word in [item]), self.global_word_bins))

    def get_local_word_bins(self, migration):
        return migration['column_name'].split()    

    def get_global_word_bins(self, migrations):
        return np.unique(list(
                
                map(
                        lambda migration: migration['column_name'],
                        migrations
                )
        ))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)