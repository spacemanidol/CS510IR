import numpy as np
from torch.utils.data import Dataset
import random
import os

class MSMARCO_PassageDataset(Dataset):
    def __init__(self, path, size=None):
        super().__init__()
        self.path = path
        self.size = size
        self.data = []
        self.negative_pair = []
        with open(self.path,'r') as f:
            for l in f:
                l = l.strip().split('\t')
                if len(l) == 3:
                    self.data.append(("{}[SEP]{}".format(l[0],l[1]), 1))
                    self.data.append(("{}[SEP]{}".format(l[0],l[2]), 0))
        if self.size != None:
            self.data = self.data[:self.size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item[0], item[1]
