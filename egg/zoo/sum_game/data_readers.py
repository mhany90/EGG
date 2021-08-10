# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import Dataset

# These input-data-processing classes take input data from a text file and convert them to the format
# appropriate for the recognition and discrimination games, so that they can be read by
# the standard pytorch DataLoader. The latter requires the data reading classes to support
# a __len__(self) method, returning the size of the dataset, and a __getitem__(self, idx)
# method, returning the idx-th item in the dataset. We also provide a get_n_features(self) method,
# returning the dimensionality of the Sender input vector after it is transformed to one-hot format.

# The AttValSumDataset class is used in the sumation game. It takes an input file with a
# space-delimited vector of integer_1, integer_2, and their sum per line and  creates a data-frame with the two mandatory
# fields expected in EGG games, namely sender_input and labels.
# The two integer fields are represented as one-hot in sender_input, and the sum is in the original integer-based format in
# labels.
class AttValSumDataset(Dataset):
    def __init__(self, path, n_range):
        frame = np.loadtxt(path, dtype="S10")
        self.frame = []
        for row in frame:
            config = list(map(int, row))
            z = torch.zeros((2, n_range))
            for i in range(2):
                z[i, config[i]] = 1
            label = torch.tensor(int(row[-1]))
            self.frame.append((z.view(-1), label))

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


# The AttValRecoDataset class is used in the reconstruction game. It takes an input file with a
# space-delimited attribute-value vector per line and  creates a data-frame with the two mandatory
# fields expected in EGG games, namely sender_input and labels.
# In this case, the two fields contain the same information, namely the input attribute-value vectors,
# represented as one-hot in sender_input, and in the original integer-based format in
# labels.
class AttValRecoDataset(Dataset):
    def __init__(self, path, n_attributes, n_values):
        frame = np.loadtxt(path, dtype="S10")
        self.frame = []
        for row in frame:
            if n_attributes == 1:
                row = row.split()
            config = list(map(int, row))
            z = torch.zeros((n_attributes, n_values))
            for i in range(n_attributes):
                z[i, config[i]] = 1
            label = torch.tensor(list(map(int, row)))
            self.frame.append((z.view(-1), label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]


