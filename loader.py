from __future__ import print_function, division
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
import os

class SawyerSimDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):

        self.labels = pd.read_csv(csv_file)
        print(csv_file)
        print(self.labels)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1:].as_matrix()
        label = label.astype('float')
        sample = {'image': image, 'landmarks': label}

        if self.transform:
            sample = self.transform(sample)

        return sample