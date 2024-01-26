import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class ClipDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image_path = os.path.join('./data/all_img', os.path.basename(image_path))
        label = self.dataframe.iloc[idx, 1]
        image = Image.open(image_path).convert("RGB")
        if label == 'good':
            ret_label = torch.tensor([1, 0])
        else:
            ret_label = torch.tensor([0, 1])
        return self.preprocess(image), ret_label
