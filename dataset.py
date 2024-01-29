import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import os
import random

from utils import prepare_image_with_bbox


class ClipDataset(Dataset):
    def __init__(self, dataframe, preprocess, split_train=False):
        super(ClipDataset, self).__init__()
        self.dataframe = dataframe if not split_train else dataframe[dataframe['label'] == 'good']
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
    

class DisturbClipDataset(ClipDataset):
    def __init__(self, dataframe, preprocess):
        super(DisturbClipDataset, self).__init__(dataframe, preprocess, split_train=True)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image_path = os.path.join('/home/zhaoyue/groundingdata/allsuccess_ori', os.path.basename(image_path))
        bboxes_json_path = image_path.replace('.png', '.json').replace('allsuccess_ori', 'allsuccess_gt')
        disturb = random.choice([True, False])
        image = Image.open(image_path).convert("RGB")
        with open(bboxes_json_path, 'r') as f:
            page_structure = json.load(f)
        image, actual_disturb = prepare_image_with_bbox(image, page_structure, disturb, width=3)
        if actual_disturb:
            ret_label = torch.tensor([0, 1])
        else:
            ret_label = torch.tensor([1, 0])
        return self.preprocess(image), ret_label
