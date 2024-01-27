import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import os
import random

from utils import prepare_image_with_bbox, get_available_gpu


os.environ['CUDA_VISIBLE_DEVICES'] = str(get_available_gpu())


class ClipDataset(Dataset):
    def __init__(self, dataframe, preprocess):
        super(ClipDataset, self).__init__()
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
    

class AugClipDataset(ClipDataset):
    def __init__(self, dataframe, preprocess):
        super(AugClipDataset, self).__init__(dataframe, preprocess)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        image_path = os.path.join('~/groundingdata/allsuccuess_ori', os.path.basename(image_path))
        bboxes_json_path = image_path.replace('.jpg', '.json').replace('allsuccuess_ori', 'allsuccuess_gt')
        disturb = random.choice([True, False])
        if disturb:
            ret_label = torch.tensor([0, 1])
        else:
            ret_label = torch.tensor([1, 0])
        image = Image.open(image_path).convert("RGB")
        with open(bboxes_json_path, 'r') as f:
            page_structure = json.load(f)
        image = prepare_image_with_bbox(image, page_structure, disturb, width=3)
        return self.preprocess(image), ret_label
