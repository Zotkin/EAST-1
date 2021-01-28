from typing import List
import os

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import csv

from icdar import generate_data


def load_image_from_local(path: str):
    img = cv2.imread(path)
    return img


def load_annotation_from_local(path: str):
    """
    Args
        path:str

    Returns:

    """
    text_polys = []
    text_tags = []

    if not os.path.exists(path):
        return np.array(text_polys, dtype=np.float32)
    with open(path, 'r', encoding='cp437') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


class EastDataset(Dataset):

    def __init__(self, img_paths: List[str], label_paths: List[str], memory_flags: List[bool], size: int):

        self.img_paths = img_paths
        self.label_paths = label_paths
        self.memory_flags = memory_flags
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,  idx):

        img = load_image_from_local(self.img_paths[idx])
        boxes, tag = load_annotation_from_local(self.label_paths[idx])
        img, score_map, geo_map, training_mask = generate_data(img, boxes, tag, self.size)
        memory_flag = self.memory_flags[idx]

        return img, score_map, geo_map, training_mask, memory_flag

    @staticmethod
    def collate_fn(batch):
        images = []
        score_maps = []
        geo_maps = []
        training_masks = []
        memory_flags = []

        for b in batch:
            img, score_map, geo_map, training_mask, memory_flag = b

            img = torch.tensor(np.moveaxis(img[:, :, ::-1], 2, 0).astype(np.float32))
            images.append(img)

            score_map = torch.tensor(score_map[::4, ::4, np.newaxis].astype(np.float32))
            score_maps.append(score_map)

            geo_map = torch.tensor(geo_map[::4, ::4, :].astype(np.float32))
            geo_maps.append(geo_map)

            training_mask = torch.tensor(training_mask[::4, ::4, np.newaxis].astype(np.float32))
            training_masks.append(training_mask)

            memory_flag = torch.tensor(memory_flag)
            memory_flags.append(memory_flag)

        images = torch.stack(images)
        score_maps = torch.stack(score_maps)
        geo_maps = torch.stack(geo_maps)
        training_masks = torch.stack(training_masks)
        memory_flags = torch.stack(memory_flags)

        return images, score_maps, geo_maps, training_masks, memory_flags

