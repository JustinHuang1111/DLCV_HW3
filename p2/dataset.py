import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

## max len of train: 54
## max len of val : 50
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def check_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class InfDataset(Dataset):
    def __init__(self, data_dir, tokenizer, imgsize=224, train=True):
        super(InfDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        # self.transform = T.Compose([
        #                 T.Resize((imgsize, imgsize)),
        #                 T.ToTensor()
        #                 ])
        train_transform = T.transforms.Compose(
            [
                T.Resize((imgsize, imgsize)),
                RandomRotation(),
                T.transforms.Lambda(check_rgb),
                T.transforms.ColorJitter(
                    brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]
                ),
                T.transforms.RandomHorizontalFlip(),
                T.transforms.ToTensor(),
                T.transforms.Normalize(MEAN, STD),
            ]
        )

        val_transform = T.transforms.Compose(
            [
                T.Resize((imgsize, imgsize)),
                T.transforms.Lambda(check_rgb),
                T.transforms.ToTensor(),
                T.transforms.Normalize(MEAN, STD),
            ]
        )

        if train:
            self.transform = train_transform
        else:
            self.transform = val_transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        # print(img.size())

        if img.size(0) != 3:
            img = torch.cat((img, img, img))
        # print(captions.shape)
        # print(captions)

        name = fname.split(".")[0]

        return img, name
