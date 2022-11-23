import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from PIL import Image
import json
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tokenizers import Tokenizer
import random
import torchvision as tv

from utils import nested_tensor_from_tensor_list, read_json


## max len of train: 54
## max len of val : 50
MAX_DIM = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image

def check_rgb(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    return image

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = T.transforms.Compose([
                            T.Resize((224, 224)),
                            RandomRotation(),
                            T.transforms.Lambda(check_rgb),
                            T.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
                            T.transforms.RandomHorizontalFlip(),
                            T.transforms.ToTensor(),
                            T.transforms.Normalize(MEAN, STD),
                            ])

val_transform = T.transforms.Compose([
                        T.Resize((224, 224)),
                        T.transforms.Lambda(check_rgb),
                        T.transforms.ToTensor(),
                        T.transforms.Normalize(MEAN, STD)
                        ])

def read_json(file_name):
    with open(file_name) as handle:
        out = json.load(handle)
    return out


class CocoCaption(Dataset):
    def __init__(
        self, root, ann, max_length, limit, transform=train_transform,  mode="training"
    ):
        super().__init__()
        self.ann = read_json(ann)
        self.root = root
        self.transform = transform
        if mode == "validation":
            self.transform = val_transform
        self.annot = [
            (self._process(val["image_id"]), val["caption"])
            for val in self.ann["annotations"]
        ]
        print("dataset lenth: " , len(self.annot))
        if mode == "validation":
            self.annot = [
            (self._process(val["image_id"]), val["caption"])
            for val in self.ann["annotations"]
        ]
        if mode == "training":
            self.annot = self.annot[:limit]

        self.tokenizer = Tokenizer.from_file(os.path.join("../hw3_data/caption_tokenizer.json"))
        self.max_length = max_length + 1
        self.pad = 0
    def _process(self, image_id):
        for item in self.ann["images"]:
            if item["id"] == image_id:
                return item["file_name"]
        # val = str(image_id).zfill(12)

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        # image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        
        caption_encoded = self.tokenizer.encode(
            caption,
        )

        # print(caption_encoded.tokens)
        caption = torch.as_tensor(caption_encoded.ids, dtype=torch.long)
        pad_right = self.max_length - caption.size(0)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            caption = nn.ConstantPad1d((0, pad_right), value=self.pad)(caption)
        # caption = pad_sequence(caption, batch_first=True, padding_value=0)
        # cap_mask = (1 - np.array(caption_mask)).astype(bool)
        # print(caption)
        # print(cap_mask)

        return image, image_id.split('.')[0], caption


class ImgCaptrionDataset(Dataset):

    def __init__(self, data_dir, tokenizer, caption_path, imgsize=224, train=True):
        super(ImgCaptrionDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted([p for p in os.listdir(data_dir)])
        # self.transform = T.Compose([
        #                 T.Resize((imgsize, imgsize)),
        #                 T.ToTensor()
        #                 ])
        train_transform = T.transforms.Compose([
                            T.Resize((imgsize, imgsize)),
                            RandomRotation(),
                            T.transforms.Lambda(check_rgb),
                            T.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[0.8, 1.5], saturation=[0.2, 1.5]),
                            T.transforms.RandomHorizontalFlip(),
                            T.transforms.ToTensor(),
                            T.transforms.Normalize(MEAN, STD),
                            ])

        val_transform = T.transforms.Compose([
                        T.Resize((imgsize, imgsize)),
                        T.transforms.Lambda(check_rgb),
                        T.transforms.ToTensor(),
                        T.transforms.Normalize(MEAN, STD)
                        ])
        
        if train:
            self.transform = train_transform
        else:
            self.transform = val_transform
        self.tokenizer = tokenizer

        with open(caption_path, 'r') as caption_file:
            self.caption_dict = json.load(caption_file)
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)
        # print(img.size())

        id = [dic["id"] for dic in self.caption_dict["images"] if dic["file_name"]==fname]
        id = id[0]
        caption_ls = [dic["caption"] for dic in self.caption_dict["annotations"] if dic["image_id"] == id]
        caption_ls = [self.tokenizer.encode(caption) for caption in caption_ls]
        captions = [torch.as_tensor(caption.ids, dtype=torch.long) for caption in caption_ls]
        captions = pad_sequence(captions, batch_first=True, padding_value=0)
        if captions.size(0) == 6:
            # print(id)
            captions = captions[1:, :]

        if img.size(0) != 3:
            img = torch.cat((img, img, img))
        # print(captions.shape)
        # print(captions)

        name = fname.split('.')[0]
        
        return img, name, torch.transpose(captions, 0, 1)


class collate_padd(object):

    def __init__(self, max_len=54,pad_id=0):
        self.pad = pad_id
        self.max_len = max_len

    def __call__(self, batch):
        """
        Padds batch of variable lengthes to a fixed length (max_len)
        """
        imgs, names, captions = zip(*batch)
        imgs = torch.stack(imgs)  # (B, 3, 256, 256)
        # for cap in captions:
        #     print(cap.size())
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad) # (B, max_len, captns_num=5)

        pad_right = self.max_len - captions.size(1)
        if pad_right > 0:
            # [B, captns_num, max_seq_len]
            captions = captions.permute(0, 2, 1)  
            captions = nn.ConstantPad1d((0, pad_right), value=self.pad)(captions)
            captions = captions.permute(0, 2, 1) 
        
        return imgs, names, captions

# tokenizer = Tokenizer.from_file("/home/eegroup/ee50525/b09901015/hw3-WEIoct/hw3_data/caption_tokenizer.json")
# caption_dict = json.load(open('/home/eegroup/ee50525/b09901015/hw3-WEIoct/hw3_data/p2_data/val.json'))
# print(type(caption_dict["images"]))
# ls = np.array([len(tokenizer.encode(dic["caption"])) for dic in caption_dict["annotations"]])
# print(np.max(ls))


