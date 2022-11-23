from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader

from model import Transformer
from trainer import Trainer, seed_everything
from dataset import ImgCaptrionDataset, collate_padd, CocoCaption
from tokenizers import Tokenizer
import os

import argparse


working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"

parser = argparse.ArgumentParser()
parser.add_argument("--tok", "-t", type=str, default=os.path.join(working_dir, "hw3_data/caption_tokenizer.json"), help="tokenizer path to load")
parser.add_argument("--model", "-m", type=str, default=None, help="Model path to load")
config = {
    "seed": 1314520,
    "max_len": 55,
    "num_epochs": 100,
    "val_interval": 2,
    "pad_id": 0,
    "vocab_size": 18202,
    "d_model": 768,
    "dec_ff_dim": 2048,
    "dec_n_layers": 6,
    "dec_n_heads": 12,
    "dropout": 0.1,
    "save_path": os.path.join(working_dir ,"p2/reference/ckpt")
}


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'using device {device}')
args = parser.parse_args()
tokenizer_file = args.tok
tokenizer = Tokenizer.from_file(tokenizer_file)

print("start to create datasets")
train_set = ImgCaptrionDataset(os.path.join(working_dir, "hw3_data/p2_data/images/train"), tokenizer, os.path.join(working_dir, "hw3_data/p2_data/train.json"), 384)
valid_set = ImgCaptrionDataset(os.path.join(working_dir, "hw3_data/p2_data/images/val"), tokenizer, os.path.join(working_dir, "hw3_data/p2_data/val.json"), 384, False)

# train_set = CocoCaption(
#             os.path.join(working_dir, "hw3_data/p2_data/images/train"),
#             os.path.join(working_dir, "hw3_data/p2_data/train.json"),
#             max_length=config['max_len'],
#             limit=-1,
#             mode="training",
#         )
# valid_set = CocoCaption(
#             os.path.join(working_dir, "hw3_data/p2_data/images/val"),
#             os.path.join(working_dir, "hw3_data/p2_data/val.json"),
#             max_length=config['max_len'],
#             limit=-1,
#             mode="validation",
#         )

train_loader = DataLoader(train_set, collate_fn=collate_padd(config["max_len"], config["pad_id"]), batch_size=32, pin_memory=True, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, collate_fn=collate_padd(config["max_len"], config["pad_id"]), batch_size=32, pin_memory=True, shuffle=True, num_workers=0)
print("finished creating datasets")

print("start to create model")
transformer = Transformer(config["vocab_size"], config["d_model"], config["dec_ff_dim"], config["dec_n_layers"], config["dec_n_heads"], config["max_len"]-1 )
print("finished creating model")


SEED = config["seed"]
seed_everything(SEED)

trainer = Trainer(transformer, device, config["num_epochs"], config["val_interval"], 5.0, config["save_path"], config["pad_id"])
name = "epsilon"
print("start training")
trainer.run_train(name, transformer, tokenizer, (train_loader, valid_loader), SEED,)