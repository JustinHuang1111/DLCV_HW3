from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import language_evaluation
import clip
import json
from tokenizers import Tokenizer
import os

from dataset import ImgCaptrionDataset, collate_padd
from model import Transformer
from clip_score import CLIPscore
from trainer import seed_everything


def cider_score(evaluator, pred_dict, capt_file):
        with open(capt_file, 'r') as caption_file:
            caption_dict = json.load(caption_file)
            pred = []
            captions = []
        for image_name in pred_dict:
            pred += [pred_dict[image_name]]
            id = [dict["id"] for dict in caption_dict["images"] if dict["file_name"]==f"{image_name}.jpg"]
            id = id[0]
            captions += [[dict["caption"] for dict in caption_dict["annotations"] if dict["image_id"]==id]]
        
        result = evaluator.run_evaluation(pred, captions)
        return result['CIDEr']

if __name__ == "__main__":
    working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"
    config = {
                "seed": 1314520,
                "max_len": 55,
                "batch_size": 32,
                "num_epochs": 100,
                "val_interval": 1,
                "pad_id": 0,
                "vocab_size": 18202,
                "d_model": 768, #768 for beta
                "dec_ff_dim": 2048,
                "dec_n_layers": 6,
                "dec_n_heads": 12, #12 for beta# 8 for alpha
                "dropout": 0.1,
                "save_path": os.path.join(working_dir ,"p2/reference/ckpt"),
                "max_norm": 0.1,
            }


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'using device {device}')

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    pad_id = 0
    BOS = 2
    EOS = 3
    vocab_size = config["vocab_size"]

    tokenizer_file = os.path.join(working_dir ,'hw3_data/caption_tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # --------------- dataloader --------------- #
    print("start to create dataset")
    valid_set = ImgCaptrionDataset(os.path.join(working_dir, "hw3_data/p2_data/images/val"), tokenizer, os.path.join(working_dir, "hw3_data/p2_data/val.json"), 224, False)
    valid_loader = DataLoader(valid_set, collate_fn=collate_padd(config["max_len"], config["pad_id"]), batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    print("finished creating dataset")

    img_dir = os.path.join(working_dir, "hw3_data/p2_data/images/val")
    cap_file = os.path.join(working_dir, "hw3_data/p2_data/val.json")
    # construct models
    transformer = Transformer(config["vocab_size"], config["d_model"], config["dec_ff_dim"], config["dec_n_layers"], config["dec_n_heads"], config["max_len"]-1 ) ## original
    # load
    ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1611.1532/1116-1532_delta_best.pth')
    print(f"loading model from {ckpt_path}")
    state = torch.load(ckpt_path, device)
    transformer.load_state_dict(state['models'])
    print("finished loading model")

    transformer = transformer.to(device)
    transformer.eval()

    # CLIP scorer
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # CIDEr scorer
    cider_eval = language_evaluation.CocoEvaluator()

    pred_caps = {}

    pb = tqdm(valid_loader, leave=False, total=len(valid_loader))
    pb.unit = "step"
    for img, names, cptns_all in pb:
        img: Tensor  # images [1, 3, 256, 256]
        cptns_all: Tensor  # all 5 captions [1, lm, cn=5]

        k = 5
        # start: [1, 1]
        img = img.to(device)
        caption = torch.zeros((1, config["max_len"]), dtype=torch.long)
        caption[:, 0] = BOS
        

        for i in range(config["max_len"] - 1):
            caption = caption.to(device)
            predictions, _ = transformer(img, caption)
            # print(predictions.size())
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            caption[:, i+1] = predicted_id[0]
            
            if predicted_id[0] == EOS:
                break
        
        # print(caption.size())
        pred = tokenizer.decode_batch(caption.tolist())
        pred = pred[0]
        print(pred)
        img_name = names[0]
        pred_caps[img_name] = pred

    pb.close()
    with open(os.path.join(working_dir, f"p2/reference/json_out/eval.json"), "w") as json_out:
        json.dump(pred_caps, json_out)
    

    # print("calculating scores...")
    # # CIDEr score
    # val_cider = cider_score(cider_eval ,pred_caps, cap_file)
    # # CLIP score
    # val_clip = CLIPscore(pred_caps, clip_model, clip_preprocess, img_dir, device)
    # print("finished calculating scores")
    # print(f"[ CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f} ]")
    # with open(os.path.join(f"p2/reference/log/alpha_evaluate_log.txt"), "w") as file:
    #     file.write(f"[ CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f} ]")
                                
        
