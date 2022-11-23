from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F

import language_evaluation
import clip
import json
from tokenizers import Tokenizer
import os
from dataset import ImgCaptrionDataset, collate_padd
from model import Transformer
from clip_score import CLIPscore
from trainer import seed_everything


working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"

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
    
    # config = {
    #             "seed": 9001,
    #             "max_len": 55,
    #             "batch_size": 32,
    #             "num_epochs": 100,
    #             "val_interval": 1,
    #             "pad_id": 0,
    #             "vocab_size": 18202,
    #             "d_model": 768, #768
    #             "dec_ff_dim": 2048,
    #             "dec_n_layers": 5,
    #             "dec_n_heads": 12, #12 # 8
    #             "dropout": 0.1,
    #             "save_path": os.path.join(working_dir, "p2/reference/ckpt"),
    #             "max_norm": 0.1,
    #         } best
    
    config = {
                "seed": 1314520,
                "max_len": 55,
                "batch_size": 32,
                "num_epochs": 100,
                "val_interval": 1,
                "pad_id": 0,
                "vocab_size": 18202,
                "d_model": 768, #768
                "dec_ff_dim": 2048,
                "dec_n_layers": 6,
                "dec_n_heads": 12, #12 # 8
                "dropout": 0.1,
                "save_path": os.path.join(working_dir, "p2/reference/ckpt"),
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

    tokenizer_file = os.path.join(working_dir, 'hw3_data/caption_tokenizer.json')
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # --------------- dataloader --------------- #
    print("start to create dataset")
    valid_set = ImgCaptrionDataset(os.path.join(working_dir, "hw3_data/p2_data/images/val"), tokenizer, os.path.join(working_dir, "hw3_data/p2_data/val.json"), 384, False)
    valid_loader = DataLoader(valid_set, collate_fn=collate_padd(config["max_len"], config["pad_id"]), batch_size=1, pin_memory=True, shuffle=True, num_workers=0)
    print("finished creating dataset")

    img_dir = os.path.join(working_dir, "hw3_data/p2_data/images/val")
    cap_file =os.path.join(working_dir, "hw3_data/p2_data/val.json")
    # construct models
    transformer = Transformer(config["vocab_size"], config["d_model"], config["dec_ff_dim"], config["dec_n_layers"], config["dec_n_heads"], config["max_len"]-1 ) ## original
    # load
    # ckpt_path = '/home/eegroup/ee50525/b09901015/hw3-WEIoct/p2/ckpt/1511.1029/1115-1029_vit_large_r50_s32_384_12heads_best.pth'
    # ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1611.1529/1116-1529_gamma_best.pth') # best
    ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1711.0215/1117-0215_epsilon_50_best.pth') # best CIDEr: 0.934278617992512 | CLIPScore: 0.7123361569399804
    # ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1711.0033/1117-0033_zeta_16_best.pth') # best CIDEr: 0.8996480976581225 | CLIPScore: 0.7031726272096842

    print(f"loading model from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    transformer.load_state_dict(state['models'])
    print("finished loading model")

    transformer = transformer.to(device)
    transformer.eval()

    # CLIP scorer
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    # CIDEr scorer
    cider_eval = language_evaluation.CocoEvaluator()

    # Sizes:
    # B:   batch_size
    # is:  image encode size^2: image seq len: [default=196]
    # ie:  image features dim: [default=512]
    # vsc: vocab_size: vsz
    # lm:  max_len: [default=52]
    # S:   length of the predicted captions; increases incremently
    # cn:  number of captions: [default=5]
    # hn:  number of transformer heads: [default=8]
    # ln:  number of layers
    # k:   Beam Size
    max_len = config["max_len"]

    pred_caps = {}

    pb = tqdm(valid_loader, leave=False, total=len(valid_loader))
    pb.unit = "step"
    for imgs, names, cptns_all in pb:
        imgs: Tensor  # images [1, 3, 256, 256]
        cptns_all: Tensor  # all 5 captions [1, lm, cn=5]

        k = 5
        # start: [1, 1]
        imgs = imgs.to(device)
        start = torch.full(size=(1, 1),
                           fill_value=BOS,
                           dtype=torch.long,
                           device=device)
       
        with torch.no_grad():

            imgs_enc = transformer.encoder(imgs)
            imgs_enc = transformer.match_size(imgs_enc)
            logits, attns = transformer.decoder(start, imgs_enc.permute(1, 0, 2))
            logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()

            logits: Tensor  # [k=1, 1, vsc]
            attns: Tensor  # [ln, k=1, hn, S=1, is]

            log_prob = F.log_softmax(logits, dim=2)
            log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)
            # log_prob_topk [1, 1, k]
            # indices_topk [1, 1, k]
            current_preds = torch.cat(
                [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)
            # current_preds: [k, S]

            # get last layer, mean across transformer heads
            # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
            # current_attns = attns.repeat_interleave(repeats=k, dim=0)
            # [k, s=1, h, w]

        seq_preds = []
        seq_log_probs = []
        seq_attns = []
        while current_preds.size(1) <= (
                max_len - 2) and k > 0 and current_preds.nelement():
            with torch.no_grad():
                imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                # print('size of current_pred', current_preds.size())
                # print('size of imgs_expand', imgs_expand.size())
                # [k, is, ie]
                logits, attns = transformer.decoder(current_preds, imgs_expand.permute(1, 0, 2))
                logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()
                # logits: [k, S, vsc]
                # attns: # [ln, k, hn, S, is]
                # get last layer, mean across transformer heads
                # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # # [k, S, h, w]
                # attns = attns[:, -1].view(k, 1, h, w)  # current word

                # next word prediction
                log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                # log_prob: [k, vsc]
                log_prob = log_prob + log_prob_topk.view(k, 1)
                # top k probs in log_prob[k, vsc]
                log_prob_topk, indxs_topk = log_prob.view(-1).topk(k,
                                                                   sorted=True)
                # indxs_topk are a flat indecies, convert them to 2d indecies:
                # i.e the top k in all log_prob: get indecies: K, next_word_id
                prev_seq_k, next_word_id = np.unravel_index(
                    indxs_topk.cpu(), log_prob.size())
                next_word_id = torch.as_tensor(next_word_id).to(device).view(
                    k, 1)
                # prev_seq_k [k], next_word_id [k]

                current_preds = torch.cat(
                    (current_preds[prev_seq_k], next_word_id), dim=1)
                # current_attns = torch.cat(
                #     (current_attns[prev_seq_k], attns[prev_seq_k]), dim=1)

            # find predicted sequences that ends
            seqs_end = (next_word_id == EOS).view(-1)
            if torch.any(seqs_end):
                seq_preds.extend(seq.tolist()
                                 for seq in current_preds[seqs_end])
                seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                # get last layer, mean across transformer heads
                print(attns.shape)
                # h, w = 1, 50 best
                h, w = 1, 145
                attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # [k, S, h, w]
                seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

                k -= torch.sum(seqs_end)
                current_preds = current_preds[~seqs_end]
                log_prob_topk = log_prob_topk[~seqs_end]
                # current_attns = current_attns[~seqs_end]

        # Sort predicted captions according to seq_log_probs
        specials = [pad_id, BOS, EOS]
        if seq_attns and seq_preds and seq_log_probs:
            seq_preds, seq_attns, seq_log_probs = zip(*sorted(
                zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))

        # print(seq_preds)
            pred = tokenizer.decode_batch(seq_preds)
            pred = pred[0]
        else:
            pred = "a"
        print(pred)
        img_name = names[0]
        pred_caps[img_name] = pred        

    pb.close()

    with open(os.path.join(working_dir, f"p2/reference/json_out/beamEval_epsilon.json"), "w") as json_out:
        json.dump(pred_caps, json_out)
    

    # print("calculating scores...")
    # # CIDEr score
    # val_cider = cider_score(cider_eval ,pred_caps, cap_file)
    # # CLIP score
    # val_clip = CLIPscore(pred_caps, clip_model, clip_preprocess, img_dir, device)
    # print("finished calculating scores")
    # print(f"[ CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f} ]")
    
    # with open(f"/home/eegroup/ee50525/b09901015/hw3-WEIoct/p2/log/beamSearch.txt", "w") as file:
    #     file.write(f"[ CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f} ]")
