import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from dataset import InfDataset
from model import Transformer
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"


# reference:
# https://github.com/zarzouram/image_captioning_with_transformers/blob/main/code/inference_test.py
def cider_score(evaluator, pred_dict, capt_file):
    with open(capt_file, "r") as caption_file:
        caption_dict = json.load(caption_file)
        pred = []
        captions = []
    for image_name in pred_dict:
        pred += [pred_dict[image_name]]
        id = [
            dict["id"]
            for dict in caption_dict["images"]
            if dict["file_name"] == f"{image_name}.jpg"
        ]
        id = id[0]
        captions += [
            [
                dict["caption"]
                for dict in caption_dict["annotations"]
                if dict["image_id"] == id
            ]
        ]

    result = evaluator.run_evaluation(pred, captions)
    return result["CIDEr"]


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate test set with pretrained model."
    )
    parser.add_argument("--token", "-t", type=str, default="./", help="token")
    parser.add_argument("--imgpath", "-i", type=str, default="./", help="output path")
    parser.add_argument("--outpath", "-o", type=str, default="./", help="output path")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./p2/p2.pth",
        help="model path",
    )
    return parser.parse_args()


def inference():
    config = {
        "seed": 1314520,
        "max_len": 55,
        "batch_size": 32,
        "num_epochs": 100,
        "val_interval": 1,
        "pad_id": 0,
        "vocab_size": 18202,
        "d_model": 768,  # 768
        "dec_ff_dim": 2048,
        "dec_n_layers": 6,
        "dec_n_heads": 12,  # 12 # 8
        "dropout": 0.1,
        "max_norm": 0.1,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    BOS = 2
    EOS = 3

    tokenizer_file = os.path.join(args.token)
    tokenizer = Tokenizer.from_file(tokenizer_file)

    # --------------- dataloader --------------- #
    valid_set = InfDataset(
        args.imgpath,
        tokenizer,
        384,
        False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        pin_memory=True,
        shuffle=True,
        num_workers=2,
    )

    # construct models
    transformer = Transformer(
        config["vocab_size"],
        config["d_model"],
        config["dec_ff_dim"],
        config["dec_n_layers"],
        config["dec_n_heads"],
        config["max_len"] - 1,
    )  ## original
    # load
    # ckpt_path = '/home/eegroup/ee50525/b09901015/hw3-WEIoct/p2/ckpt/1511.1029/1115-1029_vit_large_r50_s32_384_12heads_best.pth'
    # ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1611.1529/1116-1529_gamma_best.pth') # best
    # best CIDEr: 0.934278617992512 | CLIPScore: 0.7123361569399804
    # ckpt_path = os.path.join(working_dir, 'p2/reference/ckpt/1711.0033/1117-0033_zeta_16_best.pth') # best CIDEr: 0.8996480976581225 | CLIPScore: 0.7031726272096842

    state_dict = torch.load(
        args.model,
        map_location="cpu",
    )
    transformer.load_state_dict(state_dict["models"])

    transformer = transformer.to(device)
    transformer.eval()

    predict = {}

    pb = tqdm(valid_loader, leave=False, total=len(valid_loader))
    pb.unit = "step"
    for imgs, names in pb:
        imgs: Tensor  # images [1, 3, 256, 256]

        k = 5
        # start: [1, 1]
        imgs = imgs.to(device)
        start = torch.full(size=(1, 1), fill_value=BOS, dtype=torch.long, device=device)

        with torch.no_grad():

            imgs_enc = transformer.encoder(imgs)
            imgs_enc = transformer.match_size(imgs_enc)
            logits, attns = transformer.decoder(start, imgs_enc.permute(1, 0, 2))
            logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()

            logits: Tensor  # [k=1, 1, vsc]
            attns: Tensor  # [ln, k=1, hn, S=1, is]

            log_prob = F.log_softmax(logits, dim=2)
            log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)

            current_preds = torch.cat(
                [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1
            )

        seq_preds = []
        seq_log_probs = []
        seq_attns = []
        while (
            current_preds.size(1) <= (config["max_len"] - 2)
            and k > 0
            and current_preds.nelement()
        ):
            with torch.no_grad():
                imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                # print('size of current_pred', current_preds.size())
                # print('size of imgs_expand', imgs_expand.size())
                # [k, is, ie]
                logits, attns = transformer.decoder(
                    current_preds, imgs_expand.permute(1, 0, 2)
                )
                logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()
                # next word prediction
                log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                # log_prob: [k, vsc]
                log_prob = log_prob + log_prob_topk.view(k, 1)
                # top k probs in log_prob[k, vsc]
                log_prob_topk, indxs_topk = log_prob.view(-1).topk(k, sorted=True)
                # indxs_topk are a flat indecies, convert them to 2d indecies:
                # i.e the top k in all log_prob: get indecies: K, next_word_id
                prev_seq_k, next_word_id = np.unravel_index(
                    indxs_topk.cpu(), log_prob.size()
                )
                next_word_id = torch.as_tensor(next_word_id).to(device).view(k, 1)
                # prev_seq_k [k], next_word_id [k]

                current_preds = torch.cat(
                    (current_preds[prev_seq_k], next_word_id), dim=1
                )
                # current_attns = torch.cat(
                #     (current_attns[prev_seq_k], attns[prev_seq_k]), dim=1)

            # find predicted sequences that ends
            seqs_end = (next_word_id == EOS).view(-1)
            if torch.any(seqs_end):
                seq_preds.extend(seq.tolist() for seq in current_preds[seqs_end])
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

        # In some cases, seq_attns, seq_preds and
        if seq_attns and seq_preds and seq_log_probs:
            seq_preds, seq_attns, seq_log_probs = zip(
                *sorted(
                    zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]
                )
            )

            # print(seq_preds)
            pred = tokenizer.decode_batch(seq_preds)
            pred = pred[0]
        else:
            pred = "a"
        print(pred)
        img_name = names[0]
        predict[img_name] = pred

    pb.close()

    with open(args.outpath, "w") as json_out:
        json.dump(predict, json_out)


if __name__ == "__main__":
    args = get_args()
    inference()
