import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from model import Transformer
from PIL import Image
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from trainer import seed_everything

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--tok",
    "-t",
    type=str,
    default=os.path.join(working_dir, "hw3_data/caption_tokenizer.json"),
    help="tokenizer path to load",
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default=os.path.join(
        working_dir, "p2/reference/ckpt/1711.0215/1117-0215_epsilon_50_best.pth"
    ),
    help="Model path to load",
)
parser.add_argument("--cuda", "-c", type=str, default="cuda:1", help="choose a cuda")
config = {
    "seed": 1314520,
    "max_len": 55,
    "batch_size": 32,
    "outpath": os.path.join(working_dir, "p3"),
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

args = parser.parse_args()
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")

SEED = config["seed"]
seed_everything(SEED)
tokenizer_file = args.tok
tokenizer = Tokenizer.from_file(tokenizer_file)


def check_rgb(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


class ImgDataset(Dataset):
    def __init__(self, data_dir, imgsize=384):
        super(ImgDataset, self).__init__()
        self.data_dir = data_dir
        self.files = sorted(
            [
                p
                for p in os.listdir(data_dir)
                if p == "000000392315.jpg" or p == "000000302838.jpg"
            ]
        )
        self.transform = T.transforms.Compose(
            [
                T.Resize((imgsize, imgsize)),
                T.transforms.Lambda(check_rgb),
                T.transforms.ToTensor(),
                T.transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)

        return img, fname


img_dir = os.path.join(working_dir, "hw3_data/p2_data/images/val")
test_set = ImgDataset(
    img_dir,
    384,
)
test_loader = DataLoader(
    test_set,
    batch_size=1,
    pin_memory=True,
    shuffle=False,
    num_workers=0,
)


transformer = Transformer(
    config["vocab_size"],
    config["d_model"],
    config["dec_ff_dim"],
    config["dec_n_layers"],
    config["dec_n_heads"],
    config["max_len"] - 1,
).to(device)

state = torch.load(args.model, device)
transformer_state = state["models"]
transformer.load_state_dict(transformer_state)
seq_mask = []


def remove_pad(tensor, mask):
    out = []
    max_len = tensor.size(1)
    is3d = len(tensor.size()) == 3

    if is3d:
        tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
        mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

    for i in range(tensor.size(0)):
        unpad = list(torch.masked_select(tensor[i], mask=mask[i]))
        unpad = [int(e) for e in unpad]
        out.append(unpad)

    if is3d:
        out = [out[i : i + 5] for i in range(0, len(out), 5)]

    return out


def create_caption_and_mask(start_token, max_length, device):
    caption_template = torch.zeros((1, max_length), dtype=torch.long, device=device)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


def multihead_hook(_, _, feat_out):
    _, mask = feat_out
    seq_mask.append(mask)


seq_mask = []
transformer.decoder.layers[-1].multihead_attn.register_forward_hook(hook=multihead_hook)
start_token = torch.tensor(2)
caption, _ = create_caption_and_mask(start_token, config["max_len"], device)
output_dir = config["outpath"]


transformer.eval()
EOS = 3
BOS = 2
pb = tqdm(test_loader, leave=False, total=len(test_loader))
for imgs, [fname] in pb:
    imgs: Tensor  # images [B, 3, 224, 224]
    imgs = imgs.to(device)
    max_len = 55
    imgs: Tensor  # images [1, 3, 256, 256]
    cptns_all: Tensor  # all 5 captions [1, lm, cn=5]

    k = 5
    # seq_preds = beam_search(imgs, k, transformer, max_len, self.device)
    # start: [1, 1]
    start = torch.full(
        size=(1, 1),
        fill_value=BOS,
        dtype=torch.long,
        device=device,
    )

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
        current_preds = torch.cat([start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)

    seq_preds = []
    seq_log_probs = []
    seq_attns = []
    while current_preds.size(1) <= (max_len - 2) and k > 0 and current_preds.nelement():
        with torch.no_grad():

            imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
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

            current_preds = torch.cat((current_preds[prev_seq_k], next_word_id), dim=1)

        # find predicted sequences that ends
        seqs_end = (next_word_id == EOS).view(-1)
        if torch.any(seqs_end):
            seq_preds.extend(seq.tolist() for seq in current_preds[seqs_end])
            seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
            # get last layer, mean across transformer heads
            h, w = 1, 145
            attns = attns[-1].mean(dim=1).view(k, -1, h, w)
            # [k, S, h, w]
            seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

            k -= torch.sum(seqs_end)
            current_preds = current_preds[~seqs_end]
            log_prob_topk = log_prob_topk[~seqs_end]
            # current_attns = current_attns[~seqs_end]

    # Sort predicted captions according to seq_log_probs
    specials = [config["pad_id"], BOS, EOS]
    seq_preds, seq_attns, seq_log_probs = zip(
        *sorted(
            zip(seq_preds, seq_attns, seq_log_probs),
            key=lambda tup: -tup[2],
        )
    )
    preds = tokenizer.decode_batch(seq_preds)[0]  # preds: string

    # Visualization
    seq_len = len(preds.split())
    seq_mask = seq_mask[-1][0, : seq_len + 1, 1:]

    """
    encoder use vit 32_384 => 12 x 12 pathes, therefore /12 is needed to reshape 
    """
    seq_mask = torch.reshape(seq_mask, (len(seq_mask), int(seq_mask.shape[1] / 12), 12))
    seq_mask = seq_mask.detach().cpu().numpy()
    image = Image.open(os.path.join(img_dir, fname))
    pic = [image]
    for i in range(len(seq_mask)):
        pic.append(Image.fromarray(seq_mask[i]).resize(image.size))

    fig, ax = plt.subplots(math.ceil(len(pic) / 5), 5, figsize=(16, 8))
    ax[0][0].imshow(pic[0])
    ax[0][0].axis("off")
    ax[0][0].set_title("<BOS>")
    fig.suptitle(preds.capitalize(), fontsize=16)
    ss = preds.capitalize().split(" ")

    for i in range(1, math.ceil(len(pic) / 5) * 5):
        row = math.floor(i / 5)
        col = i % 5
        if i < len(pic):
            ax[row][col].set_title("<EOS>")
            if i - 1 < len(ss):
                ax[row][col].set_title(ss[i - 1])
            if i - 1 == len(ss) - 1:
                ax[row][col].set_title(ss[i - 1][:-1])
            ax[row][col].imshow(pic[0])
            ax[row][col].imshow(pic[i] / np.max(pic[i]), alpha=0.7, cmap="rainbow")
        ax[row][col].axis("off")
    fname = fname.split(".")[0]
    plt.savefig(os.path.join(output_dir, f"{fname}_p3.png"))
    seq_mask = []
