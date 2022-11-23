from datetime import datetime
from typing import List, Union, Optional
from tqdm.auto import tqdm
from pathlib import Path
import json

import numpy as np
import os, random
import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import clip

from clip_score import CLIPscore
import language_evaluation

working_dir = "/home/eegroup/ee50526/b09901062/hw3-JustinHuang1111"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer():

    def __init__(self,
                 transformer,
                 device,
                 epochs: int,
                 val_interval: int,
                 grad_clip: float,
                 checkpoints_path: str,
                 pad_id: int,
                 resume: Optional[str] = None) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.resume = resume
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop trianing if the model doesn't improve for n-validation epochs
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        # start tune embeddings after n training epochs have beed passed
        self.pad_id = pad_id

        # criterion, optims and schedulers
        self.criterion = nn.CrossEntropyLoss(ignore_index = pad_id).to(device)
        
        self.transformer = transformer.to(self.device)
        self.transformer_optim = Adam(self.transformer.parameters(), lr = 1e-4, weight_decay=1e-5)
        self.transformer_scheduler = StepLR(self.transformer_optim, step_size=1, gamma=0.75)

        if self.resume:
            self.load_checkpoint(os.path.join(working_dir, ""))

        # CLIP scorer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

        # CIDEr scorer
        self.cider_eval = language_evaluation.CocoEvaluator()


        # Some coeffecient
        self.grad_clip_c = grad_clip  # gradient clip coeffecient

        if resume is None:
            time_tag = str(datetime.now().strftime("%d%m.%H%M"))
        else:
            time_tag = Path(resume).parent

        # make folder for the experment
        checkpoints_path = Path(checkpoints_path) / f"{time_tag}"  # type: Path
        checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path = str(checkpoints_path)

    def loss_fn(self, logits: Tensor, targets: Tensor,
                attns: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))
        return loss 
    
    def remove_pad(self, tensor, mask):
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
            out = [out[i:i + 5] for i in range(0, len(out), 5)]
        
        return out
            
    def cider_score(self, pred_dict, capt_file):
        with open(capt_file, 'r') as caption_file:
            caption_dict = json.load(caption_file)
            pred = []
            captions = []
        for image_name in pred_dict:
            pred += [pred_dict[image_name]]
            id = [dict["id"] for dict in caption_dict["images"] if dict["file_name"]==f"{image_name}.jpg"]
            id = id[0]
            captions += [[dict["caption"] for dict in caption_dict["annotations"] if dict["image_id"]==id]]
        
        result = self.cider_eval.run_evaluation(pred, captions)
        
        return result['CIDEr']

    def clip_gradient(self):
        for group in self.transformer_optim.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-self.grad_clip_c,
                                            self.grad_clip_c)

    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)


    def load_checkpoint(self, load_path):
        print("loading model from ", load_path)
        # load checkopoint
        state = torch.load(load_path, self.device)
        transformer_state = state["models"]
        transformer_optim_state = state["optims"]
        transformer_scheduler_state = state["schedulers"]

        # load state dicts
        self.transformer.load_state_dict(transformer_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)

        self.transformer = self.transformer.to(self.device)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        return

    def save_checkpoint(self, name, model):

        Transformer_state = model.state_dict()
        transformer_optim_state = self.transformer_optim.state_dict()
        transformer_scheduler_state = self.transformer_scheduler.state_dict()

        state = {
            "models": Transformer_state,
            "optims": transformer_optim_state,
            "schedulers": transformer_scheduler_state,
            "phase": self.train,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
        }

        # set save path
        file_name = f"{name}_best"
        save_path = os.path.join(self.checkpoints_path, f"{file_name}.pth")

        torch.save(state, save_path)


    def run_train(self, name, transformer, tokenizer,
            data_iters, SEED: int):
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        time = datetime.now().strftime("%m%d-%H%M_")
        train_name = time + name

        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        best_clip = 0
        best_cider = 0

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {self.epoch:02d}")

            
            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                transformer.train()
                data_iter = data_iters[0]
                img_dir = os.path.join(working_dir, "hw3_data/p2_data/images/train")
                cap_file = os.path.join(working_dir, "hw3_data/p2_data/train.json")
                # fine tune the embeddings layer after some epochs and add the
                # parameters to the optimizer
            else:
                transformer.eval()
                data_iter = data_iters[1]
                img_dir = os.path.join(working_dir, "hw3_data/p2_data/images/val")
                cap_file = os.path.join(working_dir, "hw3_data/p2_data/val.json")
                val_clip = []
                val_cider = []

            # Iterate over data
            # prgress bar
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"
            for step, (imgs, names, cptns_all) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 224, 224]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]
                # print(cptns_all)  

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                # if not self.train:
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)
                # else:
                #     cptns = cptns_all.to(self.device)

                # zero the parameter gradients
                self.transformer_optim.zero_grad()

                with torch.set_grad_enabled(self.train):
                    # embed images using CNN then get logits prediction using
                    # the transformer
            
                    logits, attns = transformer(imgs, cptns[:, :-1])
                    # print(attns.size())
                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:], attns)

                    # in train, gradient clip + update weights
                    if self.train:
                        loss.backward()
                        self.clip_gradient()
                        self.transformer_optim.step()
                
                # get predections then alculate some metrics
                preds = torch.argmax(logits, dim=2).cpu()  # predections
                # print(preds)
                # print(preds.size())
                # print(cptns_all.size())

                targets = cptns_all[:, 1:]  # remove <SOS>
                mask = targets != self.pad_id
                targets = self.remove_pad(targets, mask)
                preds = self.remove_pad(preds, mask[:, :, idx])
                preds = tokenizer.decode_batch(preds)

                ## write to json file
                # print("targets: \n",targets)
                # print("preds: \n", preds)
                pred_caps = {}
                for i in range(len(preds)):
                    img_name = names[i]
                    pred_caps[img_name] = preds[i]

                if self.train:
                    if step == len(pb) - 1:
                        with open(os.path.join(working_dir, f"p2/reference/json_out/{train_name}.json"), "w") as json_out:
                            json.dump(pred_caps, json_out)
                        
                        # CIDEr score
                        cider_score = self.cider_score(pred_caps, cap_file)
                        
                        # CLIP score
                        clip_score = CLIPscore(pred_caps, self.clip_model, self.clip_preprocess, img_dir, self.device)
                        ## write to log file
                        with open(
                            os.path.join(working_dir, f"p2/reference/log/{train_name}_log.txt"), "a"
                        ) as f:
                            f.write(
                                f"[ {self.epoch + 1:03d}/{self.epochs_num:03d} ] loss = {loss:.4f} | CIDEr score: {cider_score:.4f} | CLIP score: {clip_score:.4f}\n"
                            )
                            print(
                                f"[ {self.epoch + 1:03d}/{self.epochs_num:03d} ] loss = {loss:.4f} | CIDEr score: {cider_score:.4f} | CLIP score: {clip_score:.4f}]"
                            )
                else:
                    val_clip.append(CLIPscore(pred_caps, self.clip_model, self.clip_preprocess, img_dir, self.device))

                    # CIDEr score
                    cider_score = self.cider_score(pred_caps, cap_file)
                    val_cider.append(cider_score)




                # step ended
                # update progress bar
                # pb.update(1)

            # self.metrics_tracker.update(phases[self.train])  # save metrics
            if not self.train:
                val_cider = np.mean(val_cider)
                val_clip = np.mean(val_clip)
                with open(
                            os.path.join(working_dir, f"p2/reference/log/{train_name}_log.txt"), "a"
                        ) as f:
                        f.write(
                            f"[ at epoch {self.epoch}: CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ] [current best CLIP: {best_clip:.4f} | current best CIDEr: {best_cider:.4f}]\n"
                        )
                        print(
                            f"[ at epoch {self.epoch}: CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )
                if val_cider > best_cider and val_clip > best_clip:
                    best_cider = val_cider
                    best_clip = val_clip
                    self.save_checkpoint(train_name + "_" + str(self.epoch), transformer)
                    with open(
                            os.path.join(working_dir, f"p2/reference/log/{train_name}_log.txt"), "a"
                        ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )

                if val_clip > best_clip and val_cider > 0.87:
                    best_clip = val_clip
                    self.save_checkpoint(train_name + "_" + str(self.epoch), transformer)
                    with open(
                            os.path.join(working_dir, f"p2/reference/log/{train_name}_log.txt"), "a"
                        ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )
                if val_cider > best_cider and val_clip > 0.7:
                    best_cider = val_cider
                    self.save_checkpoint(train_name + "_" + str(self.epoch), transformer)
                    with open(
                            os.path.join(working_dir, f"p2/reference/log/{train_name}_log.txt"), "a"
                        ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )
            elif self.epoch % 5 ==0:
                self.transformer_scheduler.step()



            # epoch ended
            self.set_phase()  # set train or val phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                break