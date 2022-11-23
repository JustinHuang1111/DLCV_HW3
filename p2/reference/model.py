import timm
import torch
import torch.nn as nn
from typing import Tuple
from torch.nn import MultiheadAttention
from copy import deepcopy
from utils import PositionalEncoding
# from pprint import pprint
# model_names = timm.list_models(pretrained=True)
# pprint(model_names)

# pretrained_model = timm.create_model('vit_base_patch32_224', pretrained=True)
# print(pretrained_model)
# print(pretrained_model.blocks)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # self.vit = timm.create_model('vit_large_r50_s32_224', pretrained=True) # best
        self.vit = timm.create_model('vit_large_r50_s32_384', pretrained=True)
        # self.vit = timm.create_model('vit_base_patch32_224', pretrained=True)
    
    def forward(self, x):
        x = self.vit.forward_features(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        """
        param:
        d_model:    features size.
                    int
        num_heads:  number of heads in the multiheadattention model.
                    int
        dropout:    dropout value
                    float
        """

        self.dec_self_attn = MultiheadAttention(d_model,
                                                num_heads,
                                                dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 num_heads,
                                                 dropout=dropout)

        self.self_attn_norm = nn.LayerNorm(d_model)
        self.multihead_norm = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.multihead_dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, feedforward_dim),
                                nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                                nn.Linear(feedforward_dim, d_model))

        self.ff_norm = nn.LayerNorm(d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, dec_inputs: torch.Tensor, enc_outputs: torch.Tensor,
                tgt_mask: torch.Tensor,
                tgt_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        param:
        dec_inputs:     Captions to decode
                        Tensor
                        [max_len, batch_size, embed_dim]
        enc_outputs:    Encoded image to decode
                        Tensor
                        [encode_size^2=196, batch_size, embed_dim]
        tgt_mask:       Mask to ensure that decoder doesn't look at future
                        tokens from a given subsequence
                        [max_len , max_len]
        tgt_pad_mask:   Mask to ensure that decoder doesn't attend pad tokens
                        [batch_size , max_len]
        outputs:
        output:         Decoder output
                        Tensor
                        [max_len, batch_size, embed_dim]
        attn:           Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        To be able to do so, I have changed the code at
                        /.virtualenvs/<env_name>/lib/python3.8/site-packages/torch/nn/functional.py
                        line 4818 and changed
                        `return attn_output, attn_output_weights.sum(dim=1) /
                        num_heads` to be
                        `return attn_output, attn_output_weights`
        """
        # print(enc_outputs.size())
        # print(dec_inputs.size())
        # self attention + resedual summation + norm
        output, _ = self.dec_self_attn(dec_inputs,
                                       dec_inputs,
                                       dec_inputs,
                                       attn_mask=tgt_mask,
                                       key_padding_mask=tgt_pad_mask)
        output = dec_inputs + self.self_attn_dropout(output)
        output = self.self_attn_norm(output)  # type: torch.Tensor

        # # self attention + residual + norm + FF
        output2, attns = self.multihead_attn(output, enc_outputs, enc_outputs)
        output = output + self.multihead_dropout(output2)
        output = self.multihead_norm(output)

        output2 = self.ff(output)  # type: torch.Tensor
        output = self.ff_norm(output + self.ff_dropout(output2))
        # print(attns.size())

        return output, attns

class Decoder(nn.Module):
    """
    param:
    layer:          an instance of the EecoderLayer() class
    vocab_size:     the number of vocabulary
                    int
    d_model:        size of features in the transformer inputs
                    int
    num_layers:     the number of decoder-layers
                    int
    max_len:        maximum len pf target captions
                    int
    dropout:        dropout value
                    float
    pad_id:         padding token id
                    float
    """

    def __init__(self,
                 layer: DecoderLayer,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 max_len: int,
                 dropout: float,
                 pad_id: int):
        super().__init__()

        self.pad_id = pad_id

        # Embedding layer + pos encoding
        self.cptn_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        # Make copies of the decoder layer
        self.layers = nn.ModuleList(
            [deepcopy(layer) for _ in range(num_layers)])

        self.dropout = nn.Dropout(p=dropout)

    def get_attn_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        """
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, tgt_cptn: torch.Tensor,
                src_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        param:
        tgt_cptn:   Captions (Transformer target sequence)
                    Tensor
                    [batch_size, max_len-1]
        src_img:    Encoded images (Transformer source sequence)
                    Tensor
                    [encode_size^2, batch_size, image_embed_dim]
        outputs:
        output:     Decoder output
                    Tensor
                    [max_len, batch_size, model_embed_dim]
        attn_all:   Attension weights
                    Tensor
                    [layer_num, batch_size, head_num, max_len-1,
                    encode_size^2]
                    See comments in decoder_layers.DecoderLayer
        """

        # create masks, then pass to decoder
        tgt_pad_mask = (tgt_cptn == self.pad_id)
        tgt_mask = self.get_attn_subsequent_mask(tgt_cptn.size()[1])
        tgt_mask = tgt_mask.to(tgt_cptn.device)

        # encode captions + pos enc
        # (B, max_len) -> (B, max_len, d_model) -> (max_len, B, d_model)
        tgt_cptn = self.cptn_emb(tgt_cptn)  # type: torch.Tensor
        # print(tgt_cptn.shape)
        tgt_cptn = self.dropout(self.pos_emb(tgt_cptn.permute(1, 0, 2)))

        attns_all = []
        for layer in self.layers:
            tgt_cptn, attns = layer(tgt_cptn, src_img, tgt_mask, tgt_pad_mask)
            attns_all.append(attns)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attns_all = torch.stack(attns_all)

        return tgt_cptn, attns_all


class Transformer(nn.Module):
    """
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 dec_ff_dim: int,
                 dec_n_layers: int,
                 dec_n_heads: int,
                 max_len: int,
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super(Transformer, self).__init__()
        decoder_layer = DecoderLayer(d_model=d_model,
                                     num_heads=dec_n_heads,
                                     feedforward_dim=dec_ff_dim,
                                     dropout=dropout)
        self.encoder = Encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.match_size = nn.Linear(1024, d_model) # best
        self.decoder = Decoder(layer=decoder_layer,
                               vocab_size=vocab_size,
                               d_model=d_model,
                               num_layers=dec_n_layers,
                               max_len=max_len,
                               dropout=dropout,
                               pad_id=pad_id)

        self.predictor = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, images: torch.Tensor,
                captions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        param:
        image:      source images
                    [batch_size, encode_size^2=196, image_feature_size=512]
        captions:   target captions
                    [batch_size, max_len-1=51]
        outputs:
        predictions:    Decoder output
                        Tensor
                        [batch_size, max_len, vocab_size]
        attn_all:       Attension weights
                        Tensor
                        [layer_num, batch_size, head_num, max_len,
                        encode_size^2]
                        See comments in decoder_layers.DecoderLayer
        """
        # encode, decode, predict
        images_encoded = self.encoder(images)  # type: torch.Tensor
        images_encoded = self.match_size(images_encoded)
        tgt_cptn, attns = self.decoder(captions, images_encoded.permute(1, 0, 2))
        # print(attns.size())
        predictions = self.predictor(tgt_cptn).permute(1, 0, 2)  # type: torch.Tensor

        return predictions.contiguous(), attns.contiguous()