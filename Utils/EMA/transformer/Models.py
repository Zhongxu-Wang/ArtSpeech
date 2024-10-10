import torch
import torch.nn as nn
import numpy as np
from .Layers import FFTBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]
    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0
    return torch.FloatTensor(sinusoid_table)

class Encoder(nn.Module):
    """ Encoder """
    def __init__(self, encoder_hidden=512, encoder_head=4):
        super(Encoder, self).__init__()
        n_position = 1001
        self.max_seq_len = 1000
        n_layers = 4

        d_k = d_v = (encoder_hidden//encoder_head)
        d_inner = 1024
        kernel_size = [9, 1]
        dropout = 0.2

        self.d_model = encoder_hidden
        self.src_word_emb = nn.Embedding(178, encoder_hidden,)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, encoder_hidden).unsqueeze(0),
            requires_grad=False,)

        self.layer_stack = nn.ModuleList(
            [FFTBlock(
                self.d_model, encoder_head, d_k, d_v, d_inner, kernel_size, dropout=dropout)
            for _ in range(n_layers)])

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask

    def forward(self, src_seq, src_lens):
        mask = self.get_mask_from_lengths(src_lens, max(src_lens))
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )

        return enc_output
