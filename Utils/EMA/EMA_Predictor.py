import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .transformer import Encoder
from torch.nn.utils import weight_norm
from .conformer.conformer.encoder import ConformerBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Permute(nn.Module):
    def __init__(self, dim1, dim2):
        super(Permute, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = x.transpose(self.dim1,self.dim2)
        return x

class EMA_Predictor(nn.Module):
    def __init__(self):
        super(EMA_Predictor, self).__init__()
        #encoder1
        self.encoder1 = nn.Sequential(
            nn.Linear(82,256),
            Permute(1,2),
            nn.BatchNorm1d(256),
            Permute(1,2),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.decoder = nn.ModuleList([ConformerBlock(
            encoder_dim=256,
            num_attention_heads=4,
            feed_forward_expansion_factor=4,
            conv_expansion_factor=2,
            feed_forward_dropout_p=0.05,
            attention_dropout_p=0.05,
            conv_dropout_p=0.05,
            conv_kernel_size=31,
            half_step_residual=True,
        ) for _ in range(5)])
        self.pool = weight_norm(nn.ConvTranspose1d(256, 256, kernel_size=3, stride=2, groups=256, padding=1, output_padding=1))
        self.decoder2 = nn.LSTM(input_size=256,hidden_size=256,num_layers=1,dropout=0,bidirectional =True)

        self.decoder3 = nn.Sequential(
            nn.Linear(512,128),
            Permute(1,2),
            nn.BatchNorm1d(128),
            Permute(1,2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(128,10),
        )
        
    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()

        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask

    def forward(self, F0, energy, mels=None):

        """
        src_lens, mel_lens ：   1*B
        texts：  max_src_len*B
        mels：  max_mel_len*80*B
        max_src_len, max_mel_len：    1   表示当前batch最长的序列长度
        """
        #因为数据会自动补零对齐到最长的长度，所以要找出补零的位置：masks
        #增广后的mask

        features = torch.cat((F0, energy, mels),1)
        dec_output = self.encoder1(features.transpose(1,2))

        # dec_output = self.decoder(dec_output)
        for layer in self.decoder:
            dec_output = layer(dec_output)
        dec_output, (_, _) = self.decoder2(dec_output)
        outputs = self.decoder3(dec_output).transpose(1,2)
        return outputs
