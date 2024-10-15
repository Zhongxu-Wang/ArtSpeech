#coding:utf-8
import math
import yaml
import random
import numpy as np
from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.EMA.EMA_Predictor import EMA_Predictor
from Utils.RelTransformerEnc import RelTransformerEncoder



class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        elif self.layer_type == 'channelpreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(1, 3), stride=(1, 2), groups=dim_in, padding=(0, 1)))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'channelpreserve':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, (1, 2))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class ResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none', dropout_p=0.2):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample_type = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)
        self.dropout_p = dropout_p

        if self.downsample_type == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.Conv1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1))

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def downsample(self, x):
        if x.shape[-1] % 2 != 0:
            x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
        return F.avg_pool1d(x, 2)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample_type == True:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv1(x)
        x = self.pool(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)

        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))


    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)

class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')

"""
=============================== Models ===============================
"""
class ArtsSpeech(nn.Module):
    def __init__(self, args, stage = "first", distribution={}):
        super().__init__()
        if stage != "first":
            self.arts_encoder = RelTransformerEncoder(n_layers = 4, hidden_channels = args.hidden_dim)
            self.durationPredictor = DurationPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, nlayers=args.n_layer, dropout=args.dropout)
            self.artsPredictor = ArtsPredictor(style_dim=args.style_dim, d_hid=args.hidden_dim, dropout=args.dropout)

        self.text_encoder = RelTransformerEncoder(n_layers = 4, hidden_channels = args.hidden_dim)
        self.style_encoder = StyleEncoder(dim_in=args.dim_in, style_dim=args.style_dim)
        self.decoder = Decoder(dec_dim=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels)

        self.distribution = distribution

    def forward(self, batch, s2s_attn, s2s_attn_mono, step, mode = "train", epoch = 0):
        texts, input_lengths, mels, mel_input_length, _, _, _ = batch
        if step == "first": 
            T_en = self.text_encoder(texts, input_lengths).transpose(1,2)
            if bool(random.getrandbits(1)) and mode == "train":
                T_en = (T_en @ s2s_attn)
            else:
                T_en = (T_en @ s2s_attn_mono)

            F0s_ext, Ns_ext, EMAs_ext, Style = self.style_encoder(mels, mel_input_length, step, self.distribution, epoch)
            mel_len = int(mel_input_length.min().item() / 2 - 1)
            en = []
            f_list = []
            features = torch.cat((Ns_ext, F0s_ext, EMAs_ext, mels), axis=1)
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                en.append(T_en[bib, :, random_start:random_start+mel_len])
                f_list.append(features[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
            en = torch.stack(en)
            f_list = torch.stack(f_list)
            ns_ext = f_list[:,0,:].unsqueeze(1)
            f0s_ext = f_list[:,1,:].unsqueeze(1)
            emas_ext = f_list[:,2:12,:]
            Mel_gt = f_list[:,12:,:].detach()
            Mel_ext = self.decoder(en, Style, f0s_ext, ns_ext, emas_ext)
            return Mel_ext, Mel_gt, F0s_ext, EMAs_ext

        elif step == "second":
            with torch.no_grad():
                T_en = self.text_encoder(texts, input_lengths).transpose(1,2)
            F0_real, N_real, EMA_real, Style = self.style_encoder(mels, mel_input_length, step, self.distribution)
            A_en = self.arts_encoder(texts, input_lengths).transpose(1,2)

            T_en = T_en @ s2s_attn_mono
            A_en = A_en @ s2s_attn_mono
            F0_fake, N_fake, EMA_fake = self.artsPredictor(A_en, Style)
            mel_len = int(mel_input_length.min().item() / 2 - 1)
            T_ens = []
            features = []
            exts = torch.cat((mels, EMA_real, F0_real, N_real, EMA_fake, F0_fake, N_fake), axis=1)
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                T_ens.append(T_en[bib, :, random_start:random_start+mel_len])
                features.append(exts[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
            T_ens = torch.stack(T_ens)
            features = torch.stack(features)
            Mel_gt = features[:,:80,:]
            EMA_gt = features[:,80:90,:]
            F0_gt = features[:,90:91,:]
            N_gt = features[:,91:92,:]
            EMA_pd = features[:,92:102,:]
            F0_pd = features[:,102:103,:]
            N_pd = features[:,103:,:]
            duration_fake = self.durationPredictor(texts, EMA_gt, input_lengths, mel_input_length)
            if epoch <= 20:
                Mel_ext = self.decoder(T_ens, Style, F0_gt, N_gt, EMA_gt)
            else:
                Mel_ext = self.decoder(T_ens, Style, F0_pd, N_pd, EMA_pd)

            return ([Mel_ext, Mel_gt.detach()],
                    [F0_fake, F0_real.detach()],
                    [EMA_fake, EMA_real.detach()],
                    [N_fake, N_real.detach()],
                    duration_fake)
        
        elif step == "test":
            T_en = self.text_encoder(texts, input_lengths).transpose(1,2)
            A_en = self.arts_encoder(texts, input_lengths).transpose(1,2)
            F0s_ext, Ns_ext, EMAs_ext, Style = self.style_encoder(mels, mel_input_length, "second", self.distribution)
            duration = self.durationPredictor(texts, EMAs_ext, input_lengths, mel_input_length)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            T_en = T_en @ pred_aln_trg.unsqueeze(0).to("cuda")
            A_en = A_en @ pred_aln_trg.unsqueeze(0).to("cuda")
            F0_fake, N_fake, EMA_fake = self.artsPredictor(A_en, Style)
            Mel_ext = self.decoder(T_en, Style, F0_fake, N_fake, EMA_fake)
            return Mel_ext

class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48):
        super().__init__()

        self.pitch_extractor = JDCNet(num_class=1, seq_len=192).to("cuda")
        params = torch.load("Utils/JDC/bst.t7", map_location='cuda')['net']
        self.pitch_extractor.load_state_dict(params)
   
        self.ema_extractor = EMA_Predictor().to("cuda")
        params = torch.load("Utils/EMA/200000.pth.tar", map_location='cuda')['model']
        self.ema_extractor.load_state_dict(params)

        self.Mel_block = nn.Sequential(spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1)),
                      ResBlk(dim_in, 2*dim_in, downsample='half'),
                      ResBlk(2*dim_in, 4*dim_in, downsample='half'),
                      ResBlk(4*dim_in, 8*dim_in, downsample='half'),
                      ResBlk(8*dim_in, 8*dim_in, downsample='half'),
                      nn.LeakyReLU(0.2, inplace=True),
                      spectral_norm(nn.Conv2d(8*dim_in, 8*dim_in, 5, 1, 0)),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.AdaptiveAvgPool2d(1))
        self.EMA_block = nn.Sequential(spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1)),
                      ResBlk(dim_in, 2*dim_in, downsample='channelpreserve'),
                      ResBlk(2*dim_in, 4*dim_in, downsample='channelpreserve'),
                      ResBlk(4*dim_in, 4*dim_in, downsample='half'),
                      nn.LeakyReLU(0.2, inplace=True),
                      spectral_norm(nn.Conv2d(4*dim_in, 4*dim_in, 5, 2, 0)),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.AdaptiveAvgPool2d(1))
        self.F0_block = nn.Sequential(spectral_norm(nn.Conv1d(1, dim_in, 3, 1, 1)),
                     ResBlk1d(dim_in, 2*dim_in, downsample=True, dropout_p=0.0),
                     *[ResBlk1d(2*dim_in, 2*dim_in, downsample=True, dropout_p=0.0) for _ in range(3)],
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.AdaptiveAvgPool1d(1))
        self.energy_block = nn.Sequential(spectral_norm(nn.Conv1d(1, dim_in, 3, 1, 1)),
                     ResBlk1d(dim_in, 2*dim_in, downsample=True, dropout_p=0.0),
                     *[ResBlk1d(2*dim_in, 2*dim_in, downsample=True, dropout_p=0.0) for _ in range(3)],
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.AdaptiveAvgPool1d(1))
        self.Mellinear = nn.Linear(8*dim_in, style_dim)
        self.EMAlinear = nn.Linear(4*dim_in, style_dim//2)
        self.F0linear = nn.Linear(2*dim_in, style_dim//4)
        self.Energylinear = nn.Linear(2*dim_in, style_dim//4)

    def style_extractor(self, augment = False, **input):
        batch_size = input["mel"].size(0)
        Mel_style = self.Mellinear(self.Mel_block(input["mel"]).view(batch_size, -1))
        EMA_style = self.EMAlinear(self.EMA_block(input["EMA"]).view(batch_size, -1))
        F0_style = self.F0linear(self.F0_block(input["F0"]).view(batch_size, -1))
        energy_style = self.Energylinear(self.energy_block(input["Energy"]).view(batch_size, -1))
        Style = torch.cat((Mel_style, EMA_style, F0_style, energy_style),axis=1)
        return Style

    def forward(self, mel, mel_input_length, step, distribution, epoch = 20):
        if step == "second" or (step == "first" and epoch<20):
            self.pitch_extractor.eval()
            self.ema_extractor.eval()
            with torch.no_grad():
                n_ext = log_norm(mel.unsqueeze(1))
                f0_ext= self.pitch_extractor(mel.unsqueeze(1))
                ema_ext = self.ema_extractor(f0_ext, n_ext, mel)
        else:
            for module in self.pitch_extractor.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                    module.eval()
            # for module in self.ema_extractor.modules():
            #     if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
            #         module.eval()
            n_ext = log_norm(mel.unsqueeze(1))
            f0_ext  = self.pitch_extractor(mel.unsqueeze(1))
            #ema_ext = self.ema_extractor(f0_ext, n_ext, mel)
            self.ema_extractor.eval()
            with torch.no_grad():
                ema_ext = self.ema_extractor(f0_ext, n_ext, mel)
        n_ext = (n_ext - distribution["energy_mean"])/distribution["energy_std"]
        f0_ext = (f0_ext - distribution["pitch_mean"])/distribution["pitch_std"]
        ema_ext = ((ema_ext.transpose(1, 2)-distribution["EMA_mean"])/distribution["EMA_std"]).transpose(1, 2)
        #---
        # Style = []
        # for bib in range(len(mel_input_length)):
        #     Style.append(self.style_extractor(mel = mel[bib, :, :mel_input_length[bib]].unsqueeze(0).unsqueeze(1),
        #                         EMA = ema_ext[bib, :, :mel_input_length[bib]].unsqueeze(0).unsqueeze(1),
        #                         F0 = f0_ext[bib, :mel_input_length[bib]].unsqueeze(0),
        #                         Energy = n_ext[bib, :mel_input_length[bib]].unsqueeze(0)))
        # Style = torch.stack(Style).squeeze(1)
        #---
        mel_len = int(mel_input_length.min().item()-1)
        f_list = []
        features = torch.cat((n_ext, f0_ext, ema_ext, mel), axis=1)
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item())
            random_start = np.random.randint(0, mel_length - mel_len)
            f_list.append(features[bib, :, random_start:(random_start+mel_len)])
        f_list = torch.stack(f_list)
        ns_ext = f_list[:,0,:].unsqueeze(1)
        f0s_ext = f_list[:,1,:].unsqueeze(1)
        emas_ext = f_list[:,2:12,:].unsqueeze(1)
        Mel_gt = f_list[:,12:,:].unsqueeze(1)
        Style = self.style_extractor(mel = Mel_gt, EMA = emas_ext, F0 = f0s_ext, Energy = ns_ext).squeeze(1)
        return f0_ext, n_ext, ema_ext, Style

class Decoder(nn.Module):
    def __init__(self, dec_dim=512, style_dim=64, residual_dim=64, dim_in=64, dim_out=80):
        super().__init__()
        self.bottleneck_dim = dec_dim * 2

        self.encode = AdainResBlk1d(dec_dim + 128, self.bottleneck_dim, style_dim*2)
        self.F0_conv = weight_norm(nn.Conv1d(1, 32, kernel_size=1))
        self.N_conv = weight_norm(nn.Conv1d(1, 32, kernel_size=1))
        self.EMA_conv = weight_norm(nn.Conv1d(10, 64, kernel_size=1))

        self.asr_res = nn.Sequential(
            weight_norm(nn.Conv1d(dec_dim, residual_dim, kernel_size=1)))

        self.decode = nn.ModuleList(
            [AdainResBlk1d(self.bottleneck_dim + residual_dim + 128, self.bottleneck_dim, style_dim*2),
            AdainResBlk1d(self.bottleneck_dim + residual_dim + 128, self.bottleneck_dim, style_dim*2),
            AdainResBlk1d(self.bottleneck_dim + residual_dim + 128, dec_dim, style_dim*2),
            AdainResBlk1d(dec_dim, dec_dim, style_dim),
            AdainResBlk1d(dec_dim, dec_dim, style_dim),
            AdainResBlk1d(dec_dim, dec_dim, style_dim)])

        self.to_out = nn.Sequential(weight_norm(nn.Conv1d(dec_dim, dim_out, 1, 1, 0)))

    def forward(self, asr, Style, F0, N, EMA):

        Mel_style = Style[:,:256]
        asr = F.interpolate(asr, scale_factor=2, mode='nearest')

        F0 = self.F0_conv(F0)
        N = self.N_conv(N)
        EMA = self.EMA_conv(EMA)

        x = torch.cat([asr, F0, N, EMA], axis=1)
        x = self.encode(x, Style)
        asr_res = self.asr_res(asr)

        for block in self.decode[:3]:
            x = torch.cat([x, asr_res, F0, N, EMA], axis=1)
            x = block(x, Style)

        for block in self.decode[3:]:
            x = block(x, Mel_style)
        x = self.to_out(x)
        return x

class DurationPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, nlayers, dropout=0.1):
        super().__init__()
        self.text_encoder = RelTransformerEncoder(n_layers = 2, hidden_channels = d_hid)
        self.duration = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim//4, dropout_p=dropout),
                                 AdainResBlk1d(d_hid, d_hid, style_dim//4, dropout_p=dropout),
                                 AdainResBlk1d(d_hid, d_hid, style_dim//4, dropout_p=dropout)])
        self.LSTM = nn.LSTM(d_hid, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, 1)

        dim_in = 64
        self.dur_block =  nn.Sequential(spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1)),
            ResBlk(dim_in, 2*dim_in, downsample='channelpreserve'),
            ResBlk(2*dim_in, 2*dim_in, downsample='channelpreserve'),
            ResBlk(2*dim_in, 2*dim_in, downsample='half'),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2*dim_in, 2*dim_in, 5, 2, 0)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.dur_linear = nn.Linear(2*dim_in, style_dim//4)

    def forward(self, texts, style, text_lengths, mel_input_length):
        dur_style = []
        batch_size = len(text_lengths)
        for bib in range(batch_size):
            # dur_style.append(self.dur_linear(self.dur_block(style[bib, :mel_input_length[bib]].unsqueeze(0)).view(1, -1)))
            dur_style.append(self.dur_linear(self.dur_block(style[bib, :].unsqueeze(0)).view(1, -1)))
        dur_style = torch.stack(dur_style).squeeze(1)

        d = self.text_encoder(texts, text_lengths).transpose(1,2)
        for block in self.duration:
            d = block(d, dur_style)
        d = d.transpose(1,2)
        m = self.length_to_mask(text_lengths)
        # predict duration
        input_lengths = text_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            d, input_lengths, batch_first=True, enforce_sorted=False)
        m = m.to(text_lengths.device).unsqueeze(1)
        self.LSTM.flatten_parameters()
        x, _ = self.LSTM(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        x_pad = torch.zeros([x.shape[0], m.shape[-1], x.shape[-1]])
        x_pad[:, :x.shape[1], :] = x
        x = x_pad.to(x.device)
        duration = self.duration_proj(nn.functional.dropout(x, 0.5, training=self.training))
        return duration.squeeze(-1)

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

class ArtsPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, dropout=0.1):
        super().__init__()
        self.shared = AdainResBlk1d(d_hid, d_hid, style_dim*2, dropout_p=dropout)
        d_hid2 = d_hid // 2
        d_hid4 = d_hid // 4
        self.F0 = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim*2, upsample=True, dropout_p=dropout),
                                 AdainResBlk1d(d_hid, d_hid2, style_dim//4, dropout_p=dropout),
                                 AdainResBlk1d(d_hid2, d_hid4, style_dim//4, dropout_p=dropout)])
        self.N = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim*2, upsample=True, dropout_p=dropout),
                                AdainResBlk1d(d_hid, d_hid2, style_dim//4, dropout_p=dropout),
                                AdainResBlk1d(d_hid2, d_hid4, style_dim//4, dropout_p=dropout)])
        self.EMA = nn.ModuleList([AdainResBlk1d(d_hid, d_hid, style_dim*2, upsample=True, dropout_p=dropout),
                                AdainResBlk1d(d_hid, d_hid2, style_dim//2, dropout_p=dropout),
                                AdainResBlk1d(d_hid2, d_hid4, style_dim//2, dropout_p=dropout)])

        self.F0_LSTM = nn.LSTM(d_hid4, d_hid4, 1, batch_first=True, bidirectional=True)
        self.N_LSTM = nn.LSTM(d_hid4, d_hid4, 1, batch_first=True, bidirectional=True)
        self.EMA_LSTM = nn.LSTM(d_hid4, d_hid4, 1, batch_first=True, bidirectional=True)
        self.F0_proj = nn.Conv1d(d_hid2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid2, 1, 1, 1, 0)
        self.EMA_proj = nn.Conv1d(d_hid2, 10, 1, 1, 0)

    def forward(self, A_ens, style):
        EMA_style = style[:,256:384]
        F0_style = style[:,384:448]
        N_style = style[:,448:512]

        A_ens = self.shared(A_ens, style)

        F0 = self.F0[0](A_ens, style)
        for block in self.F0[1:]:
            F0 = block(F0, F0_style)
        F0, _ = self.F0_LSTM(F0.transpose(-1, -2))
        F0 = self.F0_proj(F0.transpose(-1, -2))

        N = self.N[0](A_ens, style)
        for block in self.N[1:]:
            N = block(N, N_style)
        N, _ = self.N_LSTM(N.transpose(-1, -2))
        N = self.N_proj(N.transpose(-1, -2))

        EMA = self.EMA[0](A_ens, style)
        for block in self.EMA[1:]:
            EMA = block(EMA, EMA_style)
        EMA, _ = self.EMA_LSTM(EMA.transpose(-1, -2))
        EMA = self.EMA_proj(EMA.transpose(-1, -2))

        return F0, N, EMA

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2, inplace=True)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        features = []
        for l in self.main:
            x = l(x)
            features.append(x)
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, x):
        out, features = self.get_feature(x)
        out = out.squeeze()  # (batch)
        return out, features

def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model
    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config
    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model
    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()

    return asr_model

def build_model(args, text_aligner, stage = "first", distribution={}):
    artsspeech = ArtsSpeech(args, stage, distribution=distribution)
    discriminator = Discriminator2d(dim_in=args.dim_in, num_domains=1, max_conv_dim=args.hidden_dim)
    return Munch(ArtsSpeech = artsspeech, discriminator = discriminator, text_aligner = text_aligner)

def load_checkpoint(model, optimizer, path, load_only_params=True):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key],False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
    return model, optimizer, epoch, iters
