import os
import yaml
import json
import shutil
import numpy as np
import torch
import click
import random
import warnings
import os.path as osp
from tqdm import tqdm
from torch import nn
from munch import Munch
from attrdict import AttrDict
import torch.nn.functional as F
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter

from utils import *
from optimizers import build_optimizer
from meldataset import build_dataloader
#from monotonic_align import maximum_path
#from monotonic_align import mask_from_lens
from models import load_ASR_models, build_model, load_checkpoint
from Vocoder.vocoder import Generator

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

import logging
from logging import StreamHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yaml', type=str)
def main(config_path):

    config = yaml.safe_load(open(config_path))
    MAS_type = config['MAS_type']
    global maximum_path, mask_from_lens
    if MAS_type == 'v1':  # Cython-free super-monotonic-align-v1 (https://github.com/supertone-inc/super-monotonic-align)
        import S_monotonic_align
        maximum_path = S_monotonic_align.maximum_path1 # I/O : [batch_size=B, text_length=T, audio_length=S]
        mask_from_lens = S_monotonic_align.mask_from_lens
    elif MAS_type == 'triton':  # super-monotonic-align-triton (needs triton)
        import S_monotonic_align_Triton
        maximum_path = S_monotonic_align_Triton.maximum_path  # same as above
        mask_from_lens = S_monotonic_align_Triton.mask_from_lens
    elif MAS_type == 'legacy':  # the previous one (https://github.com/resemble-ai/monotonic_align)
        import monotonic_align
        maximum_path = monotonic_align.maximum_path # I/O : [batch_size=B, symbol_len=S, mel_lens=T] (reversed symbols, but it is the same anyway)
        mask_from_lens = monotonic_align.mask_from_lens

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs_2nd', 100)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    multigpu = config.get('multigpu', False)
    log_interval = config.get('log_interval', 10)
    saving_epoch = config.get('save_freq', 2)
    data_path = config['data_path']
    stats_path = config['stats_path']
    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)

    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=2,
                                      device=device,
                                      dataset_config={"data_path": data_path})
    # load pretrained ASR model
    text_aligner = load_ASR_models(config['ASR_path'],config['ASR_config'])

    with open('Vocoder/config.json') as f:
        data = f.read()
    h = AttrDict(json.loads(data))
    generator = Generator(h).to(device)
    state_dict_g = torch.load("Vocoder/g_00935000", map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,}

    distribution = {
        **load_and_move_to_cuda("EMA", stats_path),
        **load_and_move_to_cuda("pitch", stats_path),
        **load_and_move_to_cuda("energy", stats_path)
    }

    model = build_model(Munch(config['model_params']), text_aligner, stage = "second", distribution= distribution)

    _ = [model[key].to(device) for key in model]

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict= {key: scheduler_params.copy() for key in model})

    # multi-GPU support
    if multigpu:
        for key in model:
            model[key] = MyDataParallel(model[key])

    if config.get('pretrained_model', '') != '':
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
    else:
        start_epoch = 0
        iters = 0

        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, optimizer, start_epoch, iters = load_checkpoint(model, optimizer, first_stage_path,
                                        load_only_params=True)
        else:
            raise ValueError('You need to specify the path to the first stage model.')

    best_loss = float('inf')  # best test loss

    loss_params = Munch(config['loss_params'])
    for epoch in range(start_epoch, epochs):

        # If the data set is too large, you can use the subset of train_list
        train_dataloader = build_dataloader(train_list,
                                    batch_size=batch_size,
                                    num_workers=6,
                                    dataset_config={"data_path": data_path},
                                    device=device)
        running_loss = 0
        criterion = nn.L1Loss()

        _ = [model[key].train() for key in model]
        model.ArtsSpeech.text_encoder.eval()
        model.text_aligner.eval()
        inner_bar = tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch), position=1)
        for i, batch in enumerate(train_dataloader):
            batch = [b.to(device) for b in batch]
            texts, input_lengths, mels, mel_input_length, _, _, _ = batch
            mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
            text_mask = length_to_mask(input_lengths).to(texts.device)

            """duration_gt"""
            with torch.no_grad():

                _, s2s_pred, s2s_attn_feat = model.text_aligner(mels, mask, texts)
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                text_mask = length_to_mask(input_lengths).to(texts.device)
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)
                s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

            [mel_rec, mel_gt], F0, EMA, Energy, d_fake = model.ArtsSpeech(batch, _, s2s_attn_mono, step="second", epoch = epoch)

            mel_masks = length_to_mask(mel_input_length).to(device)
            EMA_masks =  mel_masks.unsqueeze(1).expand_as(EMA[0])
            loss_F0_rec =  F.smooth_l1_loss(F0[0].masked_fill(mel_masks, 0), F0[1].masked_fill(mel_masks, 0))
            loss_norm_rec = F.smooth_l1_loss(Energy[0].masked_fill(mel_masks, 0), Energy[1].masked_fill(mel_masks, 0))
            loss_EMA_rec = F.smooth_l1_loss(EMA[0].masked_fill(EMA_masks, 0), EMA[1].masked_fill(EMA_masks, 0))

            # discriminator loss
            optimizer.zero_grad()
            mel_gt.requires_grad_()
            out, _ = model.discriminator(mel_gt.unsqueeze(1))
            loss_real = adv_loss(out, 1)
            loss_reg = r1_reg(out, mel_gt)
            out, _ = model.discriminator(mel_rec.detach().unsqueeze(1))
            loss_fake = adv_loss(out, 0)
            d_loss = loss_real + loss_fake + loss_reg
            d_loss.backward()
            optimizer.step('discriminator')

            # generator loss
            optimizer.zero_grad()
            loss_mel = criterion(mel_rec, mel_gt)

            with torch.no_grad():
                _, f_real = model.discriminator(mel_gt.unsqueeze(1))
            out_rec, f_fake = model.discriminator(mel_rec.unsqueeze(1))
            loss_adv = adv_loss(out_rec, 1)

            # feature matching loss
            loss_fm = 0
            for m in range(len(f_real)):
                for k in range(len(f_real[m])):
                    loss_fm += torch.mean(torch.abs(f_real[m][k] - f_fake[m][k]))

            #duration loss
            loss_dur = 0
            for s2s_pred, text_input, text_length in zip(d_fake, d_gt, input_lengths):
                loss_dur += F.l1_loss(s2s_pred[1:text_length-1], text_input[1:text_length-1])
            loss_dur /= texts.size(0)

            g_loss = loss_params.lambda_mel * loss_mel + \
                    loss_params.lambda_dur * loss_dur + \
                    loss_params.lambda_params* (loss_F0_rec + loss_norm_rec + loss_EMA_rec)+ \
                    loss_params.lambda_adv * loss_adv + \
                    loss_params.lambda_fm * loss_fm

            running_loss += loss_mel.item()
            g_loss.backward()
            optimizer.step('ArtsSpeech')
            optimizer.scheduler()
            iters = iters + 1
            if (i+1)%log_interval == 0:
                logger.info('Epoch[%d/%d], Step[%d/%d], Loss:%.5f, Avd Loss:%.5f,  Disc Loss:%.5f, Dur Loss:%.5f, Norm Loss:%.5f, F0 Loss:%.5f, EMA loss:%.5f'
                        %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_adv, d_loss, loss_dur, loss_norm_rec, loss_F0_rec, loss_EMA_rec))

                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/adv_loss', loss_adv.item(), iters)
                writer.add_scalar('train/d_loss', d_loss.item(), iters)
                writer.add_scalar('train/dur_loss', loss_dur, iters)
                writer.add_scalar('train/norm_loss', loss_norm_rec, iters)
                writer.add_scalar('train/F0_loss', loss_F0_rec, iters)
                writer.add_scalar('train/EMA_loss', loss_EMA_rec, iters)

                running_loss = 0
                inner_bar.update(log_interval)


        loss_test = 0
        loss_align = 0
        _ = [model[key].eval() for key in model]
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                batch = [b.to(device) for b in batch]
                texts, input_lengths, mels, mel_input_length = batch

                #duration gt
                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)
                _, _, s2s_attn_feat = model.text_aligner(mels, mask, texts)
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                text_mask = length_to_mask(input_lengths).to(texts.device)
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)
                s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                d_gt = s2s_attn_mono.sum(axis=-1).detach()

                [mel_rec, mel_gt], _, _, _, d_fake = model.ArtsSpeech(batch, _, s2s_attn_mono, step="second", mode="val", epoch = epoch)

                loss_dur = 0
                for s2s_pred, text_input, text_length in zip(d_fake, d_gt, input_lengths):
                    loss_dur += F.l1_loss(s2s_pred[1:text_length-1], text_input[1:text_length-1])
                loss_dur /= texts.size(0)
                mel_rec = mel_rec[..., :mel_gt.shape[-1]]
                loss_mel = criterion(mel_rec, mel_gt)

                loss_test += loss_mel
                loss_align += loss_dur
                iters_test += 1

        with torch.no_grad():
            for batch in val_dataloader:
                num = random.randint(0, batch_size-1)
                mels = batch[2][num,...]
                mel_input_length = torch.LongTensor([[batch[3][num,...]]]).to(device)
                break

            device = torch.device("cuda")
            mels = mels.to(device).unsqueeze(0)
            text = [0, 52, 156, 86, 62, 16, 81, 86, 123, 16, 50, 70, 64, 44, 102, 56, 16, 72, 56, 46, 16, 61, 62, 156, 102, 54, 16, 69, 158, 123, 16, 46, 147, 157, 51, 158, 57, 135, 55, 102, 62, 123, 156, 102, 131, 83, 56, 68, 16, 72, 56, 46, 16, 48, 102, 54, 156, 69, 158, 61, 83, 48, 85, 68, 3, 16, 72, 56, 46, 16, 156, 51, 158, 64, 83, 56, 16, 61, 157, 138, 55, 16, 138, 64, 81, 83, 16, 55, 156, 57, 135, 61, 62, 16, 46, 102, 61, 62, 156, 102, 112, 92, 65, 102, 131, 62, 3, 16, 50, 157, 63, 158, 16, 46, 156, 43, 135, 62, 16, 65, 156, 86, 81, 85, 16, 81, 83, 16, 50, 156, 57, 135, 54, 16, 52, 156, 63, 158, 56, 102, 64, 157, 87, 158, 61, 3, 16, 76, 158, 123, 16, 62, 83, 16, 61, 58, 156, 51, 158, 53, 16, 55, 156, 57, 158, 123, 16, 65, 156, 43, 102, 46, 54, 51, 16, 81, 83, 16, 50, 156, 57, 135, 54, 16, 138, 64, 16, 44, 156, 51, 158, 102, 112, 3, 16, 65, 138, 68, 16, 156, 57, 135, 56, 54, 51, 16, 53, 123, 51, 158, 156, 47, 102, 125, 177, 46, 16, 102, 56, 16, 52, 156, 63, 158, 53, 54, 102, 46, 68, 16, 46, 147, 51, 156, 69, 158, 55, 83, 62, 123, 51, 1, 16, 81, 47, 102, 16, 156, 51, 158, 64, 83, 56, 16, 46, 156, 86, 123, 16, 62, 83, 16, 46, 123, 156, 51, 158, 55, 16, 81, 72, 62, 16, 62, 156, 63, 158, 16, 58, 156, 72, 123, 83, 54, 157, 86, 54, 16, 54, 156, 43, 102, 56, 68, 3, 16, 65, 157, 102, 62, 131, 16, 70, 53, 156, 57, 158, 123, 46, 102, 112, 16, 62, 83, 16, 52, 156, 63, 158, 53, 54, 102, 46, 16, 53, 72, 56, 16, 56, 156, 86, 64, 85, 16, 55, 156, 51, 158, 62, 16, 157, 76, 56, 16, 156, 87, 158, 119, 3, 16, 55, 156, 47, 102, 16, 55, 156, 51, 158, 62, 16, 61, 156, 138, 55, 65, 86, 123, 16, 102, 56, 16, 102, 56, 48, 156, 102, 56, 177, 125, 51, 4, 0]
            text = torch.LongTensor(text).to(device).unsqueeze(0)
            input_lengths = torch.LongTensor([text.shape[-1]]).to(device)

            mel_rec = model.ArtsSpeech([text,input_lengths,mels,mel_input_length], _, _, step="test")
            audio = generator(mel_rec.squeeze().unsqueeze(0))
            audio_gt = generator(mels.squeeze().unsqueeze(0))
            audio = audio.squeeze().cpu().numpy()
            audio_gt = audio_gt.squeeze().cpu().numpy()

        print('Epochs:', epoch + 1)
        logger.info('Validation loss: %.3f, %.3f' % (loss_test / iters_test, loss_align / iters_test))
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)
        writer.add_scalar('eval/dur_loss', loss_align / iters_test, epoch + 1)
        writer.add_audio( "eval/audio", audio, global_step=epoch+1, sample_rate=24000)
        writer.add_audio( "eval/audio_gt", audio_gt, global_step=epoch+1, sample_rate=24000)
        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        writer.add_figure('eval/attn', attn_image, epoch)
        mel_image = get_image(mel_rec[0].cpu().numpy().squeeze())
        writer.add_figure('eval/mel_rec', mel_image, epoch)

        if (epoch+1) % saving_epoch == 0:
            if (loss_test / iters_test) < best_loss:
                best_loss = loss_test / iters_test
            print('Saving..')
            state = {
                'net':  {key: model[key].state_dict() for key in model},
                'optimizer': optimizer.state_dict(),
                'iters': iters,
                'val_loss': loss_test / iters_test,
                'epoch': epoch,
            }
            save_path = osp.join(log_dir, 'epoch_2nd_%05d.pth' % epoch)
            torch.save(state, save_path)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    setup_seed(3407)
    main()
