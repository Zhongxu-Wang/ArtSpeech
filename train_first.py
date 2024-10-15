import os
import yaml
import json
import torch
import click
import random
import shutil
import numpy as np
from torch import nn
import os.path as osp
from tqdm import tqdm
from munch import Munch
from attrdict import AttrDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Vocoder.vocoder import Generator
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from optimizers import build_optimizer
from meldataset import build_dataloader
from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from models import load_ASR_models, build_model, load_checkpoint
from utils import get_data_path_list, length_to_mask, adv_loss, r1_reg, get_image, load_and_move_to_cuda

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

    train_step = "first"
    config = yaml.safe_load(open(config_path))
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")
    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    # load config
    stats_path = config['stats_path']
    batch_size = config['batch_size']
    device = config['device']
    epochs = config['epochs_1st']
    log_interval = config['log_interval']
    data_path = config['data_path']
    # load data
    train_list, val_list = get_data_path_list(train_path = config['train_data'], val_path = config['val_data'])
    train_dataloader = build_dataloader(train_list, batch_size=batch_size, num_workers=8,
                                        dataset_config={"data_path": data_path}, device=device)
    val_dataloader = build_dataloader(val_list, batch_size=batch_size, validation=True,
                                      num_workers=2,  device=device, dataset_config={"data_path": data_path})
    
    distribution = {
        **load_and_move_to_cuda("EMA", stats_path),
        **load_and_move_to_cuda("pitch", stats_path),
        **load_and_move_to_cuda("energy", stats_path)
    }

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
        "max_lr": float(config['optimizer_params'].get('lr')),
        "pct_start": float(config['optimizer_params'].get('pct_start')),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader)}

    model = build_model(Munch(config['model_params']), text_aligner, distribution = distribution)
    _ = [model[key].to(device) for key in model]

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict= {key: scheduler_params.copy() for key in model})
    op_align = optimizer.get("text_aligner")
    op_Arts = optimizer.get("ArtsSpeech")
    op_discr = optimizer.get("discriminator")

    # multi-GPU support
    if config.get('multigpu'):
        for key in model:
            model[key] = MyDataParallel(model[key])

    if config.get('pretrained_model') != '':
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        print("loading")
        start_epoch = 21
    else:
        start_epoch = 0
        iters = 0

    best_loss = float('inf')  # best test loss

    loss_params = Munch(config['loss_params'])
    TMA_epoch = loss_params.TMA_epoch
    TMA_CEloss = loss_params.TMA_CEloss

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, epochs):
        
        running_loss = 0
        criterion = nn.L1Loss()
        _ = [model[key].train() for key in model]
        inner_bar = tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch), position=1)
        for i, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            batch = [b.to(device) for b in batch]
            texts, input_lengths, mels, mel_input_length, ema_gt, f0_gt, n_gt = batch

            n_gt = (n_gt - distribution["energy_mean"])/distribution["energy_std"]
            f0_gt = (f0_gt-distribution["pitch_mean"])/distribution["pitch_std"]
            ema_gt = ((ema_gt.transpose(1, 2)-distribution["EMA_mean"])/distribution["EMA_std"]).transpose(1, 2)
            with autocast():
                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)
                """Text Aligner"""
                _, s2s_pred, s2s_attn_feat = model.text_aligner(mels, mask, texts)
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                with torch.no_grad():
                    attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                    attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                    attn_mask = (attn_mask < 1)
                s2s_attn_feat.masked_fill_(attn_mask, -float("inf"))
                if TMA_CEloss:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=1) # along the mel dimension
                else:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension
                # get monotonic version
                with torch.no_grad():
                    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                s2s_attn = torch.nan_to_num(s2s_attn)
                Mel_ext, Mel_gt, f0_ext, ema_ext = model.ArtsSpeech(batch, s2s_attn, s2s_attn_mono, train_step, epoch = epoch)


                mel_masks = length_to_mask(mel_input_length).to(device)
                EMA_masks =  mel_masks.unsqueeze(1).expand_as(ema_ext)
                ema_ext = ema_ext.masked_fill(EMA_masks, 0)
                f0_ext = f0_ext.squeeze().masked_fill(mel_masks, 0)
                loss_f0 = F.smooth_l1_loss(f0_gt, f0_ext)
                loss_EMA = F.smooth_l1_loss(ema_gt, ema_ext)

                # discriminator loss
                Mel_gt.requires_grad_()
                out, _ = model.discriminator(Mel_gt.unsqueeze(1))
                loss_real = adv_loss(out, 1)
                loss_reg = r1_reg(out, Mel_gt)
                out, _ = model.discriminator(Mel_ext.detach().unsqueeze(1))
                loss_fake = adv_loss(out, 0)
                d_loss = loss_real + loss_fake + loss_reg * loss_params.lambda_reg
            optimizer.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(op_discr)
            scaler.step(op_discr)
            with autocast():
                # generator loss
                loss_mel = criterion(Mel_ext, Mel_gt)
                # adversarial loss
                with torch.no_grad():
                    _, f_real = model.discriminator(Mel_gt.unsqueeze(1))
                out_rec, f_fake = model.discriminator(Mel_ext.unsqueeze(1))
                loss_adv = adv_loss(out_rec, 1)

                # feature matching loss
                loss_fm = 0
                for m in range(len(f_real)):
                    for k in range(len(f_real[m])):
                        loss_fm += torch.mean(torch.abs(f_real[m][k] - f_fake[m][k]))

                if epoch >= TMA_epoch:
                    loss_s2s = 0
                    for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                        loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
                    loss_s2s /= texts.size(0)
                    if TMA_CEloss:
                        # cross entropy loss for monotonic alignment
                        log_attn = torch.nan_to_num(F.log_softmax(s2s_attn_feat, dim=1)) # along the mel dimension
                        loss_mono = -(torch.mul(log_attn, s2s_attn_mono).sum(axis=[-1, -2]) / input_lengths).mean()
                    else:
                        # L1 loss for monotonic alignment
                        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10
                else:
                    loss_s2s = 0
                    loss_mono = 0

                g_loss = loss_params.lambda_mel * loss_mel + \
                    loss_params.lambda_adv * loss_adv + \
                    loss_params.lambda_fm * loss_fm + \
                    loss_params.lambda_mono * loss_mono + \
                    loss_params.lambda_s2s * loss_s2s+\
                    loss_params.lambda_params*(loss_f0+loss_EMA)

                running_loss += loss_mel.item()
            optimizer.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.unscale_(op_Arts)
            scaler.step(op_Arts)

            if epoch >= TMA_epoch:
                scaler.unscale_(op_align)
                scaler.step(op_align)
            optimizer.scheduler()
            scaler.update()
            iters = iters + 1
            if (i+1)%log_interval == 0:
                logger.info ('Epoch [%d/%d], Step [%d/%d], Mel Loss: %.5f, Adv Loss: %.5f, Disc Loss: %.5f, Mono Loss: %.5f, S2S Loss: %.5f, Params Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_list)//batch_size, running_loss / log_interval, loss_adv.item(), d_loss.item(), loss_mono, loss_s2s, loss_params.lambda_params*(loss_EMA+loss_f0)))

                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/adv_loss', loss_adv.item(), iters)
                writer.add_scalar('train/d_loss', d_loss.item(), iters)
                writer.add_scalar('train/mono_loss', loss_mono, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/Params_Loss', loss_EMA+loss_f0, iters)
                running_loss = 0
                inner_bar.update(log_interval)

        ema_ext = ema_ext.detach().cpu().numpy()
        ema_gt = ema_gt.detach().cpu().numpy()
        fig, axs = plt.subplots(10, 1, facecolor='white')
        for num, ax in zip(range(10), axs):
            ax.plot(ema_gt[0,num,:], 'b-')
            ax.plot(ema_ext[0,num,:], 'r-')

        loss_test = 0
        _ = [model[key].eval() for key in model]
        with torch.no_grad():
            iters_test = 0
            for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                batch = [b.to(device) for b in batch]
                texts, input_lengths, mels, mel_input_length, _, _, _ = batch

                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

                """Text Aligner"""
                ppgs, s2s_pred, s2s_attn_feat = model.text_aligner(mels, mask, texts)
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                s2s_attn_feat = s2s_attn_feat[..., 1:]
                s2s_attn_feat = s2s_attn_feat.transpose(-1, -2)
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)
                s2s_attn_feat.masked_fill_(attn_mask, -float("inf"))
                if TMA_CEloss:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=1) # along the mel dimension
                else:
                    s2s_attn = F.softmax(s2s_attn_feat, dim=-1) # along the text dimension
                # get monotonic version
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
                s2s_attn = torch.nan_to_num(s2s_attn)
                Mel_ext, Mel_gt, f0_ext, ema_ext = model.ArtsSpeech(batch, s2s_attn, s2s_attn_mono, train_step, mode = "val", epoch = epoch)
                
                Mel_ext = Mel_ext[..., :Mel_gt.shape[-1]]
                loss_mel = criterion(Mel_ext, Mel_gt)
                loss_test += loss_mel
                iters_test += 1
        logger.info('Validation mel loss: %.3f' % (loss_test / iters_test))
        writer.add_scalar('eval/mel_loss', loss_test / iters_test, epoch + 1)

        attn_image = get_image(s2s_attn[0].cpu().numpy().squeeze())
        writer.add_figure('eval/attn', attn_image, epoch)
        mel_image = get_image(Mel_ext[0].cpu().numpy().squeeze())
        writer.add_figure('eval/mel_rec', mel_image, epoch)

        writer.add_figure('train/EMA', fig, epoch)

        
        if epoch % config.get('save_freq') == 0:
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
            save_path = osp.join(log_dir, 'epoch_1st_%05d.pth' % epoch)
            torch.save(state, save_path)

    print('Saving..')
    state = {
        'net':  {key: model[key].state_dict() for key in model},
        'optimizer': optimizer.state_dict(),
        'iters': iters,
        'val_loss': loss_test / iters_test,
        'epoch': epoch,
    }
    save_path = osp.join(log_dir, "first_stage.pth")
    torch.save(state, save_path)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def occumpy_mem():
    x = torch.zeros(15000000000, dtype=torch.float32, device='cuda')
    del x
    
if __name__=="__main__":
    setup_seed(3407)
    #occumpy_mem()
    main()
