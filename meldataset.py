#coding: utf-8
import os
import torch
import random
import torchaudio
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i
class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            indexes.append(self.word_index_dictionary[char])
        return indexes


np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave_tensor):
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dataset_config,
                 validation = False,
                 sr=24000,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.data_path = dataset_config["data_path"]
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, text_tensor, speaker_id, EMA, F0, energy = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        F0 = F0[:(length_feature - length_feature % 2)]
        energy = energy[:(length_feature - length_feature % 2)]
        EMA = EMA[:(length_feature - length_feature % 2),:]
        return speaker_id, acoustic_feature, text_tensor, path, EMA, F0, energy

    def getitem(self, idx):
        data = self.data_list[idx]
        path = data[0]
        wave, _, speaker_id, _, _, _ = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()
        return speaker_id, mel_tensor

    def _load_tensor(self, data):

        wave_path, text, speaker_id = data
        Wav_path = os.path.normpath(os.path.join(self.data_path, wave_path)).replace("\\", "/")
        Path = wave_path.replace("\\", "").replace("/", "").split(".")[0]+".npy"
        EMA_path = self.data_path + "/predict/EMA/" + Path
        F0_path = self.data_path + "/predict/F0/" + Path
        N_path = self.data_path + "/predict/energy/" + Path
        speaker_id = int(speaker_id)
        wave, _ = torchaudio.load(Wav_path)

        EMA = torch.from_numpy(np.load(EMA_path))
        F0 = torch.from_numpy(np.load(F0_path))
        energy = torch.from_numpy(np.load(N_path))

        if wave.shape[0] == 2:
            wave = wave[0,:].squeeze()
        else:
            wave = wave.squeeze()

        text = self.text_cleaner(text)
        text.insert(0, 0)
        text.append(0)
        text = torch.LongTensor(text)
        return wave.float(), text, speaker_id, EMA, F0, energy

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """
    def __init__(self):
        pass
    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()

        EMAs = torch.zeros((batch_size, 10, max_mel_length)).float()
        F0s = torch.zeros((batch_size, max_mel_length)).float()
        energys = torch.zeros((batch_size, max_mel_length)).float()

        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        paths = ['' for _ in range(batch_size)]

        for bid, (label, mel, text, path, EMA, F0, energy) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)

            mels[bid, :, :mel_size] = mel
            EMAs[bid, :, :mel_size] = EMA.transpose(0,1)
            F0s[bid, :mel_size] = F0
            energys[bid, :mel_size] = energy
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            paths[bid] = path

        return texts, input_lengths, mels, output_lengths, EMAs, F0s, energys

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     Parallel = False):

    dataset = FilePathDataset(path_list, validation=validation,  dataset_config=dataset_config)
    collate_fn = Collater(**collate_config)
    if Parallel == True :
        train_sampler = DistributedSampler(dataset)
        train_loader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size,
                                  shuffle=(not validation),
                                  num_workers=num_workers,
                                  drop_last=(not validation),
                                  collate_fn=collate_fn,
                                  pin_memory="cuda")
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=(not validation),
                                 collate_fn=collate_fn,
                                 pin_memory="cuda")

    return data_loader
