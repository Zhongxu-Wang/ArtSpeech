
import yaml
import json
import yaml
import torch
import random
import librosa
import phonemizer
import torchaudio
import numpy as np
import soundfile as sf
from munch import Munch
from attrdict import AttrDict

from utils import *
from Vocoder.vocoder import Generator
from models import load_ASR_models, build_model, load_checkpoint

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(char)
        return indexes

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def load_and_move_to_cuda(data_key, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    _, _, mean_val, std_val = data[data_key]
    return {
        f"{data_key}_mean": torch.tensor(mean_val).to("cuda"),
        f"{data_key}_std": torch.tensor(std_val).to("cuda")
    }

class ArtSpeech:
    def __init__(self, config_path = "Configs/config.yaml"):
        config = yaml.safe_load(open(config_path))
        device = config.get('device', 'cpu')
        stats_path = config.get('stats_path')
        # load pretrained ASR model
        text_aligner = load_ASR_models(config['ASR_path'],config['ASR_config'])

        with open('Vocoder/config.json') as f:
            data = f.read()
        h = AttrDict(json.loads(data))
        self.generator = Generator(h).to(device)
        state_dict_g = torch.load("Vocoder/g_00935000", map_location=device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()

        distribution = {
            **load_and_move_to_cuda("EMA", stats_path),
            **load_and_move_to_cuda("pitch", stats_path),
            **load_and_move_to_cuda("energy", stats_path)
        }

        model = build_model(Munch(config['model_params']), text_aligner, stage = "second", distribution = distribution)

        _ = [model[key].to(device) for key in model]

        self.model, _, _, _ = load_checkpoint(model,  _, config['pretrained_model'], load_only_params=True)

        self.textclenaer = TextCleaner()
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

    def synthesis(self, text, ref_wav, save_path):    
        with torch.no_grad():
            device = torch.device("cuda")

            ps = self.global_phonemizer.phonemize([text])
            print(ps)
            tokens = self.textclenaer(ps[0])
            text = torch.LongTensor(tokens).to(device).unsqueeze(0)
            input_lengths = torch.LongTensor([text.shape[-1]]).to(device)

            wave, sr = librosa.load(ref_wav, sr=24000)
            audio, _ = librosa.effects.trim(wave, top_db=30)
            if audio.shape[0] == 2:
                audio = audio[0,:].squeeze()
            else:
                audio = audio.squeeze()
            if sr != 24000:
                audio = librosa.resample(audio, sr, 24000)
            mels = preprocess(audio).to(device)

            input_lengths = torch.LongTensor([text.shape[-1]]).to(device)
            mel_input_length = torch.LongTensor([mels.shape[-1]]).to(device)

            mel_rec = self.model.ArtsSpeech([text,input_lengths,mels,mel_input_length, _ ,_ ,_], _, _, step="test")
            
            audio = self.generator(mel_rec.squeeze().unsqueeze(0))
            audio_gt = self.generator(mels.squeeze().unsqueeze(0))
            audio = audio.squeeze().cpu().numpy()
            audio_gt = audio_gt.squeeze().cpu().numpy()
            sf.write(save_path, audio, 24000)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__=="__main__":
    setup_seed(3407)
    TTSModel = ArtSpeech()

    text = "The condition is that I will be permitted to make Luther talk American, 'streamline' him, so to speak-because you will never get people, whether in or outside the Lutheran Church, actually to read Luther unless we make him talk as he would talk today to Americans."
    ref_wav = "ref.wav"
    save_path = "output.wav"
    TTSModel.synthesis(text, ref_wav, save_path)
