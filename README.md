[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repo contains the official implementation of the **ACM MM 2024**:

<div align="center">
<h1>
<b>
ArtSpeech: Adaptive Text-to-Speech Synthesis with Articulatory Representations
</b>
</h1>
<h4>
<b>
Zhongxu Wang, Yujia Wang, Mingzhu Li, Hua Huang
</b>
</h4>
</div>

## Introduction

We devise an articulatory representation-based text-to-speech (TTS) model, ArtSpeech, an explainable and effective network for humanlike speech synthesis, by revisiting the sound production system. Current deep TTS models learn acoustic-text mapping in a fully parametric manner, ignoring the explicit physical significance of articulation movement. ArtSpeech, on the contrary, leverages articulatory representations to perform adaptive TTS, clearly describing the voice tone and speaking prosody of different speakers. Specifically, energy, F0, and vocal tract variables are utilized to represent airflow forced by articulatory organs, the degree of tension in the vocal folds of the larynx, and the coordinated movements between different organs, respectively. We also design a multidimensional style mapping network to extract speaking styles from the articulatory representations, guided by which variation predictors could predict the final mel spectrogram output. To validate the effectiveness of our approach, we conducted comprehensive experiments and analyses using the widely recognized speech corpus, such as LJSpeech and LibriTTS datasets, yielding promising similarity enhancement between the generated results and the target speaker’s voice and prosody.

Demo Page: <a href="https://zhongxu-wang.github.io/artspeeech.demopage/" target="_blank">ArtSpeech demopage</a>

Paper Link: <a href="https://dl.acm.org/doi/10.1145/3664647.3681097" target="_blank">paper</a>

## Pre-requisites

1. Python >= 3.8  
```bash
conda create -n ArtSpeech python==3.8.0
conda activate ArtSpeech
```
(Some code adjustments might be necessary when using the latest versions of Python and PyTorch.)

2. Clone this repository:
```bash
git clone https://github.com/Zhongxu-Wang/ArtSpeech.git
cd ArtSpeech
```

3. Install python requirements.
```bash
pip install torchaudio munch torch librosa pyyaml click tqdm attrdict matplotlib tensorboard Cython
``` 

## Inference

Download the pre-training weight and put it in the `Outputs/LibriTTS/`: <a href="https://drive.google.com/file/d/1_c07vqqd_102e2y73v5jTGJbptcukRCh/view?usp=sharing" target="_blank">LibriTTS pre-training weight</a>

Set `pretrained_model` in `Configs\config.yaml` to `"Outputs/LibriTTS/epoch_2nd_00119.pth"`

Before running the inference, make sure you have a reference audio file and the text you want to synthesize. 
```python
text = "XXXXX"
ref_wav = "ref.wav"
save_path = "output.wav"
```
Execute the test script
```bash
python test.py
```

## Training
### Dataset Preparation

Create a new folder: `Data/LibriTTS/train-clean-460/`
Download the `train-clean-100.tar.gz` and `train-clean-360.tar.gz` datasets from <a href="https://www.openslr.org/60/" target="_blank">LibriTTS</a>, merge them, and place them in the `train-clean-460` directory.

Download the articulatory features gt <a href="https://drive.google.com/file/d/1aVO9EKqgL7CQ7bYbNmBn-xs7EgZQD0cP/view?usp=drive_link" target="_blank"> predict </a> file and place it in the `Data/` directory. The predict file contains the articulatory features GT for the LibriTTS, LJSpeech, and VCTK datasets. If you want to train on your own dataset, please refer to this project: <a href="https://github.com/Zhongxu-Wang/TVsExtractor" target="_blank">TVsExtractor</a>.

### Train the Model

Run the following commands:
```bash
python train_first.py
python train_second.py
```
## Additional Training Data

All results in the paper, as well as the pre-trained models provided in the repository, were trained using the LJSpeech and LibriTTS datasets.

Here we publish the multi-age, emotionally rich speech data we collected. I hope this will be useful for future research.

### Multi-age dataset
<a href="https://drive.google.com/drive/folders/1XlCWqwI1tCL8Xhd-tpjeRAv4Pm7WnzJL?usp=drive_link" target="_blank">Multi-age dataset</a>
The Multi-Age Dataset consists of 36 children's videos and 15 elderly videos sourced from YouTube.  The corresponding audio has been transcribed through Automatic Speech Recognition (ASR), segmented, and manually verified.  The dataset contains a total of 4,695 spoken sentences.
