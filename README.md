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

Paper Link: <a href="https://openreview.net/forum?id=nagiMHx4A3" target="_blank">OpenReview</a>

## Pre-requisites

1. Python >= 3.8
```bash
conda create -n ArtSpeech python==3.8.0
conda activate ArtSpeech
```

2. Clone this repository:
```bash
git clone https://github.com/Zhongxu-Wang/ArtSpeech.git
cd ArtSpeech
```

3. Install python requirements.
```bash
pip install torchaudio munch torch librosa pyyaml click tqdm attrdict matplotlib tensorboard Cython
``` 

4. Build Monotonic Alignment Search.
```bash
git clone https://github.com/resemble-ai/monotonic_align.git
cd monotonic_align
python setup.py install
```

to be continued ......

## Inference


## Training

## Additional Training Data

All results in the paper as well as the pre-trained models provided in the repository were trained using the LJSpeech and LibriTTS datasets.
However, due to the lack of elderly, children, and emotionally rich data in these two datasets, the trained TTS models performed poorly on some data, so we expanded the datasets.
Here we publish the multi-age, emotionally richer speech data we collected.

