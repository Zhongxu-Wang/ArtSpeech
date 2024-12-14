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
cd ..
```

## Inference

Download the pre-training weight and put it in the "Outputs/LibriTTS/": <a href="https://drive.google.com/file/d/1_c07vqqd_102e2y73v5jTGJbptcukRCh/view?usp=sharing" target="_blank">LibriTTS pre-training weight</a>

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

to be continued

## Additional Training Data

All results in the paper, as well as the pre-trained models provided in the repository, were trained using the LJSpeech and LibriTTS datasets.

Here we publish the multi-age, emotionally rich speech data we collected. I hope this will be useful for future research.
