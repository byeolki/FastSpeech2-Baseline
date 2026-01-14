# FastSpeech2-Baseline

PyTorch implementation of FastSpeech2 for text-to-speech research.

## Description

This repository implements FastSpeech2, a non-autoregressive text-to-speech model with explicit duration, pitch, and energy predictors. This implementation serves as a baseline for comparing novel TTS architectures in research.

FastSpeech2 generates mel-spectrograms in parallel from input text, achieving fast inference speed while maintaining high speech quality. The model uses feed-forward Transformer blocks and requires an external vocoder (e.g., HiFi-GAN) to convert mel-spectrograms to waveforms.

## Citation
```bibtex
@inproceedings{ren2021fastspeech2,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
