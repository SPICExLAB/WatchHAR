# nnAudio

This directory contains code from the nnAudio project by Kin Wai Cheuk.

**Original Repository:** https://github.com/KinWaiCheuk/nnAudio

**License:** MIT

## Why nnAudio?

nnAudio uses PyTorch 1D convolutions to compute spectrograms on GPU. This produces
slightly different numerical results compared to numpy/librosa implementations.
We include this code to ensure reproducibility of our results.

## Included Components

- `mel.py` - MelSpectrogram and MFCC
- `stft.py` - STFT and iSTFT
- `utils.py` - Helper functions
- `librosa_functions.py` - Librosa-compatible functions

## Citation

If you use this code, please cite the original nnAudio paper:

```bibtex
@article{cheuk2020nnaudio,
  author={K. W. Cheuk and H. Anderson and K. Agres and D. Herremans},
  journal={IEEE Access},
  title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks},
  year={2020},
  volume={8},
  pages={161981-162003},
  doi={10.1109/ACCESS.2020.3019084}
}
```
