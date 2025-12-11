# EfficientAT

This directory contains code from the EfficientAT project by Florian Schmid.

**Original Repository:** https://github.com/fschmid56/EfficientAT

**License:** MIT

## Modifications

The following modifications were made for integration with WatchHAR:

1. Changed import paths to work within the `third_party` package structure
2. Included only the MobileNetV3 (`mn`) model files needed for mn05
3. Added `feature_extraction` mode for extracting audio features

## Citation

If you use this code, please cite the original EfficientAT paper:

```bibtex
@inproceedings{schmid2023efficientat,
  title={Efficient Large-Scale Audio Tagging Via Transformers},
  author={Schmid, Florian and Koutini, Khaled and Widmer, Gerhard},
  booktitle={ICASSP},
  year={2023}
}
```
