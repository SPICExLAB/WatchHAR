# WatchHAR

Official Repository for WatchHAR: Real-time On-device Human Activity Recognition System for Smartwatches (ICMI'25)

<p align="center">
  <img src="teaser.gif" alt="WatchHAR Demo" width="600">
</p>

## Overview

WatchHAR is a multimodal human activity recognition system that fuses audio and IMU sensor data for real-time activity recognition on smartwatches. This repository contains the training pipeline using Leave-One-Participant-Out (LOPO) cross-validation on the SAMoSA dataset.

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environments.yml
conda activate watchhar
```

### 2. Data Preparation

Download the SAMoSA dataset and place it in `./Dataset/` directory.

**Expected data structure**:
```
Dataset/
├── 7---Kitchen---Chopping---1.pkl
├── 7---Kitchen---Chopping---2.pkl
├── 17---Bathroom---Brushing_hair---1.pkl
├── 103---Workshop---Drilling---1.pkl
└── ... (additional files)
```

### 3. Preprocess Data

```bash
python preprocess.py ./Dataset ./preprocessed_data
```

This creates preprocessed files with:
- Mel spectrograms (64 mel bins, using nnAudio)
- Aligned audio-IMU windows (1 second, 50Hz IMU, 1kHz Audio)
- Same file structure as raw data in `preprocessed_data/`

Settings are loaded from `config/default.yaml`.

### 4. Train Models (LOPO)

For Leave-One-Participant-Out cross-validation, train one model per held-out participant:

```bash
# Example: Train with participant 103 as test set
python train.py \
  --config config/default.yaml \
  --data-path ./preprocessed_data \
  --target-participants 103 \
  --experiment-name "audioimu_lopo"
```

This trains on all participants except the target (103), saves the best model to `trained_models/audioimu_lopo/best_model_excluded_103.pt`.

Repeat for all 20 participants to complete LOPO evaluation.

### 5. Evaluate Models

After training all 20 LOPO models:

```bash
python evaluate.py \
  --experiment-dir ./trained_models/audioimu_lopo \
  --data-path ./preprocessed_data
```

This generates:
- `evaluation_results.txt` - Per-participant metrics
- `classification_report.txt` - Detailed classification metrics
- `confusion_matrix.png` - Confusion matrix visualization

---

## Configuration

Main configuration file: `config/default.yaml`

**To modify settings**, edit `config/default.yaml` or pass command-line arguments to override specific values. current setting is used for publication - MobileNetV3 with width 0.5, CNN2D IMU model, final fusion layer.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yeon2025watchhar,
  author = {Yeon, Taeyoung and Xu, Vasco and Hoffmann, Henry and Ahuja, Karan},
  title = {WatchHAR: Real-time On-device Human Activity Recognition System for Smartwatches},
  doi = {10.1145/3716553.3750775},
  booktitle = {Proceedings of the 27th International Conference on Multimodal Interaction},
  year = {2025},
}
```

---

## Acknowledgments

This project uses code from:
- [EfficientAT](https://github.com/fschmid56/EfficientAT) by Florian Schmid (MIT License)
- [nnAudio](https://github.com/KinWaiCheuk/nnAudio) by Kin Wai Cheuk (MIT License)

We thank the authors for making their code available.

---

## License

This work is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

For **commercial use**, a separate commercial license is required. Please contact kahuja@northwestern.edu at Northwestern University for licensing inquiries.

See individual LICENSE files in `third_party/` for external dependencies.

---

## Contact

For other general questions, please contact: nu.spicelab@gmail.com