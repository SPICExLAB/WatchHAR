#!/usr/bin/env python3
"""Preprocess SAMoSA dataset for training.

This script preprocesses raw audio and IMU data from the SAMoSA dataset,
creating windows and computing mel spectrograms aligned with IMU data.

Settings:
- STFT window: 25ms
- STFT hop: 10ms
- Audio example window: 0.96 seconds
- IMU window: 1 second (default)
- IMU hop: 10 samples (0.2 seconds at 50Hz)
"""

import argparse
import pickle
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from typing import Dict, List, Tuple

from third_party.nnAudio import MelSpectrogram
from utils.constants import (
    DEFAULT_AUDIO_SR,
    DEFAULT_AUDIO_SR_TARGET,
    DEFAULT_AUDIO_SR_SPEECH,
    DEFAULT_IMU_SR,
    DEFAULT_IMU_WIN_SEC,
    DEFAULT_HOP_LENGTH,
    STFT_WINDOW_LENGTH_SECONDS,
    STFT_HOP_LENGTH_SECONDS,
    AUDIO_EXAMPLE_WINDOW_SECONDS,
    DATA_FILE_PATTERN
)


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Dictionary with configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found, using constants")
        return {}

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_samosa_file(file_path: Path) -> Dict:
    """Load a SAMoSA pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def power_to_db(melspec: torch.Tensor) -> torch.Tensor:
    """Convert power spectrogram to dB scale (like librosa.power_to_db)."""
    amin = 1e-10
    ref_value = 1.0
    log_spec = 10.0 * torch.log10(torch.clamp(melspec, min=amin))
    log_spec -= 10.0 * np.log10(max(amin, ref_value))
    return log_spec


def create_mel_spectrogram_transform(
    sample_rate: int,
    n_mels: int = 64,
    trainable: bool = False,
    stft_window_sec: float = STFT_WINDOW_LENGTH_SECONDS,
    stft_hop_sec: float = STFT_HOP_LENGTH_SECONDS
) -> MelSpectrogram:
    """Create mel spectrogram transform.

    Args:
        sample_rate: Audio sampling rate (e.g., 16000 or 1000)
        n_mels: Number of mel bands
        trainable: Whether mel filterbank is trainable
        stft_window_sec: STFT window length in seconds
        stft_hop_sec: STFT hop length in seconds

    Returns:
        MelSpectrogram transform
    """
    # Convert seconds to samples
    win_length = int(sample_rate * stft_window_sec)
    hop_length = int(sample_rate * stft_hop_sec)

    # Compute FFT size as next power of 2
    n_fft = 2 ** int(np.ceil(np.log(win_length) / np.log(2.0)))

    mel_transform = MelSpectrogram(
        sr=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        n_mels=n_mels,
        hop_length=hop_length,
        fmin=10,
        fmax=sample_rate // 2,
        trainable_mel=trainable,
        trainable_STFT=trainable,
        verbose=False
    )

    return mel_transform


def frame_imu_data(data: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    """Create windowed frames from IMU data.

    Args:
        data: IMU data array of shape (num_samples, num_sensors)
        window_length: Window length in samples
        hop_length: Hop length in samples

    Returns:
        Windowed data of shape (num_frames, window_length, num_sensors)
    """
    # Pad zeros if sequence too short
    if data.shape[0] < window_length:
        len_pad = int(np.ceil(window_length)) - data.shape[0]
        to_pad = np.zeros((len_pad,) + data.shape[1:])
        data = np.concatenate([data, to_pad], axis=0)

    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, int(window_length)) + data.shape[1:]
    strides = (data.strides[0] * int(hop_length),) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def compute_log_mel_spectrogram(
    audio_data: np.ndarray,
    mel_transform: MelSpectrogram,
    original_sr: int = 16000,
    target_sr: int = 16000
) -> np.ndarray:
    """Compute log mel spectrogram for entire audio signal.

    Args:
        audio_data: Raw audio data
        mel_transform: MelSpectrogram transform
        original_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Log mel spectrogram of shape (num_frames, n_mels)
    """
    # Downsample if needed
    if original_sr != target_sr:
        factor = original_sr // target_sr
        audio_data = audio_data[::factor]

    # Normalize audio
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Convert to tensor with batch dimension
    audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)

    # Compute mel spectrogram
    with torch.no_grad():
        mel_spec = mel_transform(audio_tensor)
        log_mel = power_to_db(mel_spec.squeeze(0))

    return log_mel.numpy()


def align_imu_audio_windows(
    imu_examples: np.ndarray,
    log_mel: np.ndarray,
    imu_sr: int,
    hop_len_imu: int,
    window_len_imu: int,
    audio_window_sec: float = AUDIO_EXAMPLE_WINDOW_SECONDS,
    stft_window_sec: float = STFT_WINDOW_LENGTH_SECONDS,
    stft_hop_sec: float = STFT_HOP_LENGTH_SECONDS
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Align IMU windows with corresponding audio mel spectrogram windows.

    For each IMU window, extract the most recent audio_window_sec of audio mel spectrogram
    that ends at the same time as the IMU window.

    Args:
        imu_examples: Windowed IMU data (num_frames, window_length, num_sensors)
        log_mel: Full log mel spectrogram (num_audio_frames, n_mels)
        imu_sr: IMU sampling rate
        hop_len_imu: IMU hop length in samples
        window_len_imu: IMU window length in samples
        audio_window_sec: Audio window length in seconds
        stft_window_sec: STFT window length in seconds
        stft_hop_sec: STFT hop length in seconds

    Returns:
        Tuple of (aligned_imu_windows, aligned_audio_windows)
    """
    windowed_data_audio = []
    windowed_data_imu = []

    # Calculate audio window size in frames
    audio_example_frames = int(audio_window_sec / stft_hop_sec)

    for i in range(imu_examples.shape[0]):
        # Calculate end time of IMU window in seconds
        end_sample_imu = i * hop_len_imu + window_len_imu
        end_time_imu = end_sample_imu / imu_sr

        # Find corresponding audio frame index (end of audio window)
        # Audio frame at time t corresponds to: t - stft_window_sec
        end_index = int((end_time_imu - stft_window_sec) / stft_hop_sec)
        start_index = end_index - audio_example_frames

        # Skip if we don't have enough audio history
        if start_index < 0:
            continue

        # Extract audio window
        audio_example = log_mel[start_index:end_index]

        # Pad if needed to ensure consistent shape
        if audio_example.shape[0] < audio_example_frames:
            to_pad = audio_example_frames - audio_example.shape[0]
            zero_pad = np.zeros((to_pad,) + audio_example.shape[1:])
            audio_example = np.concatenate([audio_example, zero_pad], axis=0)

        # Keep shape as (time, mels) to match reference audioIMU format
        # audio_example shape: (time_steps, n_mels)

        windowed_data_audio.append(audio_example)
        windowed_data_imu.append(imu_examples[i])

    return windowed_data_imu, windowed_data_audio


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    audio_sr: int = DEFAULT_AUDIO_SR_TARGET,
    compute_mel: bool = True,
    n_mels: int = 64,
    imu_window_sec: float = DEFAULT_IMU_WIN_SEC,
    hop_length_imu: int = DEFAULT_HOP_LENGTH,
    stft_window_sec: float = STFT_WINDOW_LENGTH_SECONDS,
    stft_hop_sec: float = STFT_HOP_LENGTH_SECONDS
):
    """Preprocess entire dataset using ubicoustics-style alignment.

    Args:
        input_dir: Path to raw SAMoSA dataset
        output_dir: Path to output preprocessed data
        audio_sr: Target audio sampling rate (16000 or 1000)
        compute_mel: Whether to compute mel spectrograms
        n_mels: Number of mel bands
        imu_window_sec: IMU window size in seconds (default: 1)
        hop_length_imu: IMU hop length in samples (default: 10)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Audio window set to 0.96s for 96 frames (not matching IMU window)
    audio_window_sec = AUDIO_EXAMPLE_WINDOW_SECONDS

    # Create mel spectrogram transform
    mel_transform = None
    if compute_mel:
        mel_transform = create_mel_spectrogram_transform(
            sample_rate=audio_sr,
            n_mels=n_mels,
            stft_window_sec=stft_window_sec,
            stft_hop_sec=stft_hop_sec
        )

    # Calculate IMU window parameters
    imu_sr = DEFAULT_IMU_SR
    window_len_imu = int(imu_window_sec * imu_sr)
    hop_len_imu = hop_length_imu

    print(f"Settings:")
    print(f"  Audio SR: {audio_sr}")
    print(f"  IMU window: {imu_window_sec}s ({window_len_imu} samples)")
    print(f"  IMU hop: {hop_len_imu} samples ({hop_len_imu/imu_sr}s)")
    print(f"  Audio window: {audio_window_sec}s")
    print(f"  STFT window: {stft_window_sec}s, hop: {stft_hop_sec}s")

    # Process all files
    pickle_files = list(input_dir.glob('*.pkl'))

    for file_path in tqdm(pickle_files, desc="Processing files"):
        # Load data
        data = load_samosa_file(file_path)

        # Get IMU data
        imu_data = data['IMU']

        # Get audio data at original 16kHz
        og_audio = data['Audio']

        # Window IMU data
        imu_examples = frame_imu_data(imu_data, window_len_imu, hop_len_imu)

        if compute_mel and mel_transform is not None:
            try:
                # Compute log mel spectrogram on entire audio
                log_mel = compute_log_mel_spectrogram(
                    og_audio,
                    mel_transform,
                    original_sr=DEFAULT_AUDIO_SR,
                    target_sr=audio_sr
                )
            except AssertionError as e:
                if "Signal length shorter than reflect padding length" in str(e):
                    print(f"Warning: Audio too short for {file_path.name}, skipping")
                    continue
                else:
                    raise

            # Align IMU and audio windows
            aligned_imu, aligned_audio = align_imu_audio_windows(
                imu_examples,
                log_mel,
                imu_sr,
                hop_len_imu,
                window_len_imu,
                audio_window_sec=audio_window_sec,
                stft_window_sec=stft_window_sec,
                stft_hop_sec=stft_hop_sec
            )

            if len(aligned_imu) == 0:
                print(f"Warning: No aligned windows for {file_path.name}, skipping")
                continue

            # Convert to arrays
            windowed_arr_imu = np.array(aligned_imu)
            windowed_arr_audio = np.array(aligned_audio)

            # Save processed data
            output_data = {
                'imu': windowed_arr_imu,
                'log_mel': windowed_arr_audio,
                'metadata': {
                    'original_file': file_path.name,
                    'audio_sr': audio_sr,
                    'imu_sr': imu_sr,
                    'imu_window_sec': imu_window_sec,
                    'audio_window_sec': audio_window_sec,
                    'hop_length_imu': hop_len_imu,
                    'num_windows': len(aligned_imu),
                    'n_mels': n_mels,
                    'imu_shape': windowed_arr_imu.shape,
                    'audio_shape': windowed_arr_audio.shape
                }
            }
        else:
            # Save without mel spectrogram
            output_data = {
                'imu': imu_examples,
                'metadata': {
                    'original_file': file_path.name,
                    'imu_sr': imu_sr,
                    'imu_window_sec': imu_window_sec,
                    'hop_length_imu': hop_len_imu,
                    'num_windows': imu_examples.shape[0]
                }
            }

        output_path = output_dir / file_path.name
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)


def main():
    """Main preprocessing function."""
    # Load config file first to get defaults
    config = load_config("config/default.yaml")

    # Extract values from config with fallbacks to constants
    config_audio_sr = config.get('data', {}).get('sampling_rate', DEFAULT_AUDIO_SR_TARGET)
    config_n_mels = config.get('audio', {}).get('n_mels', 64)
    config_imu_win_sec = config.get('data', {}).get('imu_win_sec', DEFAULT_IMU_WIN_SEC)
    config_hop_length = config.get('data', {}).get('hop_length', DEFAULT_HOP_LENGTH)
    config_stft_window_sec = config.get('audio', {}).get('stft_window_sec', STFT_WINDOW_LENGTH_SECONDS)
    config_stft_hop_sec = config.get('audio', {}).get('stft_hop_sec', STFT_HOP_LENGTH_SECONDS)

    parser = argparse.ArgumentParser(
        description="Preprocess SAMoSA dataset for AudioIMU training"
    )

    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to input dataset directory'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='Path to output directory for preprocessed data'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Path to config file (default: config/default.yaml)'
    )

    parser.add_argument(
        '--audio-sr',
        type=int,
        default=config_audio_sr,
        help=f'Audio sampling rate (default from config: {config_audio_sr})'
    )

    parser.add_argument(
        '--no-mel',
        action='store_true',
        help='Skip mel spectrogram computation'
    )

    parser.add_argument(
        '--n-mels',
        type=int,
        default=config_n_mels,
        help=f'Number of mel bands (default from config: {config_n_mels})'
    )

    parser.add_argument(
        '--imu-window-sec',
        type=float,
        default=config_imu_win_sec,
        help=f'IMU window size in seconds (default from config: {config_imu_win_sec})'
    )

    parser.add_argument(
        '--hop-length-imu',
        type=int,
        default=config_hop_length,
        help=f'IMU hop length in samples (default from config: {config_hop_length})'
    )

    parser.add_argument(
        '--stft-window-sec',
        type=float,
        default=config_stft_window_sec,
        help=f'STFT window length in seconds (default from config: {config_stft_window_sec})'
    )

    parser.add_argument(
        '--stft-hop-sec',
        type=float,
        default=config_stft_hop_sec,
        help=f'STFT hop length in seconds (default from config: {config_stft_hop_sec})'
    )

    args = parser.parse_args()

    # Run preprocessing
    preprocess_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        audio_sr=args.audio_sr,
        compute_mel=not args.no_mel,
        n_mels=args.n_mels,
        imu_window_sec=args.imu_window_sec,
        hop_length_imu=args.hop_length_imu,
        stft_window_sec=args.stft_window_sec,
        stft_hop_sec=args.stft_hop_sec
    )

    print("Preprocessing completed!")


if __name__ == "__main__":
    main()
