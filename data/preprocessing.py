import numpy as np
import torch
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
from tqdm import tqdm

from utils.constants import DATA_FILE_PATTERN, SENSOR_INDICES


def compute_normalization_params(data_path: str, 
                               sensors: str = 'all',
                               participants: List[str] = None) -> Dict[str, np.ndarray]:
    """Compute normalization parameters for IMU data.
    
    Args:
        data_path: Path to dataset directory
        sensors: Which sensors to use
        participants: List of participants to include (if None, use all)
    
    Returns:
        Dictionary with normalization parameters
    """
    data_path = Path(data_path)
    
    # Get sensor indices
    if sensors == 'all':
        sensor_indices = slice(0, 9)
    else:
        indices = []
        for sensor in sensors.split(','):
            sensor = sensor.strip()
            if sensor in SENSOR_INDICES:
                sensor_slice = SENSOR_INDICES[sensor]
                indices.extend(range(sensor_slice.start, sensor_slice.stop))
        sensor_indices = np.array(indices) if indices else slice(0, 9)
    
    # Collect all IMU data
    all_imu_data = []

    for file_path in tqdm(data_path.glob('*.pkl'), desc="Loading IMU data"):
        # Check if we should include this participant
        if participants:
            participant_id = file_path.name.split('---')[0]
            if participant_id not in participants:
                continue

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Handle both preprocessed format ('imu') and raw format ('IMU')
        if 'imu' in data:
            # Preprocessed format: (num_windows, time_steps, channels)
            imu_data = data['imu']
            # Reshape to (num_windows * time_steps, channels) for stats computation
            imu_data = imu_data.reshape(-1, imu_data.shape[-1])
            imu_data = imu_data[:, sensor_indices]
        else:
            # Raw format: (time_steps, channels)
            imu_data = data['IMU'][:, sensor_indices]

        all_imu_data.append(imu_data)

    # Concatenate all data
    all_imu_data = np.concatenate(all_imu_data, axis=0)
    
    # Compute normalization parameters
    normalization_params = {
        'max': np.percentile(all_imu_data, 80, axis=0, keepdims=True),
        'min': np.percentile(all_imu_data, 20, axis=0, keepdims=True),
        'mean': np.mean(all_imu_data, axis=0, keepdims=True),
        'std': np.std(all_imu_data, axis=0, keepdims=True)
    }
    
    return normalization_params


def save_normalization_params(params: Dict[str, np.ndarray], save_path: str):
    """Save normalization parameters to file."""
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_normalization_params(load_path: str) -> Dict[str, np.ndarray]:
    """Load normalization parameters from file."""
    with open(load_path, 'rb') as f:
        return pickle.load(f)


class DataAugmentation:
    """Data augmentation techniques for AudioIMU."""
    
    @staticmethod
    def channel_dropout(imu_data: torch.Tensor, p: float = 0.2) -> torch.Tensor:
        """Apply channel dropout to IMU data.

        Args:
            imu_data: IMU data tensor of shape (batch, time, channels)
            p: Dropout probability

        Returns:
            Augmented IMU data
        """
        if not torch.is_tensor(imu_data):
            imu_data = torch.tensor(imu_data)

        # Create dropout mask
        batch_size, _, num_channels = imu_data.shape
        mask = torch.rand(batch_size, 1, num_channels, device=imu_data.device) > p

        # Apply mask
        return imu_data * mask.float()

    @staticmethod
    def rframe_transformation(data_x: torch.Tensor, data_y: torch.Tensor,
                             win_length: int = 50, step_size: int = 25,
                             delta_range: int = 25) -> tuple:
        """Apply R-Frame transformation for data augmentation.

        Applies random temporal offset (via roll) to IMU sequences. If input sequences
        are longer than win_length, also extracts overlapping subsequences to increase
        training samples. With pre-windowed data (sequence length == win_length), this
        only applies the random offset without expanding batch size.

        Args:
            data_x: IMU data tensor of shape (batch, sequence_length, num_channels)
            data_y: One-hot encoded labels of shape (batch, num_classes)
            win_length: Length of each window/frame to extract
            step_size: Step size for sliding window (only used if sequence_length > win_length)
            delta_range: Maximum random offset value for temporal shift

        Returns:
            Tuple of (transformed_data_x, transformed_data_y)
            - If sequence_length == win_length: batch size unchanged
            - If sequence_length > win_length: batch size increases
        """
        transformed_data_x = []
        transformed_data_y = []

        for sequence, label in zip(data_x, data_y):
            # Ensure sequence is a tensor
            if not isinstance(sequence, torch.Tensor):
                sequence = torch.from_numpy(sequence).float()
            if not isinstance(label, torch.Tensor):
                label = torch.from_numpy(label).float()

            # Generate random offset ensuring it's within sequence length
            max_delta_value = min(delta_range, sequence.shape[0])
            delta = torch.randint(0, max_delta_value, (1,)).item()

            # Apply offset by rolling the sequence
            offset_sequence = torch.roll(sequence, -delta, dims=0)

            # Extract subsequences with sliding window
            for start_idx in range(0, sequence.size(0) - win_length + 1, step_size):
                subsequence = offset_sequence[start_idx:start_idx + win_length, :]
                transformed_data_x.append(subsequence)
                transformed_data_y.append(label)

        # Stack tensors
        transformed_data_x = torch.stack(transformed_data_x)
        transformed_data_y = torch.stack(transformed_data_y)

        return transformed_data_x, transformed_data_y
    
