import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from utils.constants import DATA_FILE_PATTERN, SENSOR_INDICES


class AudioIMUDataset(Dataset):
    """Dataset for loading preprocessed Audio-IMU data.

    This dataset loads data that has already been windowed and preprocessed
    by preprocess.py. This is more efficient for training as windows and
    mel spectrograms are pre-computed.

    Args:
        data_path: Path to the preprocessed dataset directory
        normalization_params: Dictionary containing normalization parameters
        participants: List of participant IDs to include (if None, include all)
        exclude_participants: List of participant IDs to exclude
        sensors: Which sensors to use ('all', 'acc', 'gyro', 'rotvec', or comma-separated)
        num_classes: Number of activity classes
    """

    def __init__(self,
                 data_path: str,
                 normalization_params: Dict[str, np.ndarray],
                 participants: Optional[List[str]] = None,
                 exclude_participants: Optional[List[str]] = None,
                 sensors: str = 'all',
                 num_classes: int = 27,
                 transform=None):

        self.data_path = Path(data_path)
        self.normalization_params = normalization_params
        self.num_classes = num_classes
        self.transform = transform

        # Setup sensor indices
        self.sensor_indices = self._get_sensor_indices(sensors)

        # Load all preprocessed windows
        self.examples = self._load_preprocessed_data(participants, exclude_participants)

    def _get_sensor_indices(self, sensors: str) -> np.ndarray:
        """Get sensor indices based on sensor configuration."""
        if sensors == 'all':
            return slice(0, 9)

        indices = []
        for sensor in sensors.split(','):
            sensor = sensor.strip()
            if sensor in SENSOR_INDICES:
                sensor_slice = SENSOR_INDICES[sensor]
                indices.extend(range(sensor_slice.start, sensor_slice.stop))

        return np.array(indices) if indices else slice(0, 9)

    def _load_preprocessed_data(self,
                                 participants: Optional[List[str]],
                                 exclude_participants: Optional[List[str]]) -> List[Dict]:
        """Load preprocessed windowed data from pickle files."""
        from utils.constants import SAMOSA_CLASS_LABEL_MAPPING

        examples = []

        # Get all pickle files
        pickle_files = list(self.data_path.glob('*.pkl'))

        for file_path in pickle_files:
            # Parse filename to get metadata
            match = re.match(DATA_FILE_PATTERN, file_path.name)
            if not match:
                continue

            participant_id = match.group(1)
            context = match.group(2)
            activity = match.group(3)
            trial = int(match.group(4))

            # Check participant filters
            if participants and participant_id not in participants:
                continue
            if exclude_participants and participant_id in exclude_participants:
                continue

            # Get class label
            if activity not in SAMOSA_CLASS_LABEL_MAPPING:
                continue

            class_idx = SAMOSA_CLASS_LABEL_MAPPING[activity]

            # Load pickle file
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # Preprocessed format: arrays of shape (num_windows, window_len, features)
            if 'imu' in data and 'log_mel' in data:
                imu_windows = data['imu']
                audio_windows = data['log_mel']

                # Create file identifier for grouping windows
                file_id = f"{participant_id}---{context}---{activity}---{trial}"

                for i in range(len(imu_windows)):
                    imu_data = imu_windows[i][:, self.sensor_indices]
                    example = {
                        'imu': imu_data,
                        'audio': audio_windows[i],
                        'label': class_idx,
                        'participant_id': participant_id,
                        'activity': activity,
                        'context': context,
                        'trial': trial,
                        'file_id': file_id,
                        'window_idx': i
                    }
                    examples.append(example)

        return examples

    def normalize_imu(self, imu_data: np.ndarray) -> np.ndarray:
        """Normalize IMU data using provided normalization parameters."""
        pseudo_max = self.normalization_params["max"]
        pseudo_min = self.normalization_params["min"]
        mean = self.normalization_params["mean"]
        std = self.normalization_params["std"]

        # Slice normalization params to match selected sensors
        if isinstance(self.sensor_indices, np.ndarray):
            pseudo_max = pseudo_max[:, self.sensor_indices]
            pseudo_min = pseudo_min[:, self.sensor_indices]
            mean = mean[:, self.sensor_indices]
            std = std[:, self.sensor_indices]

        # Apply normalization
        normalized = 1 + (imu_data - pseudo_max) * 2 / (pseudo_max - pseudo_min)
        normalized = (normalized - mean) / std

        return np.ascontiguousarray(normalized)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get preprocessed item from dataset.

        Returns:
            Tuple of (imu_data, audio_data, label)
        """
        example = self.examples[idx]

        # Get data (copy to avoid shared memory issues)
        imu_data = example['imu'].astype(np.float32).copy()
        audio_data = example['audio'].astype(np.float32).copy()
        label = example['label']

        # Normalize IMU data
        imu_data = self.normalize_imu(imu_data)

        # Convert to tensors using torch.tensor (creates new storage, unlike from_numpy)
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)

        # Create one-hot label
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        label_tensor[label] = 1.0

        # Apply transforms if any
        if self.transform:
            imu_tensor, audio_tensor = self.transform(imu_tensor, audio_tensor)

        return imu_tensor, audio_tensor, label_tensor

    def get_class_indices(self) -> List[int]:
        """Get list of class indices for all examples."""
        return [example['label'] for example in self.examples]

    def get_participant_ids(self) -> List[str]:
        """Get unique participant IDs in this dataset."""
        return list(set(example['participant_id'] for example in self.examples))
