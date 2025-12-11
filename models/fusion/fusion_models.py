import torch
import torch.nn as nn


class IndividualGatedFusionLayer(nn.Module):
    """Gated fusion layer with individual gates for each modality.

    Args:
        feature_size: Dimension of input features for both modalities
    """

    def __init__(self, feature_size):
        super().__init__()
        self.gate_imu = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(feature_size, feature_size)

    def forward(self, imu_features, audio_features):
        """Apply gating and combine features."""
        gate_imu = self.gate_imu(imu_features)
        gate_audio = self.gate_audio(audio_features)
        gated_imu = gate_imu * imu_features
        gated_audio = gate_audio * audio_features
        combined = gated_imu + gated_audio
        return self.transform(combined)


class ETE_IndividualGatedFusionModel(nn.Module):
    """Individual Gated Fusion Model matching audioIMU implementation.

    Uses individual gating for each modality with BatchNorm and
    3-stage classification layers.

    Args:
        imu_model: IMU feature extraction model
        audio_model: Audio feature extraction model
        num_classes: Number of output classes
        imu_feature_size: Dimension of IMU features
        audio_feature_size: Dimension of audio features
        hidden_dim: Hidden dimension for fusion (default: 256)
        dropout: Dropout rate (default: 0.25)
    """

    def __init__(self, imu_model, audio_model, num_classes,
                 imu_feature_size, audio_feature_size,
                 hidden_dim=256, dropout=0.25):
        super().__init__()

        self.imu_model = imu_model
        self.audio_model = audio_model
        self.num_classes = num_classes

        # Modality-specific transformation layers with BatchNorm
        self.transform_imu = nn.Sequential(
            nn.Linear(imu_feature_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        self.transform_audio = nn.Sequential(
            nn.Linear(audio_feature_size, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        # Individual gated fusion layer
        self.fusion_layer = IndividualGatedFusionLayer(hidden_dim)

        # 3-stage classification layers with BatchNorm
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, imu_data, audio_data):
        # Extract features from both modalities
        imu_features = self.imu_model(imu_data, is_ete=True)
        audio_features = self.audio_model(audio_data)

        # Transform features with BatchNorm
        imu_features = self.transform_imu(imu_features)
        audio_features = self.transform_audio(audio_features)

        # Fuse with individual gating
        features = self.fusion_layer(imu_features, audio_features)

        # Classify (no sigmoid - will be applied in BCEWithLogitsLoss for stability)
        out = self.fc_layers(features)

        return out
