import torch
import torch.nn as nn


class CNN2D_3L(nn.Module):
    """2D CNN model for IMU-based activity recognition.

    Hardcoded for 1-second window (50 samples at 50Hz).

    Args:
        num_sensors (int): Number of sensor channels (default: 9 for acc+gyro+rotvec)
        num_classes (int): Number of output classes (default: 27)
        win_size (int): Window size in samples (default: 50, must be 50)
        cnn_channels (int): Number of CNN channels (default: 256)
        dropout (float): Dropout rate (default: 0.5)
    """

    def __init__(self,
                 num_sensors=9,
                 num_classes=27,
                 win_size=50,
                 cnn_channels=256,
                 dropout=0.5):
        super().__init__()

        self.num_classes = num_classes

        # Feature extraction layers
        kernel_size = (5, 1)

        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(num_sensors, cnn_channels, kernel_size=kernel_size),
            nn.GroupNorm(4, cnn_channels),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=kernel_size),
            nn.GroupNorm(4, cnn_channels),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=kernel_size),
            nn.GroupNorm(4, cnn_channels),
            nn.ReLU(inplace=True),
        )

        # Feature dimension calculated dynamically based on win_size:
        # After Conv1: win_size-4, After Pool1: (win_size-4)//2
        # After Conv2: -4, After Pool2: //2
        # After Conv3: -4
        # Formula: (win_size - 28) // 4 * cnn_channels
        feature_dim = (win_size - 28) // 4 * cnn_channels

        # Feature transformation layers
        self.preprocess_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Classification layer
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, is_ete=False):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 50, num_sensors)
            is_ete (bool): If True, return 128-dim features for fusion (end-to-end mode).
                          If False, return class predictions with sigmoid.

        Returns:
            torch.Tensor: 128-dim features (is_ete=True) or class predictions (is_ete=False)
        """
        # Reshape input: (batch, win_size, sensors) -> (batch, sensors, win_size, 1)
        x = x.unsqueeze(1)
        x = x.permute(0, 3, 2, 1)

        # Extract CNN features
        x = self.features(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Transform features
        features = self.preprocess_classifier(x)

        if is_ete:
            # Return features for fusion model
            return features
        else:
            # Return class predictions with sigmoid
            return torch.sigmoid(self.classifier(features))
