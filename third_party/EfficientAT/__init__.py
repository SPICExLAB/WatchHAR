# EfficientAT audio models
# See LICENSE and NOTICE.md for attribution

from .mn.model import get_model, MN, mobilenet_v3

def get_mn(width_mult=0.5, pretrained_name="mn05_as", num_classes=527,
           feature_extraction=True, **kwargs):
    """Get MobileNetV3 model for audio feature extraction.

    Args:
        width_mult: Width multiplier (0.5 for mn05)
        pretrained_name: Pretrained model name
        num_classes: Number of output classes
        feature_extraction: If True, return features instead of logits
    """
    return get_model(
        num_classes=num_classes,
        pretrained_name=pretrained_name,
        width_mult=width_mult,
        feature_extraction=feature_extraction,
        head_type="mlp",
        **kwargs
    )
