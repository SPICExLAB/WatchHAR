from .fusion_models import ETE_IndividualGatedFusionModel

# Fusion model registry
FUSION_MODELS = {
    'individualgf': ETE_IndividualGatedFusionModel,
}


def get_fusion_model(model_name, **kwargs):
    """Get fusion model by name.

    Args:
        model_name (str): Name of the fusion model
        **kwargs: Additional arguments for model initialization

    Returns:
        nn.Module: Fusion model instance
    """
    if model_name not in FUSION_MODELS:
        raise ValueError(f"Unknown fusion model: {model_name}. Available: {list(FUSION_MODELS.keys())}")

    return FUSION_MODELS[model_name](**kwargs)


__all__ = [
    'ETE_IndividualGatedFusionModel',
    'get_fusion_model'
]