from .cnn2d import CNN2D_3L

# Model registry
IMU_MODELS = {
    'cnn2d': CNN2D_3L,
}


def get_imu_model(model_name, **kwargs):
    """Get IMU model by name.
    
    Args:
        model_name (str): Name of the model
        **kwargs: Additional arguments for model initialization
    
    Returns:
        nn.Module: IMU model instance
    """
    if model_name not in IMU_MODELS:
        raise ValueError(f"Unknown IMU model: {model_name}. Available: {list(IMU_MODELS.keys())}")
    
    return IMU_MODELS[model_name](**kwargs)


__all__ = ['CNN2D_3L', 'get_imu_model']