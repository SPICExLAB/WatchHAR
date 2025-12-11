import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup device based on configuration and availability."""
    device_str = config.get('hardware', {}).get('device', 'cuda')
    
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def get_args_parser():
    """Get argument parser for command line arguments."""
    parser = argparse.ArgumentParser(description="AudioIMU Training")
    
    # Config file
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    
    # Override arguments
    parser.add_argument('--data-path', type=str, help='Path to dataset')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--save-dir', type=str, help='Save directory')
    parser.add_argument('--experiment-name', type=str, help='Experiment name')
    
    # Model selection
    parser.add_argument('--imu-model', type=str, help='IMU model type')
    parser.add_argument('--audio-model', type=str, help='Audio model type')
    parser.add_argument('--fusion-model', type=str, help='Fusion model type')
    
    # Training modes
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # LOPO settings
    parser.add_argument('--target-participants', nargs='+', default=[], 
                       help='List of participant IDs to train')
    parser.add_argument('--trained-participants', nargs='+', default=[],
                       help='List of already trained participant IDs')
    
    return parser


def process_args(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process command line arguments and update config."""
    
    # Override config with command line arguments
    if args.data_path:
        config['data']['dataset_path'] = args.data_path
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.save_dir:
        config['output']['save_dir'] = args.save_dir
    if args.experiment_name:
        config['output']['experiment_name'] = args.experiment_name
    
    # Model overrides
    if args.imu_model:
        config['model']['imu_model'] = args.imu_model
    if args.audio_model:
        config['model']['audio_model'] = args.audio_model
    if args.fusion_model:
        config['model']['fusion_model'] = args.fusion_model
    
    # Add additional args to config
    config['eval_only'] = args.eval_only
    config['resume'] = args.resume
    config['target_participants'] = args.target_participants
    config['trained_participants'] = args.trained_participants
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values."""
    
    # Check required paths exist
    dataset_path = Path(config['data']['dataset_path'])
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Validate model choices
    valid_imu_models = ['cnn1d', 'cnn2d', 'convlstm', 'attend_discriminate']
    if config['model']['imu_model'] not in valid_imu_models:
        raise ValueError(f"Invalid IMU model: {config['model']['imu_model']}")
    
    valid_audio_models = ['dymn04', 'dymn10', 'dymn20', 'mn05']
    if config['model']['audio_model'] not in valid_audio_models:
        raise ValueError(f"Invalid audio model: {config['model']['audio_model']}")
    
    valid_fusion_models = ['concatenate', 'self_attention', 'modelwise_fc', 'total_fc', 'gatedfusion', 'individualgf']
    if config['model']['fusion_model'] not in valid_fusion_models:
        raise ValueError(f"Invalid fusion model: {config['model']['fusion_model']}")
    
    # Validate numeric values
    if config['training']['batch_size'] <= 0:
        raise ValueError("Batch size must be positive")
    if config['training']['epochs'] <= 0:
        raise ValueError("Epochs must be positive")
    if config['training']['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")


def create_experiment_dir(config: Dict[str, Any]) -> Path:
    """Create experiment directory and save config."""
    save_dir = Path(config['output']['save_dir'])
    exp_name = config['output']['experiment_name']
    
    exp_dir = save_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to experiment directory
    config_save_path = exp_dir / 'config.yaml'
    save_config(config, str(config_save_path))
    
    return exp_dir