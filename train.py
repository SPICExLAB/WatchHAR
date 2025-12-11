import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import logging
from typing import List, Dict, Tuple, Optional

from utils.config import (
    get_args_parser, 
    load_config, 
    process_args, 
    validate_config,
    create_experiment_dir,
    setup_device
)
from utils.constants import (
    MODEL_FEATURE_DIMS,
    SAMOSA_CLASS_LABEL_MAPPING,
    SAMOSA_CONTEXTS,
    ACTIVITY_TO_CONTEXT
)
from data import (
    AudioIMUDataset,
    compute_normalization_params,
    save_normalization_params,
    load_normalization_params,
    DataAugmentation
)
from models.imu import get_imu_model
from third_party.EfficientAT import get_mn
from models.fusion import get_fusion_model


def setup_logging(exp_dir: Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(exp_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_participant_ids(data_path: Path) -> List[str]:
    """Get all unique participant IDs from dataset."""
    participant_ids = set()
    
    for file_path in data_path.glob('*.pkl'):
        participant_id = file_path.name.split('---')[0]
        participant_ids.add(participant_id)
    
    return sorted(list(participant_ids))


def get_lopo_splits(participant_ids: List[str], 
                   excluded_participant: str,
                   val_seed: int = 4) -> Tuple[List[str], List[str], List[str]]:
    """Get LOPO train/val/test splits."""
    # Remove excluded participant
    remaining_ids = [pid for pid in participant_ids if pid != excluded_participant]
    
    # Choose validation participant
    np.random.seed(val_seed)
    val_participant = np.random.choice(remaining_ids, 1)[0]
    
    # Create splits
    train_participants = [pid for pid in remaining_ids if pid != val_participant]
    val_participants = [val_participant]
    test_participants = [excluded_participant]
    
    return train_participants, val_participants, test_participants


def create_model(config: dict, device: torch.device) -> nn.Module:
    """Create the multimodal fusion model.

    Args:
        config: Configuration dictionary
        device: Device to create model on

    Returns:
        Fusion model with AudioSet pretrained weights for audio component.
        For evaluation/conversion, checkpoint loading will override these weights.
    """
    # Get feature dimensions
    imu_feature_size = MODEL_FEATURE_DIMS['imu'][config['model']['imu_model']]
    audio_feature_size = MODEL_FEATURE_DIMS['audio'][config['model']['audio_model']]

    # Create IMU model
    imu_model = get_imu_model(
        config['model']['imu_model'],
        num_sensors=len(config['imu']['sensors'].split(',')) * 3 if config['imu']['sensors'] != 'all' else 9,
        num_classes=config['model']['num_classes'],
        win_size=config['imu']['window_length'],
        cnn_channels=config['model']['imu']['cnn_channels'],
        dropout=config['model']['imu']['dropout']
    )

    # Create audio model with AudioSet pretrained weights
    audio_model = get_mn(
        num_classes=config['model']['num_classes'],
        pretrained_name=f"{config['model']['audio_model']}_as",
        width_mult=config['model']['audio']['width_mult'],
        input_dim_t=config['model']['audio']['input_dim_t'],
        input_dim_f=config['model']['audio']['input_dim_f'],
        feature_extraction=config['model']['audio']['feature_extraction']
    )
    
    # Create fusion model
    fusion_model = get_fusion_model(
        config['model']['fusion_model'],
        imu_model=imu_model,
        audio_model=audio_model,
        num_classes=config['model']['num_classes'],
        imu_feature_size=imu_feature_size,
        audio_feature_size=audio_feature_size,
        hidden_dim=config['model']['fusion']['hidden_dim'],
        dropout=config['model']['fusion']['dropout']
    )
    
    return fusion_model.to(device)


def validate_epoch(model: nn.Module,
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  epoch: int) -> Tuple[float, float, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Context-wise metrics
    context_predictions = defaultdict(list)
    context_true_labels = defaultdict(list)
    
    with torch.no_grad():
        for imu_data, audio_data, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
            imu_data = imu_data.to(device)
            audio_data = audio_data.to(device)
            labels = labels.to(device)
            
            # Add channel dimension to audio: (batch, time, freq) -> (batch, 1, time, freq)
            if len(audio_data.shape) == 3:
                audio_data = audio_data.unsqueeze(1)

            # Forward pass
            outputs = model(imu_data, audio_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)
            
            correct_predictions += (predicted == true_labels).sum().item()
            total_samples += labels.size(0)
            
            # Store for context-wise accuracy
            for i in range(len(true_labels)):
                label_idx = true_labels[i].item()
                pred_idx = predicted[i].item()
                
                # Get activity and context
                activity = list(SAMOSA_CLASS_LABEL_MAPPING.keys())[label_idx]
                context = ACTIVITY_TO_CONTEXT.get(activity, 'Other')
                
                if context not in ['Other', 'All']:
                    context_predictions[context].append(pred_idx)
                    context_true_labels[context].append(label_idx)
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    frame_accuracy = correct_predictions / total_samples
    
    # Context-wise accuracy
    context_correct = sum(
        1 for context in context_predictions
        for pred, true in zip(context_predictions[context], context_true_labels[context])
        if pred == true
    )
    context_total = sum(len(labels) for labels in context_true_labels.values())
    context_accuracy = context_correct / context_total if context_total > 0 else 0
    
    return avg_loss, frame_accuracy, context_accuracy


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                config: dict) -> float:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    augmentation = DataAugmentation()
    
    for batch_idx, (imu_data, audio_data, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        imu_data = imu_data.to(device)
        audio_data = audio_data.to(device)
        labels = labels.to(device)
        
        # Apply R-Frame transformation (random temporal offset augmentation)
        # Note: With pre-windowed data (input length == win_length), this applies
        # random roll offset without changing batch size
        if config['training']['augmentation'].get('rframe_transformation', False):
            win_length = config['imu']['window_length']
            step_size = win_length // 2
            delta_range = win_length // 2
            imu_data, labels = augmentation.rframe_transformation(
                imu_data, labels, win_length, step_size, delta_range
            )

        # Apply channel dropout
        if config['training']['augmentation']['channel_dropout']:
            imu_data = augmentation.channel_dropout(imu_data)

        # Add channel dimension to audio: (batch, time, freq) -> (batch, 1, time, freq)
        if len(audio_data.shape) == 3:
            audio_data = audio_data.unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(imu_data, audio_data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % config['output']['log_interval'] == 0:
            logging.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(train_loader)


def train_lopo(config: dict, participant_id: str, exp_dir: Path, norm_params: Dict):
    """Train model with LOPO validation for one participant."""
    logger = logging.getLogger(__name__)
    logger.info(f"Training LOPO model excluding participant: {participant_id}")

    # Setup device
    device = setup_device(config)

    # Get data splits
    data_path = Path(config['data']['dataset_path'])
    all_participants = get_participant_ids(data_path)
    train_participants, val_participants, test_participants = get_lopo_splits(
        all_participants, participant_id, config['validation']['val_participant_seed']
    )

    logger.info(f"Train participants: {len(train_participants)}")
    logger.info(f"Val participants: {val_participants}")
    logger.info(f"Test participants: {test_participants}")
    
    # Create datasets (use AudioIMUDataset for preprocessed data)
    train_dataset = AudioIMUDataset(
        data_path=str(data_path),
        normalization_params=norm_params,
        participants=train_participants,
        sensors=config['imu']['sensors'],
        num_classes=config['model']['num_classes']
    )

    val_dataset = AudioIMUDataset(
        data_path=str(data_path),
        normalization_params=norm_params,
        participants=val_participants,
        sensors=config['imu']['sensors'],
        num_classes=config['model']['num_classes']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Calculate class weights
    if config['training']['loss']['use_class_weights']:
        class_indices = train_dataset.get_class_indices()
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(class_indices),
            y=class_indices
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    else:
        class_weights = None
    
    # Create model with AudioSet pretrained weights for initialization
    model = create_model(config, device)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss function (model outputs logits, use BCEWithLogitsLoss for numerical stability)
    if config['training']['loss']['criterion'] == 'bce':
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=config['training']['patience'],
        min_lr=config['training']['min_lr']
    )
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, config)
        
        # Validate
        val_loss, frame_acc, context_acc = validate_epoch(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log results
        logger.info(f"Epoch {epoch}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Frame Acc: {frame_acc:.4f}, Context Acc: {context_acc:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            model_path = exp_dir / f'best_model_excluded_{participant_id}.pt'
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved best model with val loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping after {epoch} epochs")
            break
    
    logger.info(f"Training completed for participant {participant_id}")


def main():
    """Main training function."""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Load and process config
    config = load_config(args.config)
    config = process_args(args, config)

    # Validate config
    validate_config(config)

    # Create experiment directory
    exp_dir = create_experiment_dir(config)

    # Setup logging
    logger = setup_logging(exp_dir)
    logger.info("Starting AudioIMU training")
    logger.info(f"Experiment directory: {exp_dir}")

    # Get all participant IDs
    data_path = Path(config['data']['dataset_path'])
    all_participants = get_participant_ids(data_path)

    # Compute or load normalization parameters (SINGLE FILE FOR ALL PARTICIPANTS)
    norm_params_path = exp_dir / 'normalization_params.pkl'
    if norm_params_path.exists():
        logger.info(f"Loading existing normalization parameters from {norm_params_path}")
        norm_params = load_normalization_params(str(norm_params_path))
    else:
        logger.info("Computing normalization parameters from ALL participants")
        norm_params = compute_normalization_params(
            str(data_path),
            sensors=config['imu']['sensors'],
            participants=None  # Use ALL participants
        )
        save_normalization_params(norm_params, str(norm_params_path))
        logger.info(f"Normalization parameters saved to {norm_params_path}")

    # Filter participants based on arguments
    if config['target_participants']:
        participants_to_train = [p for p in all_participants if p in config['target_participants']]
    else:
        participants_to_train = all_participants

    # Remove already trained participants
    participants_to_train = [p for p in participants_to_train if p not in config['trained_participants']]

    logger.info(f"Total participants to train: {len(participants_to_train)}")

    # Train LOPO models (reuse same normalization params for all)
    for participant_id in participants_to_train:
        train_lopo(config, participant_id, exp_dir, norm_params)

    logger.info("All training completed!")


if __name__ == "__main__":
    main()