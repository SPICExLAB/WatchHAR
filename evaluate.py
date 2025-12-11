#!/usr/bin/env python3
"""Evaluate trained AudioIMU models using LOPO validation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.config import (
    get_args_parser,
    load_config,
    process_args,
    setup_device
)
from utils.constants import (
    MODEL_FEATURE_DIMS,
    SAMOSA_CLASS_LABEL_MAPPING,
    SAMOSA_CONTEXTS,
    ACTIVITY_TO_CONTEXT,
    SAMOSA_INDEX_TO_ACTIVITY
)
from data import (
    AudioIMUDataset,
    load_normalization_params
)
from models.imu import get_imu_model
from third_party.EfficientAT import get_mn
from models.fusion import get_fusion_model


def collate_fn_pad_audio(batch):
    """Custom collate function that pads audio tensors to the same size.

    Args:
        batch: List of (imu_tensor, audio_tensor, label_tensor) tuples

    Returns:
        Batched tensors with audio padded to max temporal dimension in batch
    """
    imu_tensors, audio_tensors, label_tensors = zip(*batch)

    # Stack IMU and labels normally (same size)
    imu_batch = torch.stack(imu_tensors, dim=0)
    label_batch = torch.stack(label_tensors, dim=0)

    # Find max temporal dimension in audio (shape is [time, freq])
    max_time = max(audio.shape[0] for audio in audio_tensors)

    # Pad audio tensors to max_time
    padded_audio = []
    for audio in audio_tensors:
        if audio.shape[0] < max_time:
            # Pad on the right side (end of time sequence)
            pad_size = max_time - audio.shape[0]
            padding = torch.zeros(pad_size, audio.shape[1], dtype=audio.dtype)
            padded = torch.cat([audio, padding], dim=0)
            padded_audio.append(padded)
        else:
            padded_audio.append(audio)

    audio_batch = torch.stack(padded_audio, dim=0)

    return imu_batch, audio_batch, label_batch


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


def load_model(model_path: Path, config: dict, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    # Create model (checkpoint weights will override pretrained weights)
    model = create_model(config, device)
    
    # Wrap with DataParallel if needed
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Adjust state dict for DataParallel mismatch
    new_state_dict = {}
    model_keys = set(model.state_dict().keys())
    has_module = any(key.startswith('module.') for key in model_keys)
    
    for key, value in state_dict.items():
        if key.startswith('module.') and not has_module:
            new_key = key[7:]  # Remove 'module.' prefix
        elif not key.startswith('module.') and has_module:
            new_key = 'module.' + key  # Add 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model


def evaluate_participant(
    model: nn.Module,
    test_dataset: AudioIMUDataset,
    device: torch.device,
    config: dict
) -> dict:
    """Evaluate model on a single participant's data.

    Matches audioIMU evaluation exactly:
    - Frame-wise: per-window argmax
    - File-wise: sum probabilities across windows, then argmax
    - Context-wise per-file: sum probabilities per file, constrain to context classes, then argmax
    - Context-independent: file-wise unconstrained predictions

    Returns:
        Dictionary containing all prediction results for different evaluation modes.
    """

    # Frame-wise storage
    all_predictions = []
    all_true_labels = []
    all_outputs = []

    # File-wise storage (accumulate probabilities by file)
    file_outputs = defaultdict(list)
    file_labels = {}
    file_activities = {}

    model.eval()

    # Create DataLoader for batched evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_pad_audio
    )

    with torch.no_grad():
        for imu_tensor, audio_tensor, label_tensor in test_loader:
            imu_tensor = imu_tensor.to(device)
            audio_tensor = audio_tensor.to(device)

            # Add channel dimension to audio if needed
            if len(audio_tensor.shape) == 3:
                audio_tensor = audio_tensor.unsqueeze(1)

            # Forward pass
            outputs = model(imu_tensor, audio_tensor)
            predicted_labels = outputs.argmax(dim=1)
            true_labels = label_tensor.argmax(dim=1)

            # Store frame-wise results
            all_predictions.extend(predicted_labels.cpu().tolist())
            all_true_labels.extend(true_labels.cpu().tolist())
            all_outputs.extend(outputs.cpu().numpy())

    # Process each example for file-wise evaluation
    for i, example in enumerate(test_dataset.examples):
        if i >= len(all_predictions):
            break

        file_id = example.get('file_id', f"{example['participant_id']}_{example['activity']}")
        activity = example['activity']
        true_label = all_true_labels[i]
        output = all_outputs[i]

        # Accumulate outputs for file-wise evaluation
        file_outputs[file_id].append(output)
        file_labels[file_id] = true_label
        file_activities[file_id] = activity

    # File-wise predictions: SUM probabilities across windows (not mean!)
    file_predictions = []
    file_true_labels_list = []

    # Context-wise per-file storage (using accumulated file probabilities)
    context_file_predictions = defaultdict(list)
    context_file_true_labels = defaultdict(list)

    # Context-independent storage (file-wise without constraint)
    context_independent_predictions = []
    context_independent_true_labels = []

    for file_id, outputs_list in file_outputs.items():
        # Sum probabilities across all windows of this file (audioIMU style)
        accumulated_output = np.sum(outputs_list, axis=0)
        file_pred = np.argmax(accumulated_output)

        file_predictions.append(file_pred)
        file_true_labels_list.append(file_labels[file_id])

        # Context-independent: file-wise unconstrained
        context_independent_predictions.append(file_pred)
        context_independent_true_labels.append(file_labels[file_id])

        # Context-wise per-file: constrain prediction to context classes
        activity = file_activities[file_id]
        context = ACTIVITY_TO_CONTEXT.get(activity, None)

        if context is not None and context in SAMOSA_CONTEXTS:
            context_classes = [SAMOSA_CLASS_LABEL_MAPPING[act]
                             for act in SAMOSA_CONTEXTS[context]
                             if act in SAMOSA_CLASS_LABEL_MAPPING]

            # Constrain prediction to only context classes using accumulated output
            context_outputs = accumulated_output[context_classes]
            context_pred_idx = np.argmax(context_outputs)
            context_predicted = context_classes[context_pred_idx]

            context_file_predictions[context].append(context_predicted)
            context_file_true_labels[context].append(file_labels[file_id])

    return {
        'frame_predictions': all_predictions,
        'frame_true_labels': all_true_labels,
        'file_predictions': file_predictions,
        'file_true_labels': file_true_labels_list,
        'context_file_predictions': context_file_predictions,
        'context_file_true_labels': context_file_true_labels,
        'context_independent_predictions': context_independent_predictions,
        'context_independent_true_labels': context_independent_true_labels
    }


def calculate_metrics(predictions: list, true_labels: list) -> dict:
    """Calculate evaluation metrics."""
    metrics = {
        'accuracy': balanced_accuracy_score(true_labels, predictions),
        'f1_score': f1_score(true_labels, predictions, average='weighted', zero_division=0),
        'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0)
    }
    return metrics


def plot_confusion_matrix(true_labels: list, predictions: list, save_path: Path):
    """Plot and save confusion matrix."""
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Get class names
    class_names = [SAMOSA_INDEX_TO_ACTIVITY[i] for i in range(len(SAMOSA_CLASS_LABEL_MAPPING))]
    
    # Plot
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_lopo(config: dict, exp_dir: Path):
    """Evaluate all LOPO models and aggregate results."""
    logger = logging.getLogger(__name__)
    device = setup_device(config)

    # Find all model files
    model_files = list(exp_dir.glob('best_model_excluded_*.pt'))

    if not model_files:
        logger.error("No trained models found!")
        return

    logger.info(f"Found {len(model_files)} trained models")

    # Aggregate results
    all_frame_metrics = defaultdict(list)
    all_file_metrics = defaultdict(list)
    all_context_file_metrics = defaultdict(lambda: defaultdict(list))
    all_context_independent_metrics = defaultdict(list)

    all_frame_predictions = []
    all_frame_true_labels = []
    all_file_predictions = []
    all_file_true_labels = []
    all_context_independent_predictions = []
    all_context_independent_true_labels = []

    data_path = Path(config['data']['dataset_path'])

    # Load normalization parameters (SINGLE FILE FOR ALL PARTICIPANTS)
    norm_params_path = exp_dir / 'normalization_params.pkl'
    if not norm_params_path.exists():
        logger.error(f"Normalization parameters not found: {norm_params_path}")
        return

    normalization_params = load_normalization_params(str(norm_params_path))
    logger.info(f"Loaded normalization parameters from {norm_params_path}")

    for model_file in model_files:
        # Extract participant ID from filename
        participant_id = model_file.stem.split('_')[-1]
        logger.info(f"Evaluating model for participant {participant_id}")

        # Load test data for this participant using AudioIMUDataset
        test_dataset = AudioIMUDataset(
            data_path=str(data_path),
            normalization_params=normalization_params,
            participants=[participant_id],
            sensors=config['imu']['sensors'],
            num_classes=config['model']['num_classes']
        )

        # Load model
        model = load_model(model_file, config, device)

        # Evaluate - returns dictionary with frame-wise and file-wise results
        results = evaluate_participant(model, test_dataset, device, config)

        frame_preds = results['frame_predictions']
        frame_true = results['frame_true_labels']
        file_preds = results['file_predictions']
        file_true = results['file_true_labels']
        context_file_preds = results['context_file_predictions']
        context_file_true = results['context_file_true_labels']
        context_indep_preds = results['context_independent_predictions']
        context_indep_true = results['context_independent_true_labels']

        # Store frame-wise results
        all_frame_predictions.extend(frame_preds)
        all_frame_true_labels.extend(frame_true)

        # Store file-wise results
        all_file_predictions.extend(file_preds)
        all_file_true_labels.extend(file_true)

        # Store context-independent results
        all_context_independent_predictions.extend(context_indep_preds)
        all_context_independent_true_labels.extend(context_indep_true)

        # Calculate frame-wise metrics
        if frame_preds and frame_true:
            frame_metrics = calculate_metrics(frame_preds, frame_true)
            for metric, value in frame_metrics.items():
                all_frame_metrics[metric].append(value)
            logger.info(f"Participant {participant_id} - Frame accuracy: {frame_metrics['accuracy']:.4f}")

        # Calculate file-wise metrics
        if file_preds and file_true:
            file_metrics = calculate_metrics(file_preds, file_true)
            for metric, value in file_metrics.items():
                all_file_metrics[metric].append(value)
            logger.info(f"Participant {participant_id} - File accuracy: {file_metrics['accuracy']:.4f}")

        # Calculate context-independent metrics
        if context_indep_preds and context_indep_true:
            context_indep_metrics = calculate_metrics(context_indep_preds, context_indep_true)
            for metric, value in context_indep_metrics.items():
                all_context_independent_metrics[metric].append(value)

        # Context-wise per-file metrics
        for context in context_file_preds:
            if context_file_preds[context] and context_file_true[context]:
                context_metrics = calculate_metrics(context_file_preds[context], context_file_true[context])
                for metric, value in context_metrics.items():
                    all_context_file_metrics[context][metric].append(value)

    # Generate final results
    results_file = exp_dir / 'evaluation_results.txt'

    with open(results_file, 'w') as f:
        f.write("AudioIMU Evaluation Results\n")
        f.write("=" * 50 + "\n\n")

        # Frame-wise results
        f.write("Frame-wise Results (per window):\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if all_frame_metrics[metric]:
                mean_val = np.mean(all_frame_metrics[metric])
                std_val = np.std(all_frame_metrics[metric])
                f.write(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

        # File-wise results (summed probs, like audioIMU)
        f.write("File-wise Results (per recording, summed probs):\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if all_file_metrics[metric]:
                mean_val = np.mean(all_file_metrics[metric])
                std_val = np.std(all_file_metrics[metric])
                f.write(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

        # Context-independent results (file-wise, unconstrained)
        f.write("Context-independent Results (file-wise, unconstrained):\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if all_context_independent_metrics[metric]:
                mean_val = np.mean(all_context_independent_metrics[metric])
                std_val = np.std(all_context_independent_metrics[metric])
                f.write(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

        # Context-wise per-file results (with constrained prediction)
        f.write("Context-wise Per-file Results (summed probs, constrained):\n")
        f.write("-" * 30 + "\n")
        for context in all_context_file_metrics:
            f.write(f"\n{context} Context:\n")
            for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
                if metric in all_context_file_metrics[context] and all_context_file_metrics[context][metric]:
                    mean_val = np.mean(all_context_file_metrics[context][metric])
                    std_val = np.std(all_context_file_metrics[context][metric])
                    f.write(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("\n")

        # Overall Context-wise metrics (mean of per-context means, like audioIMU)
        f.write("Overall Context-wise Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            context_means = [
                np.mean(all_context_file_metrics[context][metric])
                for context in all_context_file_metrics
                if metric in all_context_file_metrics[context] and all_context_file_metrics[context][metric]
            ]
            if context_means:
                overall_mean = np.mean(context_means)
                f.write(f"{metric.capitalize()}: {overall_mean:.4f}\n")
        f.write("\n")
    
    # Generate classification report
    report = classification_report(
        all_frame_true_labels,
        all_frame_predictions,
        target_names=[SAMOSA_INDEX_TO_ACTIVITY[i] for i in range(len(SAMOSA_CLASS_LABEL_MAPPING))],
        output_dict=False
    )
    
    with open(exp_dir / 'classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_frame_true_labels,
        all_frame_predictions,
        exp_dir / 'confusion_matrix.png'
    )
    
    logger.info(f"Evaluation completed! Results saved to {exp_dir}")

    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 30)
    if all_frame_metrics['accuracy']:
        mean_frame_acc = np.mean(all_frame_metrics['accuracy'])
        mean_frame_f1 = np.mean(all_frame_metrics['f1_score'])
        print(f"Frame-wise Accuracy: {mean_frame_acc:.4f}")
        print(f"Frame-wise F1-Score: {mean_frame_f1:.4f}")

    if all_file_metrics['accuracy']:
        mean_file_acc = np.mean(all_file_metrics['accuracy'])
        mean_file_f1 = np.mean(all_file_metrics['f1_score'])
        print(f"File-wise Accuracy:  {mean_file_acc:.4f}")
        print(f"File-wise F1-Score:  {mean_file_f1:.4f}")

    # Overall context-wise accuracy (mean of per-context means)
    context_acc_means = [
        np.mean(all_context_file_metrics[context]['accuracy'])
        for context in all_context_file_metrics
        if 'accuracy' in all_context_file_metrics[context] and all_context_file_metrics[context]['accuracy']
    ]
    if context_acc_means:
        overall_context_acc = np.mean(context_acc_means)
        print(f"Overall Context-wise Accuracy: {overall_context_acc:.4f}")


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = get_args_parser()
    parser.add_argument(
        '--experiment-dir',
        type=str,
        required=True,
        help='Path to experiment directory with trained models'
    )
    
    args = parser.parse_args()
    
    # Load config from experiment directory
    exp_dir = Path(args.experiment_dir)
    config_path = exp_dir / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(str(config_path))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation
    evaluate_lopo(config, exp_dir)


if __name__ == "__main__":
    main()