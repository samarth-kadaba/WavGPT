"""Complete checkpoint management for WavGPT models.

This module provides all checkpoint functionality:
- save_checkpoint: Save complete checkpoint with training state
- load_checkpoint_for_training: Resume training with optimizer/scheduler
- load_checkpoint_for_inference: Load model for inference only
"""

import torch
import os
import glob
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    losses,
    metrics,
    config,
    save_path,
):
    """
    Save a complete checkpoint with all information needed to resume training or load for inference.
    
    Uses atomic write (temp file + rename) and automatically deletes old checkpoints to save disk space.
    
    Args:
        model: The model to save
        optimizer: The optimizer (will save state dict)
        scheduler: The learning rate scheduler (will save state dict)
        epoch: Current epoch number
        global_step: Current global step
        losses: Dictionary of loss values
        metrics: Dictionary of metrics (perplexity, compression ratio, etc.)
        config: Dictionary of model configuration parameters
        save_path: Path where checkpoint will be saved
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_path).parent
    if save_dir != Path('.'):
        save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        # Model weights
        'model_state_dict': model.state_dict(),
        
        # Optimizer and scheduler states (for resuming training)
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        
        # Training state
        'epoch': epoch,
        'global_step': global_step,
        
        # Model architecture configuration (needed to reconstruct model)
        'config': {
            'seq_len': config['seq_len'],
            'hidden_size': config['hidden_size'],
            'keep_ratio': config['keep_ratio'],
            'wavelet_levels': config['wavelet_levels'],
            'refine_n_layers': config['refine_n_layers'],
            'refine_n_heads': config['refine_n_heads'],
            'refine_dim_feedforward': config['refine_dim_feedforward'],
            'use_temporal_attention': config['use_temporal_attention'],
        },
        
        # Training hyperparameters (for reference)
        'hyperparameters': {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'temperature': config.get('temperature', 2.0),
            'warmup_ratio': config.get('warmup_ratio', 0.05),
            'batch_size': config.get('batch_size', None),
        },
        
        # Current losses
        'losses': {
            'loss_total': losses['loss_total'].item() if torch.is_tensor(losses['loss_total']) else losses['loss_total'],
            'loss_hidden_refined': losses['loss_hidden_refined'].item() if torch.is_tensor(losses['loss_hidden_refined']) else losses['loss_hidden_refined'],
            'loss_hidden_approx': losses['loss_hidden_approx'].item() if torch.is_tensor(losses['loss_hidden_approx']) else losses['loss_hidden_approx'],
            'loss_logits': losses['loss_logits'].item() if torch.is_tensor(losses['loss_logits']) else losses['loss_logits'],
            'loss_ce': losses['loss_ce'].item() if torch.is_tensor(losses['loss_ce']) else losses['loss_ce'],
            'loss_kl': losses['loss_kl'].item() if torch.is_tensor(losses['loss_kl']) else losses['loss_kl'],
        },
        
        # Current metrics
        'metrics': {
            'ppl_original': metrics['ppl_original'],
            'ppl_refined': metrics['ppl_refined'],
            'ppl_degradation': metrics['ppl_degradation'],
            'compression_ratio': metrics['compression_ratio'],
        },
    }
    
    # Delete old checkpoints before saving new one to prevent disk space issues
    # Look for checkpoints with the same ratio pattern
    save_dir_str = str(save_dir)
    keep_ratio = config['keep_ratio']
    pattern = os.path.join(save_dir_str, f"hybrid_wavelet_model_ratio{keep_ratio}_step*.pt")
    old_checkpoints = glob.glob(pattern)
    
    for old_checkpoint in old_checkpoints:
        try:
            os.remove(old_checkpoint)
            print(f"  Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
        except Exception as e:
            print(f"  Warning: Could not delete {old_checkpoint}: {e}")
    
    # Save to temporary file first (atomic write to prevent corruption)
    temp_path = str(save_path) + ".tmp"
    torch.save(checkpoint, temp_path)
    
    # Rename temp file to final path (atomic operation on most filesystems)
    os.replace(temp_path, save_path)
    
    print(f"✓ Checkpoint saved to {save_path}")
    
    return save_path


def load_checkpoint_for_training(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load a checkpoint and restore model, optimizer, and scheduler states for resuming training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing epoch, global_step, config, losses, and metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✓ Checkpoint loaded for training from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}, Global Step: {checkpoint['global_step']}")
    print(f"  PPL Degradation: {checkpoint['metrics']['ppl_degradation']:.2f}")
    
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'config': checkpoint['config'],
        'losses': checkpoint['losses'],
        'metrics': checkpoint['metrics'],
        'hyperparameters': checkpoint.get('hyperparameters', {}),
    }


def load_checkpoint_for_inference(checkpoint_path, device='cpu'):
    """
    Load checkpoint and create model for inference only (no optimizer/scheduler).
    
    This creates a fresh model instance from the saved config and loads weights.
    Use this when you just want to run inference.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        model: Loaded model in eval mode
        checkpoint_info: Dictionary with config, metrics, losses, etc.
    """
    from wavgpt.models import HybridWaveletRefinementModel
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    device = torch.device(device)
    
    # Create model from saved config
    model = HybridWaveletRefinementModel(
        seq_len=config['seq_len'],
        hidden_size=config['hidden_size'],
        keep_ratio=config['keep_ratio'],
        wavelet_levels=config['wavelet_levels'],
        refine_n_layers=config['refine_n_layers'],
        refine_n_heads=config['refine_n_heads'],
        refine_dim_feedforward=config['refine_dim_feedforward'],
        use_temporal_attention=config['use_temporal_attention'],
    ).to(device)
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare checkpoint info
    checkpoint_info = {
        'config': config,
        'epoch': checkpoint.get('epoch', -1),
        'global_step': checkpoint.get('global_step', -1),
        'metrics': checkpoint.get('metrics', {}),
        'losses': checkpoint.get('losses', {}),
        'hyperparameters': checkpoint.get('hyperparameters', {}),
    }
    
    print(f"✓ Model loaded for inference from {checkpoint_path}")
    if checkpoint_info['epoch'] >= 0:
        print(f"  Epoch: {checkpoint_info['epoch']}, Global Step: {checkpoint_info['global_step']}")
    if checkpoint_info['metrics']:
        ppl_deg = checkpoint_info['metrics'].get('ppl_degradation', 'N/A')
        print(f"  PPL Degradation: {ppl_deg}")
    
    return model, checkpoint_info

