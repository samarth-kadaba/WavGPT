"""Test that gradients flow correctly through the STE hard masks."""

import torch
from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.config import BLOCK_SIZE, HIDDEN_SIZE, KEEP_RATIO, WAVELET_LEVELS, REFINE_N_LAYERS, REFINE_N_HEADS, REFINE_DIM_FEEDFORWARD, USE_TEMPORAL_ATTENTION, DEVICE

def test_gradient_flow():
    """Verify gradients flow to all learnable parameters, especially importance scores."""
    
    print("="*80)
    print("Testing Gradient Flow Through Hard Masks (STE)")
    print("="*80)
    
    # Create model
    model = HybridWaveletRefinementModel(
        seq_len=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        keep_ratio=KEEP_RATIO,
        wavelet_levels=WAVELET_LEVELS,
        refine_n_layers=REFINE_N_LAYERS,
        refine_n_heads=REFINE_N_HEADS,
        refine_dim_feedforward=REFINE_DIM_FEEDFORWARD,
        use_temporal_attention=USE_TEMPORAL_ATTENTION,
    ).to(DEVICE)
    
    # Create dummy input
    batch_size = 2
    h_orig = torch.randn(batch_size, BLOCK_SIZE, HIDDEN_SIZE, device=DEVICE, requires_grad=True)
    
    # Store initial parameter values
    initial_band_importance = model.frequency_filter.band_importance.clone()
    initial_dim_importance = model.frequency_filter.dim_importance.clone()
    initial_predict_weights = model.wavelet_module.predict_weights[0].clone()
    
    print(f"\n1. Forward pass with hard masks...")
    h_refined, h_approx, coeffs_sparse, mask_kept, importance_map = model(h_orig, training=True)
    
    # Check sparsity
    num_nonzero = (coeffs_sparse != 0).sum().item()
    num_total = coeffs_sparse.numel()
    print(f"   Sparsity: {num_nonzero:,} / {num_total:,} ({num_nonzero/num_total:.2%})")
    print(f"   Mask sum: {mask_kept.sum().item():,.0f}")
    
    if abs(num_nonzero - mask_kept.sum().item()) / num_total > 0.01:
        print(f"   ⚠️  WARNING: Soft masks detected!")
    else:
        print(f"   ✓ Hard masks confirmed")
    
    print(f"\n2. Computing loss and backprop...")
    # Simple loss: just L2 on refined hidden states
    loss = h_refined.mean()
    loss.backward()
    
    print(f"\n3. Checking gradients...")
    
    # Check if parameters have gradients
    params_with_grad = []
    params_without_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad.append((name, param.grad.abs().mean().item()))
        else:
            params_without_grad.append(name)
    
    print(f"\n   Parameters WITH gradients ({len(params_with_grad)}):")
    for name, grad_mean in params_with_grad[:10]:  # Show first 10
        print(f"     ✓ {name}: grad_mean={grad_mean:.6f}")
    if len(params_with_grad) > 10:
        print(f"     ... and {len(params_with_grad) - 10} more")
    
    if params_without_grad:
        print(f"\n   Parameters WITHOUT gradients ({len(params_without_grad)}):")
        for name in params_without_grad[:5]:
            print(f"     ✗ {name}")
        if len(params_without_grad) > 5:
            print(f"     ... and {len(params_without_grad) - 5} more")
    
    # Check critical parameters
    print(f"\n4. Critical parameter gradient check:")
    
    critical_params = [
        ('frequency_filter.band_importance', model.frequency_filter.band_importance),
        ('frequency_filter.dim_importance', model.frequency_filter.dim_importance),
        ('frequency_filter.temperature', model.frequency_filter.temperature),
        ('wavelet_module.predict_weights[0]', model.wavelet_module.predict_weights[0]),
        ('wavelet_module.update_weights[0]', model.wavelet_module.update_weights[0]),
    ]
    
    all_good = True
    for name, param in critical_params:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            print(f"   ✓ {name}:")
            print(f"       grad_norm={grad_norm:.6f}, grad_mean={grad_mean:.6f}")
        else:
            print(f"   ✗ {name}: NO GRADIENT!")
            all_good = False
    
    print(f"\n5. Gradient flow through importance scores (most critical for STE):")
    band_grad = model.frequency_filter.band_importance.grad
    if band_grad is not None:
        print(f"   ✓ band_importance gradient:")
        print(f"       shape={band_grad.shape}")
        print(f"       norm={band_grad.norm().item():.6f}")
        print(f"       mean={band_grad.mean().item():.6f}")
        print(f"       max={band_grad.abs().max().item():.6f}")
        print(f"       nonzero={(band_grad != 0).sum().item()} / {band_grad.numel()}")
    else:
        print(f"   ✗ band_importance: NO GRADIENT!")
        all_good = False
    
    print(f"\n" + "="*80)
    if all_good:
        print("✅ SUCCESS: All critical gradients flowing correctly!")
        print("   The STE hard masks are working as expected.")
    else:
        print("❌ FAILURE: Some gradients missing!")
        print("   The STE implementation may have issues.")
    print("="*80)
    
    return all_good


if __name__ == "__main__":
    with torch.no_grad():
        torch.manual_seed(42)
    
    success = test_gradient_flow()
    exit(0 if success else 1)

