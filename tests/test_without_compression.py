"""Test wavelet reconstruction WITHOUT compressed sensing to isolate the problem."""

from transformers import AutoTokenizer, BertForMaskedLM
import torch
from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.config import DEVICE, MODEL_NAME

def test_without_compression():
    """Test if the wavelet transform alone works, skipping compressed sensing."""
    
    # Load model
    checkpoint_path = '/home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.00390625_step1000.pt'
    model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
    
    # Load BERT
    lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Test text
    text = "Hello, world! This is a test of the WavGPT model."
    
    # Tokenize
    tokens = tokenizer(text, return_tensors='pt', padding='max_length', 
                      truncation=True, max_length=256)
    input_ids = tokens.input_ids.to(DEVICE)
    attention_mask = tokens.attention_mask.to(DEVICE)
    num_real_tokens = attention_mask.sum().item()
    
    print(f"Original text: {text}")
    print(f"Real tokens: {num_real_tokens}")
    
    with torch.no_grad():
        # Get original hidden states
        outputs = lm_model(input_ids=input_ids, attention_mask=attention_mask, 
                          output_hidden_states=True)
        h_orig = outputs.hidden_states[-1]
        
        # Run through model (wavelet + learned filter)
        # IMPORTANT: Use training=True to get soft masks (what model was trained on)
        # Using training=False gives hard masks which creates train/test mismatch!
        h_refined, h_approx, coeffs_sparse, mask_kept, importance_map = model(h_orig, training=True)
        
        print(f"\nWavelet stats:")
        print(f"  Coeffs kept: {mask_kept.sum().item()} / {mask_kept.numel()}")
        print(f"  Keep ratio: {mask_kept.sum().item() / mask_kept.numel():.6f}")
        
        # Decode from h_approx (no refinement, just wavelet reconstruction)
        logits_approx = lm_model.cls.predictions(h_approx)
        predicted_ids_approx = torch.argmax(logits_approx[0, :num_real_tokens], dim=-1).cpu().tolist()
        decoded_approx = tokenizer.decode(predicted_ids_approx, skip_special_tokens=True)
        
        # Decode from original (no compression)
        logits_orig = lm_model.cls.predictions(h_orig)
        predicted_ids_orig = torch.argmax(logits_orig[0, :num_real_tokens], dim=-1).cpu().tolist()
        decoded_orig = tokenizer.decode(predicted_ids_orig, skip_special_tokens=True)
        
        print(f"\nResults:")
        print(f"  Original (no compression): {decoded_orig}")
        print(f"  After wavelet (no CS):     {decoded_approx}")
        print(f"  Expected:                  {text}")

if __name__ == "__main__":
    test_without_compression()

