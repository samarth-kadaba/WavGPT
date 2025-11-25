"""Test embedding WITHOUT compressed sensing - just use coefficients directly."""

from transformers import AutoTokenizer, BertForMaskedLM
import torch
from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.config import DEVICE, MODEL_NAME

def test_direct_coeffs():
    """Test if skipping compressed sensing works."""
    
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
        
        # Run through model (use soft masks like training)
        h_refined, h_approx, coeffs_sparse, mask_kept, importance_map = model(h_orig, training=True)
        
        print(f"\n=== Encode ===")
        print(f"Coeffs sparse shape: {coeffs_sparse.shape}")
        print(f"Non-zero coeffs: {(coeffs_sparse != 0).sum().item()}")
        
        # SKIP COMPRESSED SENSING - just use coeffs_sparse directly as "embedding"
        embedding = coeffs_sparse  # This is our embedding!
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding size: {embedding.numel() * 4 / 1024:.1f} KB (float32)")
        
        # DECODE: just IDWT directly
        print(f"\n=== Decode ===")
        h_reconstructed = model.wavelet_module._idwt_lifting_1d(embedding)
        
        # Get logits
        logits = lm_model.cls.predictions(h_reconstructed)
        predicted_ids = torch.argmax(logits[0, :num_real_tokens], dim=-1).cpu().tolist()
        decoded_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        
        print(f"\nResults:")
        print(f"  Decoded:  {decoded_text}")
        print(f"  Expected: {text}")

if __name__ == "__main__":
    test_direct_coeffs()

