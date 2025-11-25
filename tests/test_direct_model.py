"""Test if model forward pass gives better results than embeddings pipeline."""

import torch
from transformers import AutoTokenizer, BertForMaskedLM
from wavgpt.config import DEVICE, MODEL_NAME, BLOCK_SIZE
from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference

checkpoint_path = '/home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.01_step27000.pt'

print("Loading models...")
model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "Hello, world! This is a test of the WavGPT model."

# Tokenize
tokens = tokenizer(
    text,
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=BLOCK_SIZE,
)
input_ids = tokens.input_ids.to(DEVICE)
attention_mask = tokens.attention_mask.to(DEVICE)
num_real_tokens = attention_mask.sum().item()

print(f"\n{'='*80}")
print("TEST 1: Direct Model Forward Pass (How Training Works)")
print(f"{'='*80}")

with torch.no_grad():
    # Get original hidden states
    outputs = lm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )
    h_orig = outputs.hidden_states[-1]
    
    # Direct model forward (this is what training does!)
    h_refined, h_approx, coeffs_sparse, mask_kept, _ = model(h_orig, training=False)
    
    # Get logits
    logits = lm_model.cls.predictions(h_refined)
    predicted_ids = torch.argmax(logits[0, :num_real_tokens], dim=-1).cpu().tolist()
    reconstructed = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    
    print(f"Original:      {text}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Nonzero coeffs: {mask_kept.sum().item()}")

print(f"\n{'='*80}")
print("TEST 2: Manual Reconstruction (How Embeddings Work)")
print(f"{'='*80}")

with torch.no_grad():
    # Get coeffs from model
    h_orig = outputs.hidden_states[-1]
    _, _, coeffs_sparse, mask_kept, _ = model(h_orig, training=False)
    
    # Manually reconstruct (simulating embeddings pipeline)
    h_approx_manual = model.wavelet_module._idwt_lifting_1d(coeffs_sparse)
    h_refined_manual = model.refinement_network(h_approx_manual, mask_kept)
    
    # Get logits
    logits_manual = lm_model.cls.predictions(h_refined_manual)
    predicted_ids_manual = torch.argmax(logits_manual[0, :num_real_tokens], dim=-1).cpu().tolist()
    reconstructed_manual = tokenizer.decode(predicted_ids_manual, skip_special_tokens=True)
    
    print(f"Original:      {text}")
    print(f"Reconstructed: {reconstructed_manual}")
    
    # Check if they're the same
    print(f"\nAre h_refined the same? {torch.allclose(h_refined, h_refined_manual, atol=1e-6)}")
    print(f"Are logits the same? {torch.allclose(logits, logits_manual, atol=1e-6)}")
    print(f"Are predictions the same? {predicted_ids == predicted_ids_manual}")

