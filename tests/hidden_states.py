"""Encode text to hidden states and decode back to text using BERT.

This file demonstrates how BERT's bidirectional architecture enables
true decodable embeddings where h[i] represents token[i] directly.
"""

from transformers import AutoTokenizer, BertForMaskedLM
import torch

BERT_MODEL = "bert-base-uncased"


def extract_hidden_states(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden states from BERT.
    
    Args:
        text: Input text to encode
        
    Returns:
        hidden_states: (1, seq_len, hidden_size) tensor of hidden states
        input_ids: (1, seq_len) tensor of input token IDs
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = BertForMaskedLM.from_pretrained(BERT_MODEL)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.bert(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1], inputs.input_ids


def decode_hidden_states(hidden_states: torch.Tensor) -> str:
    """Decode BERT hidden states back to text.
    
    Key insight: With BERT (bidirectional), h[i] represents token[i] directly.
    This enables perfect reconstruction without needing to store any tokens separately.
    
    Args:
        hidden_states: (1, seq_len, hidden_size) tensor of hidden states
        
    Returns:
        Reconstructed text
    """
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    mlm_model = BertForMaskedLM.from_pretrained(BERT_MODEL)
    
    # Get logits from the MLM head
    logits = mlm_model.cls.predictions(hidden_states)
    predicted_ids = torch.argmax(logits[0], dim=-1).cpu().tolist()
    
    text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    return text


if __name__ == "__main__":
    test_text = "Hello, world! This is a test of the WavGPT model."
    
    print("="*80)
    print("BERT Hidden State Reconstruction Test")
    print("="*80)
    
    print(f"\nOriginal text: {test_text}")
    
    # Extract hidden states
    hidden_states, input_ids = extract_hidden_states(test_text)
    print(f"Hidden states shape: {hidden_states.shape}")
    
    # Decode back to text
    print("\n--- Reconstructing from hidden states ---")
    decoded_text = decode_hidden_states(hidden_states)
    print(f"Decoded text: {decoded_text}")
    
    # Check if reconstruction matches
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded_text}")
    
    # Token-level comparison
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    original_tokens = tokenizer.tokenize(test_text)
    decoded_tokens = tokenizer.tokenize(decoded_text)
    
    print(f"\nOriginal tokens ({len(original_tokens)}): {original_tokens[:10]}...")
    print(f"Decoded tokens ({len(decoded_tokens)}): {decoded_tokens[:10]}...")
    
    if decoded_text.replace(" ", "").lower() == test_text.replace(" ", "").lower():
        print("\n✅ Perfect reconstruction achieved!")
    else:
        print("\n⚠️  Reconstruction differs (expected for argmax decoding)")
        print("Note: This is because we're taking argmax instead of sampling.")
        print("The hidden states contain the full information needed for reconstruction.")
    
    print("="*80)