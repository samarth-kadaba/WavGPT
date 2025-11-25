"""Test embedding and decoding roundtrip functionality."""

import sys
from transformers import BertForMaskedLM, AutoTokenizer

from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.config import DEVICE, MODEL_NAME
from wavgpt.embed.embeddings import WavGPTEmbedder
from tests.similarity_utils import verify_text_roundtrip


def test_embedding_roundtrip(checkpoint_path: str):
    """
    Test the full encode/decode cycle for sparse wavelet embeddings.
    
    Tests:
    1. Embedding text to sparse COO format
    2. Decoding sparse format back to text
    3. Round-trip accuracy metrics
    """
    print("="*80)
    print("Testing Sparse Coefficient Storage & Decoding")
    print("="*80)
    
    # Load models
    print(f"\nLoading checkpoint: {checkpoint_path}")
    model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
    
    print("Loading BERT model and tokenizer...")
    lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    lm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create embedder
    embedding_model = WavGPTEmbedder(model, lm_model, lm_tokenizer)
    
    # Test texts
    test_texts = [
        "Hello, world! This is a test of the WavGPT model.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can compress and reconstruct text efficiently.",
    ]
    
    print("\n" + "="*80)
    print("EMBEDDING TESTS")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        print("-" * 80)
        
        # Test embedding
        embedding = embedding_model.embed(text)
        print(f"  Embedding values shape:  {embedding['values'].shape}")
        print(f"  Embedding indices shape: {embedding['indices'].shape}")
        print(f"  Sparse coefficients:     {len(embedding['values'])} nonzeros")
        print(f"  Real tokens:             {embedding['num_real_tokens']}")
        
        # Calculate compression
        total_size = embedding['shape'][0] * embedding['shape'][1]
        sparsity = len(embedding['values']) / total_size
        print(f"  Sparsity:                {sparsity:.2%}")
        
        # Test decoding
        reconstructed = embedding_model.decode(embedding)
        print(f"  Reconstructed:           '{reconstructed}'")
        
        # Round-trip test using test utils
        coeffs_preserved, reconstructed, metrics = verify_text_roundtrip(embedding_model, text)
        print(f"\n  Round-trip Metrics:")
        print(f"    Coefficients preserved:  {coeffs_preserved}")
        print(f"    Positional accuracy:     {metrics['positional_accuracy']:.2%}")
        print(f"    Token recall:            {metrics['token_recall']:.2%}")
        print(f"    Token precision:         {metrics['token_precision']:.2%}")
        print(f"    Token overlap:           {metrics['token_overlap']}/{metrics['original_unique_tokens']}")
    
    print("\n" + "="*80)
    print("✓ Embedding roundtrip tests complete!")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Default checkpoint
        checkpoint_path = '/home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.01_step36500.pt'
    
    test_embedding_roundtrip(checkpoint_path)

