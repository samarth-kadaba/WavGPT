"""Utilities for computing and comparing similarity metrics on embeddings."""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

from wavgpt.embed.embeddings import WavGPTEmbedder


def cosine_similarity_sparse(embedding1: csr_matrix, embedding2: csr_matrix) -> float:
    """
    Compute cosine similarity between two sparse embeddings (CSR format).
    
    Args:
        embedding1, embedding2: Sparse matrices in CSR format (1, D)
        
    Returns:
        Cosine similarity score (scalar)
    """
    return cosine_similarity(embedding1, embedding2)[0, 0]


def semantic_similarity_decoded(embedder: WavGPTEmbedder, text1: str, text2: str) -> float:
    """
    Compare semantic similarity via DECODED hidden states, not raw coefficients.
    
    Key insight: If semantic information is preserved through lossy compression,
    it should appear in the decoded representations, even if raw sparse coefficients
    don't overlap much due to different sparsity patterns.
    
    Args:
        embedder: WavGPTEmbedder instance
        text1, text2: Input texts to compare
        
    Returns:
        Cosine similarity of decoded CLS embeddings
    """
    with torch.no_grad():
        # Get sparse coefficients and masks for both texts
        coeffs1, mask1, ids1, attn1 = embedder._text_to_coeffs(text1)
        coeffs2, mask2, ids2, attn2 = embedder._text_to_coeffs(text2)
        
        # Decode sparse coefficients back to hidden states
        h_approx1 = embedder.model.wavelet_module._idwt_lifting_1d(coeffs1)
        h_refined1 = embedder.model.refinement_network(h_approx1, mask1)
        
        h_approx2 = embedder.model.wavelet_module._idwt_lifting_1d(coeffs2)
        h_refined2 = embedder.model.refinement_network(h_approx2, mask2)
        
        # Use CLS position (first token) as the semantic embedding, like BERT
        emb1 = h_refined1[0, 0, :]  # (768,)
        emb2 = h_refined2[0, 0, :]  # (768,)
        
        # Compute cosine similarity in decoded hidden space
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0)
        ).item()
        
        return similarity


def compute_frequency_metrics(embedder: WavGPTEmbedder, 
                              coeffs1: torch.Tensor, 
                              coeffs2: torch.Tensor) -> dict:
    """
    Compute frequency-based similarity metrics.
    
    Args:
        embedder: WavGPTEmbedder instance (for accessing band layout)
        coeffs1, coeffs2: Sparse wavelet coefficients
        
    Returns:
        Dictionary with frequency-based similarity metrics
    """
    # Get frequency band layout from the model
    band_layout = embedder.model.frequency_filter.band_layout
    
    # Compute energy per band for each text
    energy1 = []
    energy2 = []
    active_bands1 = set()
    active_bands2 = set()
    
    for band_idx, band in enumerate(band_layout):
        start = band['start']
        length = band['length']
        end = start + length
        
        # Extract coefficients for this band
        band_coeffs1 = coeffs1[0, start:end, :]  # (length, hidden_size)
        band_coeffs2 = coeffs2[0, start:end, :]
        
        # Compute energy (L2 norm)
        e1 = torch.norm(band_coeffs1).item()
        e2 = torch.norm(band_coeffs2).item()
        
        energy1.append(e1)
        energy2.append(e2)
        
        # Mark as active if energy > threshold
        if e1 > 0.01:
            active_bands1.add(band_idx)
        if e2 > 0.01:
            active_bands2.add(band_idx)
    
    # Convert to numpy for easier computation
    energy1 = np.array(energy1)
    energy2 = np.array(energy2)
    
    # Metric 1: Jaccard similarity of active bands
    if len(active_bands1) == 0 and len(active_bands2) == 0:
        band_overlap = 1.0
    elif len(active_bands1 | active_bands2) == 0:
        band_overlap = 0.0
    else:
        band_overlap = len(active_bands1 & active_bands2) / len(active_bands1 | active_bands2)
    
    # Metric 2: Correlation of energy distributions
    if np.std(energy1) > 1e-8 and np.std(energy2) > 1e-8:
        energy_correlation = np.corrcoef(energy1, energy2)[0, 1]
    else:
        energy_correlation = 0.0
    
    # Metric 3: Cosine similarity of energy vectors
    norm1 = np.linalg.norm(energy1)
    norm2 = np.linalg.norm(energy2)
    if norm1 > 1e-8 and norm2 > 1e-8:
        band_cosine = np.dot(energy1, energy2) / (norm1 * norm2)
    else:
        band_cosine = 0.0
    
    return {
        'band_overlap': band_overlap,
        'energy_correlation': energy_correlation,
        'band_cosine': band_cosine,
    }


def comprehensive_similarity(embedder: WavGPTEmbedder, text1: str, text2: str) -> dict:
    """
    Compute ALL similarity metrics for comprehensive comparison.
    
    Returns dict with:
    - BERT baselines (original hidden states)
    - Sparse coefficient similarities
    - Decoded hidden state similarities
    - Frequency-based similarities
    
    Args:
        embedder: WavGPTEmbedder instance
        text1, text2: Texts to compare
        
    Returns:
        Dictionary with all similarity metrics
    """
    with torch.no_grad():
        # Get original BERT hidden states and sparse coefficients
        coeffs1, mask1, ids1, attn1 = embedder._text_to_coeffs(text1)
        coeffs2, mask2, ids2, attn2 = embedder._text_to_coeffs(text2)
        
        # Also get original BERT hidden states directly
        tokens1 = embedder.lm_tokenizer(text1, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=embedder.seq_len)
        tokens2 = embedder.lm_tokenizer(text2, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=embedder.seq_len)
        
        outputs1 = embedder.lm_model(
            input_ids=tokens1.input_ids.to(embedder.lm_model.device),
            attention_mask=tokens1.attention_mask.to(embedder.lm_model.device),
            output_hidden_states=True
        )
        outputs2 = embedder.lm_model(
            input_ids=tokens2.input_ids.to(embedder.lm_model.device),
            attention_mask=tokens2.attention_mask.to(embedder.lm_model.device),
            output_hidden_states=True
        )
        
        h_orig1 = outputs1.hidden_states[-1]  # (1, T, 768)
        h_orig2 = outputs2.hidden_states[-1]
        
        # === BERT BASELINES ===
        # 1. BERT CLS (standard approach)
        bert_cls1 = h_orig1[0, 0, :]
        bert_cls2 = h_orig2[0, 0, :]
        bert_cls_sim = torch.nn.functional.cosine_similarity(
            bert_cls1.unsqueeze(0), bert_cls2.unsqueeze(0)
        ).item()
        
        # 2. BERT Mean Pooling (with attention mask)
        mask1_float = tokens1.attention_mask.unsqueeze(-1).float().to(embedder.lm_model.device)
        mask2_float = tokens2.attention_mask.unsqueeze(-1).float().to(embedder.lm_model.device)
        bert_mean1 = (h_orig1 * mask1_float).sum(dim=1) / mask1_float.sum(dim=1)
        bert_mean2 = (h_orig2 * mask2_float).sum(dim=1) / mask2_float.sum(dim=1)
        bert_mean_sim = torch.nn.functional.cosine_similarity(
            bert_mean1, bert_mean2
        ).item()
        
        # === SPARSE COEFFICIENTS ===
        # 3. Raw sparse coefficient cosine
        coeffs1_flat = coeffs1.flatten()
        coeffs2_flat = coeffs2.flatten()
        norm1 = torch.norm(coeffs1_flat)
        norm2 = torch.norm(coeffs2_flat)
        if norm1 > 1e-8 and norm2 > 1e-8:
            sparse_coeff_sim = torch.dot(coeffs1_flat, coeffs2_flat) / (norm1 * norm2)
            sparse_coeff_sim = sparse_coeff_sim.item()
        else:
            sparse_coeff_sim = 0.0
        
        # === DECODED HIDDEN STATES ===
        # Decode sparse coefficients back to hidden states
        h_approx1 = embedder.model.wavelet_module._idwt_lifting_1d(coeffs1)
        h_refined1 = embedder.model.refinement_network(h_approx1, mask1)
        
        h_approx2 = embedder.model.wavelet_module._idwt_lifting_1d(coeffs2)
        h_refined2 = embedder.model.refinement_network(h_approx2, mask2)
        
        # 4. Decoded CLS
        decoded_cls1 = h_refined1[0, 0, :]
        decoded_cls2 = h_refined2[0, 0, :]
        decoded_cls_sim = torch.nn.functional.cosine_similarity(
            decoded_cls1.unsqueeze(0), decoded_cls2.unsqueeze(0)
        ).item()
        
        # 5. Decoded Mean Pooling
        attn1_3d = attn1.unsqueeze(-1).float()
        attn2_3d = attn2.unsqueeze(-1).float()
        decoded_mean1 = (h_refined1 * attn1_3d).sum(dim=1) / attn1_3d.sum(dim=1)
        decoded_mean2 = (h_refined2 * attn2_3d).sum(dim=1) / attn2_3d.sum(dim=1)
        decoded_mean_sim = torch.nn.functional.cosine_similarity(
            decoded_mean1, decoded_mean2
        ).item()
        
        # === FREQUENCY SPACE ===
        freq_metrics = compute_frequency_metrics(embedder, coeffs1, coeffs2)
        
        return {
            # BERT Baselines
            'bert_cls': bert_cls_sim,
            'bert_mean_pool': bert_mean_sim,
            
            # Sparse Coefficients
            'sparse_coeff': sparse_coeff_sim,
            
            # Decoded Hidden States
            'decoded_cls': decoded_cls_sim,
            'decoded_mean_pool': decoded_mean_sim,
            
            # Frequency Space
            'freq_band_overlap': freq_metrics['band_overlap'],
            'freq_energy_correlation': freq_metrics['energy_correlation'],
            'freq_energy_cosine': freq_metrics['band_cosine'],
        }


def verify_text_roundtrip(embedder: WavGPTEmbedder, text: str) -> tuple:
    """
    Test full round-trip: text → embed → decode → text.
    
    Args:
        embedder: WavGPTEmbedder instance
        text: Input text to test
        
    Returns:
        coeffs_match: Whether coefficients were preserved through embedding
        reconstructed_text: The decoded text
        metrics: Reconstruction quality metrics
    """
    from wavgpt.config import DEVICE
    
    # Get original coefficients
    coeffs_sparse, mask_kept, input_ids, attention_mask = embedder._text_to_coeffs(text)
    num_real_tokens = attention_mask.sum().item()
    
    # Encode to sparse format
    embedding = embedder.embed(text)
    
    # Decode back
    reconstructed_text = embedder.decode(embedding)
    
    # Check coefficient preservation (should be perfect with direct storage)
    coeffs_recon = torch.zeros_like(coeffs_sparse)
    mask_recon = torch.zeros_like(mask_kept, dtype=torch.bool)
    
    indices = embedding['indices'].to(DEVICE)
    values = embedding['values'].to(DEVICE)
    
    for (t, d), val in zip(indices, values):
        coeffs_recon[0, t, d] = val
        mask_recon[0, t, d] = True
    
    coeffs_match = torch.allclose(coeffs_sparse, coeffs_recon, rtol=1e-5, atol=1e-7)
    mask_match = torch.equal(mask_kept.bool(), mask_recon.bool())
    coeffs_preserved = coeffs_match and mask_match
    
    # Calculate reconstruction metrics
    original_tokens = embedder.lm_tokenizer.convert_ids_to_tokens(
        input_ids[0, :num_real_tokens].cpu().tolist())
    original_tokens = [t for t in original_tokens 
                      if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    
    reconstructed_ids = embedder.lm_tokenizer.encode(reconstructed_text, add_special_tokens=False)
    reconstructed_tokens = embedder.lm_tokenizer.convert_ids_to_tokens(reconstructed_ids)
    reconstructed_tokens = [t for t in reconstructed_tokens 
                           if t not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']]
    
    # Compute token overlap (bag-of-words style)
    original_set = set(original_tokens)
    reconstructed_set = set(reconstructed_tokens)
    token_overlap = len(original_set & reconstructed_set)
    token_recall = token_overlap / len(original_set) if len(original_set) > 0 else 0
    token_precision = token_overlap / len(reconstructed_set) if len(reconstructed_set) > 0 else 0
    
    # Also compute positional accuracy
    original_ids_clean = input_ids[0, :num_real_tokens].cpu().tolist()
    reconstructed_ids_padded = reconstructed_ids[:num_real_tokens]
    if len(reconstructed_ids_padded) < num_real_tokens:
        reconstructed_ids_padded += [embedder.lm_tokenizer.pad_token_id] * \
                                    (num_real_tokens - len(reconstructed_ids_padded))
    positional_matches = sum(1 for o, r in zip(original_ids_clean, reconstructed_ids_padded) if o == r)
    positional_accuracy = positional_matches / num_real_tokens if num_real_tokens > 0 else 0
    
    metrics = {
        'coeffs_preserved': coeffs_preserved,
        'positional_accuracy': positional_accuracy,
        'token_recall': token_recall,
        'token_precision': token_precision,
        'token_overlap': token_overlap,
        'original_unique_tokens': len(original_set),
        'reconstructed_unique_tokens': len(reconstructed_set),
        'num_nonzero_coeffs': len(values),
    }
    
    return coeffs_preserved, reconstructed_text, metrics

