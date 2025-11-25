"""Interface for generating embeddings from a trained WavGPT model."""

from typing import Dict, Tuple

import torch
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, BertForMaskedLM
from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.config import HIDDEN_SIZE, BLOCK_SIZE, DEVICE, MODEL_NAME
from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference

class WavGPTEmbedder:
    def __init__(self, model: HybridWaveletRefinementModel, lm_model: BertForMaskedLM, lm_tokenizer: AutoTokenizer):
        """Initialize embedder with pre-loaded models.
        
        Args:
            model: Pre-loaded and configured WavGPT model (use load_checkpoint_for_inference)
            lm_model: Pre-loaded BERT model for masked language modeling
            lm_tokenizer: Tokenizer for the BERT model
        """
        self.model = model
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        if getattr(self.lm_model.config, "pad_token_id", None) is None:
            self.lm_model.config.pad_token_id = self.lm_tokenizer.pad_token_id
        self.lm_model.eval()
        self.model.eval()
        self.seq_len = getattr(self.model, "seq_len", BLOCK_SIZE)
        self.hidden_size = getattr(self.model, "hidden_size", HIDDEN_SIZE)

    def embed(self, text: str) -> Dict:
        """
        Embed text into sparse coefficient representation (COO format).
        
        Returns:
            embedding: Dict containing sparse coefficients in COO format
                - 'indices': (k, 2) tensor of [time, dimension] positions
                - 'values': (k,) tensor of coefficient values
                - 'shape': (seq_len, hidden_size)
                - 'num_real_tokens': Number of non-pad tokens
        """
        coeffs_sparse, mask_kept, input_ids, attention_mask = self._text_to_coeffs(text)
        
        # Convert to sparse COO format
        batch_idx = 0
        nonzero_mask = mask_kept[batch_idx].bool()  # (T, d) - ensure bool type
        nonzero_positions = torch.nonzero(nonzero_mask, as_tuple=False)  # (k, 2)
        nonzero_values = coeffs_sparse[batch_idx][nonzero_mask]  # (k,)
        
        num_real_tokens = attention_mask.sum().item()
        
        return {
            'indices': nonzero_positions.cpu(),
            'values': nonzero_values.cpu(),
            'shape': (self.seq_len, self.hidden_size),
            'num_real_tokens': num_real_tokens
        }
    
    def embed_to_csr(self, text: str) -> csr_matrix:
        """
        Embed text and return as CSR sparse matrix (efficient for similarity computation).
        
        Returns:
            Sparse matrix in CSR format (1, seq_len * hidden_size) flattened representation
        """
        embedding = self.embed(text)
        
        # Convert COO to scipy sparse matrix (flattened to 1D for similarity)
        indices = embedding['indices'].numpy()
        values = embedding['values'].numpy()
        shape = embedding['shape']
        
        # Flatten 2D indices to 1D
        flat_indices = indices[:, 0] * shape[1] + indices[:, 1]
        flat_size = shape[0] * shape[1]
        
        # Create COO matrix then convert to CSR
        coo = coo_matrix((values, (np.zeros(len(values), dtype=int), flat_indices)), 
                         shape=(1, flat_size))
        return coo.tocsr()

    def _text_to_coeffs(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize text and run it through the LM + wavelet stack to obtain sparse coeffs."""
        tokens = self.lm_tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.seq_len,
        )
        input_ids = tokens.input_ids.to(DEVICE)
        attention_mask = tokens.attention_mask.to(DEVICE)
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            h_orig = outputs.hidden_states[-1]
            # Get hard-masked sparse coefficients
            _, _, coeffs_sparse, mask_kept, _ = self.model(h_orig, training=False)
        return coeffs_sparse, mask_kept, input_ids, attention_mask

    def decode(self, embedding: Dict) -> str:
        """
        Decode an embedding back to text.
        
        Key insight: With BERT (bidirectional), h[i] represents token[i] directly.
        We only decode the real tokens (not padding).
        
        Args:
            embedding: Dict containing sparse coefficients in COO format
            
        Returns:
            Reconstructed text
        """
        with torch.no_grad():
            # Reconstruct sparse tensor from COO format
            indices = embedding['indices'].to(DEVICE)
            values = embedding['values'].to(DEVICE)
            shape = embedding['shape']
            num_real_tokens = embedding['num_real_tokens']
            
            coeffs_sparse = torch.zeros((1, shape[0], shape[1]), device=DEVICE, dtype=values.dtype)
            mask_kept = torch.zeros_like(coeffs_sparse, dtype=torch.bool)
            
            for (t, d), val in zip(indices, values):
                coeffs_sparse[0, t, d] = val
                mask_kept[0, t, d] = True
            
            # Run inverse wavelet transform to get h_approx
            h_approx = self.model.wavelet_module._idwt_lifting_1d(coeffs_sparse)
            
            # Refine with learned network (fixes hard-mask artifacts)
            h_refined = self.model.refinement_network(h_approx, mask_kept)
            
            # Get logits using BERT's MLM head
            logits = self.lm_model.cls.predictions(h_refined)

            # Only decode the real tokens, not padding
            predicted_ids = torch.argmax(logits[0, :num_real_tokens], dim=-1).cpu().tolist()
            
            return self.lm_tokenizer.decode(predicted_ids, skip_special_tokens=True)
