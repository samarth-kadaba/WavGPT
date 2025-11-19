"""Interface for generating embeddings from a trained WavGPT model."""

import math
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, BertForMaskedLM
import numpy as np
from wavgpt.models import HybridWaveletRefinementModel
from wavgpt.config import HIDDEN_SIZE, BLOCK_SIZE, DEVICE, MODEL_NAME, KEEP_RATIO, REFINE_N_LAYERS, REFINE_N_HEADS, REFINE_DIM_FEEDFORWARD, USE_TEMPORAL_ATTENTION, WAVELET_LEVELS
from wavgpt.utils.load_checkpoint import load_model_from_checkpoint

class WavGPTEmbedder:
    def __init__(self, model: HybridWaveletRefinementModel, lm_model: BertForMaskedLM, lm_tokenizer: AutoTokenizer):
        """Initialize embedder with pre-loaded models.
        
        Args:
            model: Pre-loaded and configured WavGPT model (use load_model_from_checkpoint)
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
        self.max_sparse = HIDDEN_SIZE  # maximum number of non-zero coeffs we expect

        # Build structured Johnson-Lindenstrauss projections for compressed sensing.
        # We target an embedding of size 2 * HIDDEN_SIZE = 2560 entries, factorized
        # into (m_t x m_d) = (64 x 40) to exploit the (T x d) tensor structure.
        self.embed_rows_temporal = 64
        self.embed_rows_channel = (2 * HIDDEN_SIZE) // self.embed_rows_temporal
        if self.embed_rows_temporal * self.embed_rows_channel != 2 * HIDDEN_SIZE:
            raise ValueError("2 * HIDDEN_SIZE must factor into the chosen projection dimensions.")
        self.measurement_temporal, self.measurement_channel = self._init_measurement_operators()

    def embed(self, text: str) -> Tuple[torch.Tensor, int, int]:
        """
        Embed text into compressed representation.
        
        Returns:
            embedding: (2 * HIDDEN_SIZE,) compressed coefficients
            first_token_id: ID of the first token (needed for reconstruction)
            num_real_tokens: Number of real (non-pad) tokens
        """
        coeffs_sparse, mask_kept, input_ids, attention_mask = self._text_to_coeffs(text)
        embedding = self.emb_encode(coeffs_sparse, mask_kept)
        return embedding
    
    def coalesce_coeffs_and_mask_kept(self, coeffs_sparse: torch.Tensor, mask_kept: torch.Tensor):
        """Place nonzero coeffficents in a vector of length HIDDEN_SIZE"""
        nonzero_indices = np.where(mask_kept.cpu().numpy())
        non_zero_coeffs = coeffs_sparse[:, nonzero_indices[0], nonzero_indices[1]]
        return non_zero_coeffs, nonzero_indices

    def _init_measurement_operators(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create separable sensing matrices for the temporal and hidden dimensions."""
        gen_t = torch.Generator(device="cpu")
        gen_t.manual_seed(0)
        gen_d = torch.Generator(device="cpu")
        gen_d.manual_seed(1)
        A_t = torch.randn(self.embed_rows_temporal, self.seq_len, generator=gen_t) / math.sqrt(self.embed_rows_temporal)
        A_d = torch.randn(self.embed_rows_channel, self.hidden_size, generator=gen_d) / math.sqrt(self.embed_rows_channel)
        return A_t, A_d

    def emb_encode(self, coeffs_sparse: torch.Tensor, mask_kept: torch.Tensor = None) -> torch.Tensor:
        """
        Encode sparse coefficient tensors into a fixed-width embedding using separable compressed sensing.

        Args:
            coeffs_sparse: (B, T, d) sparse coefficients (zeros outside the kept mask)
            mask_kept: optional (B, T, d) boolean mask; unused but kept for interface compatibility

        Returns:
            embeddings: (B, 2 * HIDDEN_SIZE) compressed measurements
        """
        if coeffs_sparse.dim() != 3:
            raise ValueError("coeffs_sparse must be (B, T, d)")
        B, T, d = coeffs_sparse.shape
        if T != self.seq_len or d != self.hidden_size:
            raise ValueError(f"Expected coeffs of shape (_, {self.seq_len}, {self.hidden_size}), got {coeffs_sparse.shape}")

        A_t = self.measurement_temporal.to(coeffs_sparse.device, dtype=coeffs_sparse.dtype)
        A_d = self.measurement_channel.to(coeffs_sparse.device, dtype=coeffs_sparse.dtype)

        temp_proj = torch.einsum("mt,btd->bmd", A_t, coeffs_sparse)
        compressed = torch.einsum("bmd,nd->bmn", temp_proj, A_d)
        return compressed.reshape(B, -1)

    def emb_decode(
        self,
        embeddings: torch.Tensor,
        sparsity_level: int = None,
        tol: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct coeffs_sparse and mask_kept from compressed embeddings via Kronecker-OMP.

        Args:
            embeddings: (B, 2 * HIDDEN_SIZE) compressed measurements
            sparsity_level: maximum number of non-zero coefficients assumed (defaults to HIDDEN_SIZE)
            tol: residual tolerance to stop the pursuit

        Returns:
            coeffs_sparse: (B, T, d) reconstructed sparse coefficients
            mask_kept: (B, T, d) boolean mask of recovered support
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be (B, 2 * HIDDEN_SIZE)")
        B, M = embeddings.shape
        expected_width = 2 * self.hidden_size
        if M != expected_width:
            raise ValueError(f"Expected embedding width {expected_width}, got {M}")
        sparsity = sparsity_level or self.max_sparse

        A_t = self.measurement_temporal.to(embeddings.device, dtype=embeddings.dtype)
        A_d = self.measurement_channel.to(embeddings.device, dtype=embeddings.dtype)
        mt, md = A_t.shape[0], A_d.shape[0]

        coeffs_recon = torch.zeros(B, self.seq_len, self.hidden_size, device=embeddings.device, dtype=embeddings.dtype)
        masks = torch.zeros_like(coeffs_recon, dtype=torch.bool)
        for b in range(B):
            meas_matrix = embeddings[b].reshape(mt, md)
            coeffs_b, mask_b = self._kronecker_omp(meas_matrix, A_t, A_d, sparsity, tol)
            coeffs_recon[b] = coeffs_b
            masks[b] = mask_b
        return coeffs_recon, masks

    def _kronecker_omp(
        self,
        measurement: torch.Tensor,
        A_t: torch.Tensor,
        A_d: torch.Tensor,
        sparsity_level: int,
        tol: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Orthogonal Matching Pursuit specialized to separable sensing operators.
        """
        mt, md = measurement.shape
        T = A_t.shape[1]
        d = A_d.shape[1]
        device = measurement.device
        dtype = measurement.dtype

        y = measurement.reshape(-1)
        residual = y.clone()
        max_iters = min(int(sparsity_level), T * d)
        if max_iters <= 0:
            raise ValueError("sparsity_level must be positive.")

        Phi = torch.zeros(mt * md, max_iters, device=device, dtype=dtype)
        support: List[Tuple[int, int]] = []
        active_cols = 0

        for _ in range(max_iters):
            residual_matrix = residual.reshape(mt, md)
            corr = torch.matmul(torch.matmul(A_t.t(), residual_matrix), A_d)
            corr_abs = corr.abs()
            max_idx = torch.argmax(corr_abs)
            t_idx = (max_idx // d).item()
            d_idx = (max_idx % d).item()

            if corr_abs[t_idx, d_idx] <= tol or (t_idx, d_idx) in support:
                break

            support.append((t_idx, d_idx))
            a_t = A_t[:, t_idx]
            a_d = A_d[:, d_idx]
            atom = torch.outer(a_t, a_d).reshape(-1)
            Phi[:, active_cols] = atom
            active_cols += 1

            Phi_active = Phi[:, :active_cols]
            # MPS doesn't support linalg.lstsq; move to CPU for the solve
            ls_solution = torch.linalg.lstsq(Phi_active.cpu(), y.cpu()).solution.to(device)
            residual = y - Phi_active @ ls_solution
            if torch.linalg.norm(residual) <= tol:
                break

        coeffs = torch.zeros(T, d, device=device, dtype=dtype)
        mask = torch.zeros_like(coeffs, dtype=torch.bool)
        if active_cols == 0:
            return coeffs, mask

        final_solution = torch.linalg.lstsq(Phi[:, :active_cols].cpu(), y.cpu()).solution.to(device)
        for idx, (t_idx, d_idx) in enumerate(support):
            coeffs[t_idx, d_idx] = final_solution[idx]
            mask[t_idx, d_idx] = True

        return coeffs, mask

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
            h_orig = outputs.hidden_states[-1]  # No masking - match training distribution
            _, _, coeffs_sparse, mask_kept, _ = self.model(h_orig, training=False)
        return coeffs_sparse, mask_kept, input_ids, attention_mask

    def verify_text_roundtrip(
        self,
        text: str,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        sparsity_level: int = None,
    ) -> Tuple[bool, str, dict]:
        """
        Full round-trip test: text → embed → decode → text.
        
        Returns:
            coeffs_match: Whether coefficients were preserved through embedding
            reconstructed_text: The decoded text
            metrics: Reconstruction quality metrics
        """
        coeffs_sparse, mask_kept, input_ids, attention_mask = self._text_to_coeffs(text)
        first_token_id = input_ids[0, 0].item()
        num_real_tokens = attention_mask.sum().item()
        
        # Encode and decode through compressed embedding
        embedding = self.emb_encode(coeffs_sparse, mask_kept)
        target_sparsity = sparsity_level or int(mask_kept.sum().item())
        coeffs_recon, mask_recon = self.emb_decode(embedding, sparsity_level=target_sparsity)
        
        # Check coefficient preservation
        coeffs_match = torch.allclose(coeffs_sparse, coeffs_recon, rtol=rtol, atol=atol)
        mask_match = torch.equal(mask_kept.bool(), mask_recon.bool())
        
        coeffs_preserved = coeffs_match and mask_match
        if not coeffs_preserved:
            print("Coeffs difference:", torch.mean((coeffs_sparse - coeffs_recon) ** 2))
            print("Mask difference:", torch.mean((mask_kept.bool() ^ mask_recon.bool()).float()))
        
        # Reconstruct text from embedding
        reconstructed_text = self.decode(embedding, first_token_id, num_real_tokens)
        
        # Calculate reconstruction metrics
        # Get original tokens
        original_ids = input_ids[0, :num_real_tokens].cpu().tolist()
        reconstructed_ids_full = self.lm_tokenizer.encode(reconstructed_text, add_special_tokens=False)[:num_real_tokens]
        
        # Pad if needed
        if len(reconstructed_ids_full) < num_real_tokens:
            reconstructed_ids_full += [self.lm_tokenizer.pad_token_id] * (num_real_tokens - len(reconstructed_ids_full))
        
        matches = sum(1 for o, r in zip(original_ids, reconstructed_ids_full) if o == r)
        token_accuracy = matches / num_real_tokens if num_real_tokens > 0 else 0
        
        metrics = {
            'coeffs_preserved': coeffs_preserved,
            'token_accuracy': token_accuracy,
            'matching_tokens': matches,
            'total_tokens': num_real_tokens,
        }
        
        return coeffs_preserved, reconstructed_text, metrics

    def decode(self, embedding: torch.Tensor) -> str:
        """
        Decode an embedding back to text.
        
        Key insight: With BERT (bidirectional), h_refined[i] represents token[i] directly.
        No shifting needed - this is much cleaner than causal LM approach!
        
        Args:
            embedding: Compressed embedding
            
        Returns:
            Reconstructed text
        """
        with torch.no_grad():
            coeffs_sparse, mask_kept = self.emb_decode(embedding)
            # Run inverse wavelet transform to get h_approx
            h_approx = self.model.wavelet_module._idwt_lifting_1d(coeffs_sparse)
            # Refine in hidden space
            h_refined = self.model.refinement_network(h_approx, mask_kept)
            # Get logits using BERT's MLM head
            logits_refined = self.lm_model.cls.predictions(h_refined)
            print(logits_refined.shape)

            return self.lm_tokenizer.decode(torch.argmax(logits_refined[0], dim=-1).cpu().tolist(), skip_special_tokens=True)


if __name__ == "__main__":
    print("Loading models...")
    model, config = load_model_from_checkpoint("/Users/samkadaba/Desktop/WavGPT/hybrid_wavelet_model_ratio0.00390625.pt")
    lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    lm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = WavGPTEmbedder(model, lm_model, lm_tokenizer)
    
    print("\n" + "="*80)
    print("Testing BERT-Based Reconstruction (Direct h[i]→token[i])")
    print("="*80)
    
    text = "Hello, world! This is a test of the WavGPT model."
    reconstructed = embedding_model.decode(embedding_model.embed(text))
    print(f"Reconstructed: {reconstructed}")
    print(f"Original: {text}")