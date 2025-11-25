"""Test whether sparse wavelet coefficients preserve semantic similarity for retrieval."""

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, AutoTokenizer
from scipy.stats import mannwhitneyu
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sparse_cosine_similarity
import numpy as np

from wavgpt.utils.save_checkpoint import load_checkpoint_for_inference
from wavgpt.config import DEVICE, MODEL_NAME, BLOCK_SIZE
from wavgpt.embed.embeddings import WavGPTEmbedder
from tests.similarity_utils import comprehensive_similarity


def get_sparse_embedding(embedder: WavGPTEmbedder, text: str):
    """
    Get sparse wavelet coefficient embedding in CSR format for efficient similarity computation.
    
    Returns:
        embedding_csr: Sparse matrix in CSR format (1, D)
        num_nonzero: Number of nonzero coefficients
        num_total: Total possible coefficients
    """
    # Get embedding in CSR format (efficient for similarity)
    embedding_csr = embedder.embed_to_csr(text)
    
    # Get sparsity statistics from original COO format
    embedding_coo = embedder.embed(text)
    num_nonzero = len(embedding_coo['values'])
    num_total = embedding_coo['shape'][0] * embedding_coo['shape'][1]
    
    return embedding_csr, num_nonzero, num_total


def test_coefficient_similarities():
    """
    Comprehensive test of ALL similarity metrics for semantic similarity preservation.
    
    Tests 8 different similarity methods:
    1. BERT CLS (baseline)
    2. BERT Mean Pool (baseline)
    3. Raw Sparse Coefficients
    4. Decoded CLS
    5. Decoded Mean Pool
    6. Frequency Band Overlap
    7. Frequency Energy Correlation
    8. Frequency Energy Cosine
    """
    
    print("="*80)
    print("COMPREHENSIVE Semantic Similarity Test")
    print("="*80)
    print("\nTesting 8 different similarity metrics to find what works best!")
    print()
    
    # Load model
    checkpoint_path = '/home/ubuntu/WavGPT/checkpoints/hybrid_wavelet_model_ratio0.01_step37000.pt'
    print(f"Loading checkpoint: {checkpoint_path}")
    model, info = load_checkpoint_for_inference(checkpoint_path, device=DEVICE)
    
    # Load BERT
    print("Loading BERT model and tokenizer...")
    lm_model = BertForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lm_model.eval()
    
    # Create embedder
    embedder = WavGPTEmbedder(model, lm_model, tokenizer)
    
    # Define test cases: (text1, text2, expected_relationship)
    test_pairs = [
        # SIMILAR pairs (30+)
        ("The cat sat on the mat.", "A feline rested on the rug.", "SIMILAR"),
        ("I love programming in Python.", "Python is my favorite coding language.", "SIMILAR"),
        ("The weather is beautiful today.", "It's a gorgeous sunny day outside.", "SIMILAR"),
        ("She walked to the store yesterday.", "Yesterday she went to the shop on foot.", "SIMILAR"),
        ("The dog barked loudly at night.", "A canine was making loud noises after dark.", "SIMILAR"),
        ("He enjoys reading mystery novels.", "Mystery books are his favorite thing to read.", "SIMILAR"),
        ("The river flows through the valley.", "Water runs through the vale.", "SIMILAR"),
        ("Children played in the park happily.", "Kids were joyfully playing at the playground.", "SIMILAR"),
        ("The mountain peak is covered in snow.", "Snow blankets the summit of the mountain.", "SIMILAR"),
        ("She drives a red sports car.", "Her vehicle is a crimson racing automobile.", "SIMILAR"),
        ("Coffee tastes better in the morning.", "Morning coffee is the most delicious.", "SIMILAR"),
        ("Students study hard for exams.", "Pupils work diligently preparing for tests.", "SIMILAR"),
        ("The ocean waves crashed on the shore.", "Surf pounded against the beach.", "SIMILAR"),
        ("Birds sing at dawn each day.", "Avian creatures vocalize at sunrise daily.", "SIMILAR"),
        ("The pizza was delivered late.", "Our pizza order arrived behind schedule.", "SIMILAR"),
        ("Roses bloom in the garden.", "The garden has blooming rose flowers.", "SIMILAR"),
        ("He works as a software engineer.", "His profession is software engineering.", "SIMILAR"),
        ("The concert was incredibly loud.", "That musical performance was extremely noisy.", "SIMILAR"),
        ("She teaches mathematics at university.", "University mathematics is what she teaches.", "SIMILAR"),
        ("The train arrives at noon.", "The railway comes at midday.", "SIMILAR"),
        ("Clouds covered the entire sky.", "The whole sky was filled with clouds.", "SIMILAR"),
        ("He solved the puzzle quickly.", "The puzzle was rapidly solved by him.", "SIMILAR"),
        ("Flowers need water to grow.", "Plants require hydration for growth.", "SIMILAR"),
        ("The library has many books.", "Numerous books are available at the library.", "SIMILAR"),
        ("They traveled across Europe.", "Their journey went through European countries.", "SIMILAR"),
        ("The phone rang three times.", "Three rings came from the telephone.", "SIMILAR"),
        ("Ice cream melts in the sun.", "Solar heat causes ice cream to liquefy.", "SIMILAR"),
        ("She painted a beautiful landscape.", "A gorgeous scenery was painted by her.", "SIMILAR"),
        ("Traffic was heavy this morning.", "This morning saw substantial vehicular congestion.", "SIMILAR"),
        ("The baby slept peacefully.", "The infant was sleeping calmly.", "SIMILAR"),
        ("He climbed the steep hill.", "The precipitous slope was ascended by him.", "SIMILAR"),
        ("Stars shine brightly at night.", "Celestial bodies emit light after dark.", "SIMILAR"),
        ("The recipe requires three eggs.", "Three eggs are needed for this recipe.", "SIMILAR"),
        ("She won the swimming competition.", "The swimming contest was won by her.", "SIMILAR"),
        ("Trees provide shade in summer.", "Summer shade comes from trees.", "SIMILAR"),
        
        # DISSIMILAR pairs (30+)
        ("The cat sat on the mat.", "Quantum mechanics explains particle behavior.", "DISSIMILAR"),
        ("I love programming in Python.", "The pizza was delivered late yesterday.", "DISSIMILAR"),
        ("The weather is beautiful today.", "Machine learning models need training data.", "DISSIMILAR"),
        ("She walked to the store yesterday.", "Nuclear fusion powers the sun.", "DISSIMILAR"),
        ("The dog barked loudly at night.", "Democracy is a form of government.", "DISSIMILAR"),
        ("He enjoys reading mystery novels.", "Photosynthesis converts light to energy.", "DISSIMILAR"),
        ("The river flows through the valley.", "Shakespeare wrote many famous plays.", "DISSIMILAR"),
        ("Children played in the park happily.", "Economics studies resource allocation.", "DISSIMILAR"),
        ("The mountain peak is covered in snow.", "DNA contains genetic information.", "DISSIMILAR"),
        ("She drives a red sports car.", "Ancient Rome had powerful emperors.", "DISSIMILAR"),
        ("Coffee tastes better in the morning.", "Gravity keeps planets in orbit.", "DISSIMILAR"),
        ("Students study hard for exams.", "The ocean contains millions of species.", "DISSIMILAR"),
        ("The ocean waves crashed on the shore.", "Mathematics is the language of science.", "DISSIMILAR"),
        ("Birds sing at dawn each day.", "Global warming affects climate patterns.", "DISSIMILAR"),
        ("The pizza was delivered late.", "Mozart composed symphonies in Vienna.", "DISSIMILAR"),
        ("Roses bloom in the garden.", "Computers process binary instructions.", "DISSIMILAR"),
        ("He works as a software engineer.", "Dinosaurs went extinct millions ago.", "DISSIMILAR"),
        ("The concert was incredibly loud.", "Antibiotics fight bacterial infections.", "DISSIMILAR"),
        ("She teaches mathematics at university.", "The Pacific Ocean is very deep.", "DISSIMILAR"),
        ("The train arrives at noon.", "Picasso pioneered cubist painting.", "DISSIMILAR"),
        ("Clouds covered the entire sky.", "Volcanoes erupt molten rock.", "DISSIMILAR"),
        ("He solved the puzzle quickly.", "Renaissance art flourished in Italy.", "DISSIMILAR"),
        ("Flowers need water to grow.", "Satellites orbit around Earth.", "DISSIMILAR"),
        ("The library has many books.", "Chess requires strategic thinking.", "DISSIMILAR"),
        ("They traveled across Europe.", "Cells divide through mitosis.", "DISSIMILAR"),
        ("The phone rang three times.", "Einstein developed relativity theory.", "DISSIMILAR"),
        ("Ice cream melts in the sun.", "Parliament makes legislative decisions.", "DISSIMILAR"),
        ("She painted a beautiful landscape.", "Earthquakes occur at fault lines.", "DISSIMILAR"),
        ("Traffic was heavy this morning.", "Poetry expresses emotional themes.", "DISSIMILAR"),
        ("The baby slept peacefully.", "Vaccines prevent infectious diseases.", "DISSIMILAR"),
        ("He climbed the steep hill.", "Fibonacci sequence appears in nature.", "DISSIMILAR"),
        ("Stars shine brightly at night.", "Constitution defines legal rights.", "DISSIMILAR"),
        ("The recipe requires three eggs.", "Galaxies contain billions of stars.", "DISSIMILAR"),
        ("She won the swimming competition.", "Philosophy explores fundamental questions.", "DISSIMILAR"),
        ("Trees provide shade in summer.", "Metamorphosis transforms caterpillars.", "DISSIMILAR"),
    ]
    
    print(f"\nRunning {len(test_pairs)} similarity tests...\n")
    
    results = []
    
    for i, (text1, text2, relationship) in enumerate(test_pairs, 1):
        print(f"Test {i}/{len(test_pairs)}: {relationship}")
        print(f"  '{text1[:50]}...'")
        print(f"  '{text2[:50]}...'")
        
        # Get ALL similarity metrics at once using test utils
        metrics = comprehensive_similarity(embedder, text1, text2)
        
        # Print just the key metrics for readability
        print(f"  BERT CLS: {metrics['bert_cls']:.3f} | "
              f"Decoded CLS: {metrics['decoded_cls']:.3f} | "
              f"Sparse Coeff: {metrics['sparse_coeff']:.3f}")
        print()
        
        results.append({
            'relationship': relationship,
            'text1': text1[:35] + "...",
            'text2': text2[:35] + "...",
            **metrics  # Unpack all metrics
        })
    
    # Analyze results
    print("="*80)
    print("Statistical Analysis")
    print("="*80)
    
    # Analyze ALL methods
    print("="*80)
    print("STATISTICAL ANALYSIS - ALL METHODS")
    print("="*80)
    
    methods = [
        ('BERT CLS (Baseline)', 'bert_cls'),
        ('BERT Mean Pool (Baseline)', 'bert_mean_pool'),
        ('Sparse Coefficients', 'sparse_coeff'),
        ('Decoded CLS', 'decoded_cls'),
        ('Decoded Mean Pool', 'decoded_mean_pool'),
        ('Frequency Band Overlap', 'freq_band_overlap'),
        ('Frequency Energy Correlation', 'freq_energy_correlation'),
        ('Frequency Energy Cosine', 'freq_energy_cosine'),
    ]
    
    summary_results = []
    
    for method_name, metric_key in methods:
        similar = [r[metric_key] for r in results if r['relationship'] == 'SIMILAR']
        dissimilar = [r[metric_key] for r in results if r['relationship'] == 'DISSIMILAR']
        
        separation = np.mean(similar) - np.mean(dissimilar)
        pooled_std = np.sqrt((np.var(similar) + np.var(dissimilar)) / 2)
        cohens_d = separation / pooled_std if pooled_std > 0 else 0
        
        try:
            statistic, p_value = mannwhitneyu(similar, dissimilar, alternative='greater')
            significant = p_value < 0.05
        except Exception as e:
            p_value = 1.0
            significant = False
        
        summary_results.append({
            'method': method_name,
            'similar_mean': np.mean(similar),
            'dissimilar_mean': np.mean(dissimilar),
            'separation': separation,
            'cohens_d': cohens_d,
            'p_value': p_value,
            'significant': significant,
        })
        
        print(f"\n{method_name}")
        print(f"  Similar:    mean={np.mean(similar):.4f}, std={np.std(similar):.4f}")
        print(f"  Dissimilar: mean={np.mean(dissimilar):.4f}, std={np.std(dissimilar):.4f}")
        print(f"  Separation: {separation:.4f}")
        print(f"  Cohen's d:  {cohens_d:.4f}")
        print(f"  P-value:    {p_value:.6f} {'✓ SIGNIFICANT' if significant else '✗ Not significant'}")
    
    # Summary ranking
    print("\n" + "="*80)
    print("RANKING BY SEPARATION (Higher = Better)")
    print("="*80)
    summary_results.sort(key=lambda x: x['separation'], reverse=True)
    
    cohens_header = "Cohen's d"
    print(f"\n{'Rank':<5} {'Method':<35} {'Separation':<12} {cohens_header:<12} {'Significant':<12}")
    print("-" * 80)
    for rank, result in enumerate(summary_results, 1):
        sig_mark = '✓' if result['significant'] else '✗'
        print(f"{rank:<5} {result['method']:<35} {result['separation']:<12.4f} "
              f"{result['cohens_d']:<12.4f} {sig_mark:<12}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    best = summary_results[0]
    worst = summary_results[-1]
    print(f"\n✓ BEST:  {best['method']}")
    print(f"  - Separation: {best['separation']:.4f}")
    print(f"  - Effect size: {best['cohens_d']:.4f}")
    print(f"  - Statistically significant: {'YES' if best['significant'] else 'NO'}")
    print(f"\n✗ WORST: {worst['method']}")
    print(f"  - Separation: {worst['separation']:.4f}")
    print(f"  - Effect size: {worst['cohens_d']:.4f}")
    
    print("="*80)
    
    # Detailed table
    print("\n\nDETAILED RESULTS (First 20 pairs):")
    print("-" * 150)
    print(f"{'Rel':<4} {'Text 1':<30} {'Text 2':<30} {'BERT':<7} {'Decoded':<7} {'Sparse':<7} {'FreqOvlp':<8} {'FreqCorr':<8}")
    print("-" * 150)
    for r in results[:20]:
        rel = 'SIM' if r['relationship'] == 'SIMILAR' else 'DIS'
        print(f"{rel:<4} {r['text1']:<30} {r['text2']:<30} "
              f"{r['bert_cls']:<7.3f} {r['decoded_cls']:<7.3f} {r['sparse_coeff']:<7.3f} "
              f"{r['freq_band_overlap']:<8.3f} {r['freq_energy_correlation']:<8.3f}")
    print("... (showing first 20 of {} pairs)".format(len(results)))
    print("-" * 150)


if __name__ == "__main__":
    test_coefficient_similarities()

