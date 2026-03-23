#!/usr/bin/env python3
"""
Calculate Model FLOPS Utilization (MFU) for nanoGPT training.

Formula:
    MFU = (6 * N * T) / (P * t_total)

Where:
    N = Number of model parameters
    T = Total tokens processed (max_iters * batch_size * block_size)
    P = Peak theoretical FLOPs of GPU
    t_total = Total training time in seconds

Usage:
    python calculate_mfu.py
    python calculate_mfu.py --time "11m4.744s"
    python calculate_mfu.py --time 664.744 --peak_tflops 15.11
"""

import argparse

def count_parameters(n_layer=4, n_head=4, n_embd=256, block_size=20, vocab_size=16):
    """
    Count parameters for GPT model.
    
    GPT architecture:
    - Token embeddings: vocab_size * n_embd
    - Position embeddings: block_size * n_embd
    - Per transformer layer:
        - LayerNorm1: 2 * n_embd
        - Attention Q,K,V: 3 * n_embd * n_embd + 3 * n_embd
        - Attention output: n_embd * n_embd + n_embd
        - LayerNorm2: 2 * n_embd
        - MLP up: n_embd * 4*n_embd + 4*n_embd
        - MLP down: 4*n_embd * n_embd + n_embd
    - Final LayerNorm: 2 * n_embd
    - LM head: tied with token embeddings (0 extra)
    """
    token_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    
    per_layer = (
        2 * n_embd +                          # ln_1
        3 * n_embd * n_embd + 3 * n_embd +    # c_attn (Q, K, V)
        n_embd * n_embd + n_embd +            # c_proj
        2 * n_embd +                          # ln_2
        n_embd * 4 * n_embd + 4 * n_embd +    # mlp.c_fc
        4 * n_embd * n_embd + n_embd          # mlp.c_proj
    )
    
    final_ln = 2 * n_embd
    
    total = token_emb + pos_emb + (per_layer * n_layer) + final_ln
    return total


def parse_time(time_str):
    """Parse time string like '11m4.744s' or '664.744' to seconds."""
    time_str = str(time_str).strip()
    
    if 'm' in time_str:
        # Format: 11m4.744s
        time_str = time_str.replace('s', '')
        parts = time_str.split('m')
        minutes = float(parts[0])
        seconds = float(parts[1]) if parts[1] else 0
        return minutes * 60 + seconds
    else:
        # Already in seconds
        return float(time_str)


def calculate_mfu(time_seconds, peak_tflops, max_iters=4500, batch_size=4096, 
                  block_size=20, n_layer=4, n_head=4, n_embd=256, vocab_size=16):
    """
    Calculate MFU using the formula: MFU = (6 * N * T) / (P * t_total)
    """
    # N: Model parameters
    N = count_parameters(n_layer, n_head, n_embd, block_size, vocab_size)
    
    # T: Total tokens processed
    T = max_iters * batch_size * block_size
    
    # P: Peak FLOPS
    P = peak_tflops * 1e12
    
    # t_total: Training time in seconds
    t_total = time_seconds
    
    # MFU calculation
    # Factor of 6: forward pass (2N FLOPs) + backward pass (4N FLOPs)
    numerator = 6 * N * T
    denominator = P * t_total
    mfu = numerator / denominator
    
    # Additional metrics
    achieved_tflops = numerator / t_total / 1e12
    tokens_per_sec = T / t_total
    ms_per_iter = (t_total / max_iters) * 1000
    
    return {
        'mfu': mfu,
        'mfu_percent': mfu * 100,
        'N': N,
        'T': T,
        'P': P,
        't_total': t_total,
        'achieved_tflops': achieved_tflops,
        'tokens_per_sec': tokens_per_sec,
        'ms_per_iter': ms_per_iter,
        'config': {
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'block_size': block_size,
            'vocab_size': vocab_size,
            'max_iters': max_iters,
            'batch_size': batch_size,
        }
    }


def print_results(results):
    """Print formatted MFU results."""
    config = results['config']
    
    print("=" * 60)
    print("  MFU Calculator for 4-Digit Addition Training")
    print("=" * 60)
    
    print(f"\n{'Model Configuration':-^60}")
    print(f"  n_layer:    {config['n_layer']}")
    print(f"  n_head:     {config['n_head']}")
    print(f"  n_embd:     {config['n_embd']}")
    print(f"  block_size: {config['block_size']}")
    print(f"  vocab_size: {config['vocab_size']}")
    
    print(f"\n{'Parameters (N)':-^60}")
    print(f"  Total:      {results['N']:,} parameters")
    print(f"  Millions:   {results['N'] / 1e6:.3f}M")
    
    print(f"\n{'Training Configuration':-^60}")
    print(f"  max_iters:  {config['max_iters']:,}")
    print(f"  batch_size: {config['batch_size']}")
    print(f"  block_size: {config['block_size']}")
    
    print(f"\n{'Tokens Processed (T)':-^60}")
    print(f"  Per iteration: {config['batch_size'] * config['block_size']:,}")
    print(f"  Total (T):     {results['T']:,}")
    print(f"  Millions:      {results['T'] / 1e6:.2f}M")
    
    print(f"\n{'Hardware & Timing':-^60}")
    print(f"  GPU Peak (P):  {results['P'] / 1e12:.2f} TFLOPS")
    print(f"  Time (t):      {results['t_total']:.3f}s ({results['t_total']/60:.2f} min)")
    
    print(f"\n{'MFU Calculation':-^60}")
    print(f"  Formula: MFU = (6 × N × T) / (P × t_total)")
    print(f"")
    print(f"  Numerator:   6 × {results['N']:,} × {results['T']:,}")
    print(f"             = {6 * results['N'] * results['T']:.4e}")
    print(f"")
    print(f"  Denominator: {results['P']:.4e} × {results['t_total']:.3f}")
    print(f"             = {results['P'] * results['t_total']:.4e}")
    
    print(f"\n{'═' * 60}")
    print(f"  ███  MFU = {results['mfu_percent']:.4f}%  ███")
    print(f"{'═' * 60}")
    
    print(f"\n{'Performance Metrics':-^60}")
    print(f"  Achieved:      {results['achieved_tflops']:.4f} TFLOPS")
    print(f"  Throughput:    {results['tokens_per_sec']:,.0f} tokens/sec")
    print(f"  Per iteration: {results['ms_per_iter']:.2f} ms")
    
    print(f"\n{'Interpretation':-^60}")
    mfu_pct = results['mfu_percent']
    if mfu_pct < 5:
        print("  ⚠️  MFU < 5%: Very low utilization")
        print("      - Model is too small (memory-bound)")
        print("      - Overhead dominates compute time")
    elif mfu_pct < 15:
        print("  📊 MFU 5-15%: Low but expected for small models")
        print("      - Small models can't saturate GPU compute")
        print("      - Consider larger batch size if memory allows")
    elif mfu_pct < 30:
        print("  ✓  MFU 15-30%: Reasonable utilization")
    elif mfu_pct < 50:
        print("  ✓✓ MFU 30-50%: Good utilization!")
    else:
        print("  🚀 MFU > 50%: Excellent utilization!")
    
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate MFU for nanoGPT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calculate_mfu.py
  python calculate_mfu.py --time "11m4.744s"
  python calculate_mfu.py --time 664.744 --peak_tflops 15.11
  python calculate_mfu.py --time "3m0s" --peak_tflops 15.11
        """
    )
    parser.add_argument("--time", type=str, default="11m4.744s",
                        help="Training time (e.g., '11m4.744s' or '664.744')")
    parser.add_argument("--peak_tflops", type=float, default=15.11,
                        help="GPU peak TFLOPS (default: 15.11 for RTX 4060 Laptop)")
    parser.add_argument("--max_iters", type=int, default=4500,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size")
    parser.add_argument("--block_size", type=int, default=20,
                        help="Sequence length / block size")
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=16,
                        help="Vocabulary size")
    
    args = parser.parse_args()
    
    # Parse time
    time_seconds = parse_time(args.time)
    
    # Calculate MFU
    results = calculate_mfu(
        time_seconds=time_seconds,
        peak_tflops=args.peak_tflops,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        vocab_size=args.vocab_size,
    )
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()