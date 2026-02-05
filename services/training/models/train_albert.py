"""
ALBERT-base Training Script for English Morphological Analysis

Скрипт обучения модели ALBERT-base для морфологического анализа.
Использует общие компоненты из training/ модуля.

ALBERT использует factorized embeddings и cross-layer parameter sharing,
что делает модель очень экономной по памяти (12M параметров vs 110M у BERT).

Запуск:
    cd c:\Diploma\morphology
    python -m services.training.models.train_albert

С параметрами:
    python -m services.training.models.train_albert --epochs 5 --batch-size 64
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.train_base import train_model


# ============================================================================
# ALBERT-SPECIFIC CONFIGURATION
# ============================================================================

MODEL_NAME = "albert-base-v2"
MODEL_SHORT_NAME = "albert-base"
OUTPUT_DIR = "services/models/en/albert-base"

# Recommended hyperparameters for ALBERT-base
# ALBERT is very memory efficient, so we can use larger batch sizes
DEFAULT_CONFIG = {
    'model_name': MODEL_NAME,
    'output_dir': OUTPUT_DIR,
    'epochs': 6,           # May need more epochs due to parameter sharing
    'batch_size': 64,      # Can use larger batch size (very memory efficient)
    'learning_rate': 2e-5,
    'max_length': 128,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'dropout': 0.1
}


def main():
    """Запуск обучения ALBERT-base."""
    parser = argparse.ArgumentParser(
        description=f'Train {MODEL_NAME} for morphological analysis'
    )
    
    # Training arguments (override defaults)
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--learning-rate', '-lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--max-length', type=int, default=DEFAULT_CONFIG['max_length'])
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_CONFIG['output_dir'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    
    # Data arguments
    parser.add_argument('--ewt-dir', type=str, default='services/datasets/english/UD_English-EWT')
    parser.add_argument('--gum-dir', type=str, default='services/datasets/english/UD_English-GUM')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print(f"  Training: {MODEL_NAME}")
    print(f"  Output:   {args.output_dir}")
    print("=" * 60)
    
    # Train model
    results = train_model(
        model_name=MODEL_NAME,
        ewt_dir=args.ewt_dir,
        gum_dir=args.gum_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        device=args.device
    )
    
    print(f"\n✓ {MODEL_SHORT_NAME} training complete!")
    print(f"  Best F1: {results['best_f1']:.4f}")
    print(f"  Model saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
