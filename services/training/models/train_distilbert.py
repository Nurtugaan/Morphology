"""
DistilBERT Training Script for English Morphological Analysis

Скрипт обучения модели DistilBERT для морфологического анализа.
Использует общие компоненты из training/ модуля.

DistilBERT - облегчённая версия BERT (в 2 раза меньше параметров),
поэтому можно использовать больший batch_size и немного выше learning_rate.

Запуск:
    cd c:\Diploma\morphology
    python -m services.training.models.train_distilbert

С параметрами:
    python -m services.training.models.train_distilbert --epochs 5 --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.train_base import train_model


# ============================================================================
# DISTILBERT-SPECIFIC CONFIGURATION
# ============================================================================

MODEL_NAME = "distilbert-base-uncased"
MODEL_SHORT_NAME = "distilbert"
OUTPUT_DIR = "services/models/en/distilbert"

# Recommended hyperparameters for DistilBERT
# DistilBERT is smaller, so we can use larger batch size and slightly higher LR
DEFAULT_CONFIG = {
    'model_name': MODEL_NAME,
    'output_dir': OUTPUT_DIR,
    'epochs': 6,           # Can train longer due to faster training
    'batch_size': 32,      # Larger batch size due to smaller model
    'learning_rate': 3e-5, # Slightly higher LR for smaller model
    'max_length': 128,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'dropout': 0.1
}


def main():
    """Запуск обучения DistilBERT."""
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
