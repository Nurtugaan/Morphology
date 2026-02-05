"""
Base Training Module for English Morphological Analysis (TASK-004)

Общий модуль обучения для всех Transformer-моделей.
Содержит функцию train_model(), которая используется
model-specific скриптами (train_bert.py, train_roberta.py и др.)

Не запускайте этот файл напрямую!
Используйте скрипты из services/training/models/:
    python -m services.training.models.train_bert
    python -m services.training.models.train_roberta
    python -m services.training.models.train_distilbert
    python -m services.training.models.train_albert
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.en_preprocessing import (
    load_ud_dataset,
    build_label_vocab,
    combine_datasets
)
from training.config import TrainingConfig, get_config
from training.dataset import TokenClassificationDataset, create_dataloaders
from training.model import MorphologyTagger, create_model
from training.trainer import Trainer, set_seed


def load_data(
    ewt_dir: str = "services/datasets/english/UD_English-EWT",
    gum_dir: Optional[str] = None,
    combine: bool = False
) -> Tuple:
    """
    Загружает и подготавливает данные для обучения.
    
    Args:
        ewt_dir: Путь к UD English EWT
        gum_dir: Путь к UD English GUM (опционально)
        combine: Объединять датасеты
        
    Returns:
        Tuple (dataset, label2id, id2label)
    """
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load EWT dataset
    print(f"\nLoading UD English EWT from {ewt_dir}...")
    ewt_dataset = load_ud_dataset(ewt_dir)
    print(ewt_dataset.stats())
    
    # Optionally load GUM dataset
    if gum_dir and Path(gum_dir).exists():
        print(f"\nLoading UD English GUM from {gum_dir}...")
        gum_dataset = load_ud_dataset(gum_dir)
        print(gum_dataset.stats())
        
        if combine:
            print("\nCombining datasets...")
            dataset = combine_datasets([ewt_dataset, gum_dataset], name="EWT+GUM")
        else:
            dataset = ewt_dataset
    else:
        dataset = ewt_dataset
    
    # Build label vocabulary
    print("\nBuilding label vocabulary...")
    label2id, id2label = build_label_vocab(dataset.train)
    print(f"  Total unique labels: {len(label2id)}")
    
    return dataset, label2id, id2label


def train_model(
    model_name: str = "bert-base-uncased",
    ewt_dir: str = "services/datasets/english/UD_English-EWT",
    gum_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    seed: int = 42,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Основная функция обучения модели.
    
    Args:
        model_name: Название модели из HuggingFace
        ewt_dir: Путь к датасету EWT
        gum_dir: Путь к датасету GUM (опционально)
        output_dir: Директория для сохранения модели
        epochs: Количество эпох
        batch_size: Размер батча
        learning_rate: Скорость обучения
        max_length: Максимальная длина последовательности
        seed: Random seed
        device: Устройство ('cuda' или 'cpu')
        
    Returns:
        Dict с результатами обучения
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    dataset, label2id, id2label = load_data(ewt_dir, gum_dir)
    
    # Create config
    config = get_config(model_name)
    config.epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.max_length = max_length
    config.seed = seed
    config.device = device
    
    if output_dir:
        config.output_dir = output_dir
    
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets and dataloaders
    print("\nCreating datasets...")
    train_loader, val_loader = create_dataloaders(
        train_sentences=dataset.train,
        val_sentences=dataset.dev,
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        max_length=max_length
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create test loader if available
    test_loader = None
    if dataset.test:
        from torch.utils.data import DataLoader
        test_dataset = TokenClassificationDataset(
            sentences=dataset.test,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=max_length
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_name=model_name,
        num_labels=len(label2id),
        dropout=config.dropout,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        label2id=label2id,
        id2label=id2label,
        test_loader=test_loader
    )
    
    # Train model
    results = trainer.train()
    
    # Evaluate on test set if available
    if test_loader:
        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(test_loader)
        results['test_metrics'] = test_results
    
    return results


def main():
    """Главная функция с аргументами командной строки."""
    parser = argparse.ArgumentParser(
        description='Train morphological analysis models on English data'
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='bert-base-uncased',
        help='Model name from HuggingFace (default: bert-base-uncased)'
    )
    
    # Data arguments
    parser.add_argument(
        '--ewt-dir',
        type=str,
        default='services/datasets/english/UD_English-EWT',
        help='Path to UD English EWT dataset'
    )
    parser.add_argument(
        '--gum-dir',
        type=str,
        default='services/datasets/english/UD_English-GUM',
        help='Path to UD English GUM dataset (optional)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=128,
        help='Maximum sequence length (default: 128)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for saving model'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("  English Morphological Analysis - Model Training")
    print("  TASK-004-TRAIN-MODELS-EN")
    print("=" * 60)
    
    # Train model
    results = train_model(
        model_name=args.model,
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
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\nBest F1 Score: {results['best_f1']:.4f}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Total Time: {results['total_time'] / 60:.1f} minutes")
    
    return results


if __name__ == "__main__":
    main()
