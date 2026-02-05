"""
Flair Training Script for English Morphological Analysis

Скрипт обучения модели Flair для морфологического анализа.

⚠️ ВАЖНО: Flair использует ДРУГУЮ архитектуру (BiLSTM + Character Embeddings),
НЕ HuggingFace Transformers. Этот скрипт использует библиотеку flair.

Flair отличается:
- Используется BiLSTM вместо Transformer
- Character-level embeddings для морфологии
- Собственный API для обучения
- Другой формат данных (CoNLL)

Установка:
    pip install flair

Запуск:
    cd c:\Diploma\morphology
    python -m services.training.models.train_flair

С параметрами:
    python -m services.training.models.train_flair --epochs 10 --hidden-size 256
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# FLAIR-SPECIFIC CONFIGURATION
# ============================================================================

MODEL_SHORT_NAME = "flair"
OUTPUT_DIR = "services/models/en/flair"

# Flair uses different hyperparameters than Transformers
DEFAULT_CONFIG = {
    'output_dir': OUTPUT_DIR,
    'epochs': 15,              # Flair usually needs more epochs
    'learning_rate': 0.1,      # Flair uses SGD with higher LR
    'mini_batch_size': 32,
    'hidden_size': 256,
    'rnn_layers': 2,
    'use_crf': True,           # CRF layer for sequence labeling
    'embeddings': 'flair',     # 'flair', 'glove', 'bert', or 'stacked'
}


def check_flair_installed():
    """Проверяет установку библиотеки flair."""
    try:
        import flair
        print(f"Flair version: {flair.__version__}")
        return True
    except ImportError:
        print("=" * 60)
        print("ERROR: Flair library not installed!")
        print("=" * 60)
        print("\nInstall with:")
        print("  pip install flair")
        print("\nОr with conda:")
        print("  conda install -c pytorch flair")
        return False


def prepare_flair_data(
    ewt_dir: str,
    output_dir: str
) -> str:
    """
    Подготавливает данные в формате Flair (CoNLL).
    
    Flair ожидает файлы в формате:
    WORD TAG
    WORD TAG
    
    (пустая строка между предложениями)
    
    Args:
        ewt_dir: Путь к UD English EWT
        output_dir: Директория для сохранения
        
    Returns:
        Путь к директории с данными
    """
    from preprocessing.en_preprocessing import load_ud_dataset
    
    print("Preparing data for Flair format...")
    
    dataset = load_ud_dataset(ewt_dir)
    output_path = Path(output_dir) / "flair_data"
    output_path.mkdir(parents=True, exist_ok=True)
    
    def write_flair_file(sentences, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for sent in sentences:
                for token in sent.tokens:
                    f.write(f"{token.form} {token.label}\n")
                f.write("\n")  # Empty line between sentences
    
    write_flair_file(dataset.train, output_path / "train.txt")
    write_flair_file(dataset.dev, output_path / "dev.txt")
    write_flair_file(dataset.test, output_path / "test.txt")
    
    print(f"  Data saved to {output_path}")
    return str(output_path)


def train_flair_model(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    learning_rate: float = 0.1,
    mini_batch_size: int = 32,
    hidden_size: int = 256,
    rnn_layers: int = 2,
    use_crf: bool = True,
    embeddings_type: str = 'flair'
) -> Dict[str, Any]:
    """
    Обучает Flair модель для морфологического анализа.
    
    Args:
        data_dir: Путь к данным в формате Flair
        output_dir: Директория для сохранения модели
        epochs: Количество эпох
        learning_rate: Скорость обучения
        mini_batch_size: Размер мини-батча
        hidden_size: Размер скрытого слоя LSTM
        rnn_layers: Количество слоёв RNN
        use_crf: Использовать CRF слой
        embeddings_type: Тип эмбеддингов
        
    Returns:
        Dict с результатами обучения
    """
    from flair.data import Corpus
    from flair.datasets import ColumnCorpus
    from flair.embeddings import (
        TokenEmbeddings,
        WordEmbeddings,
        StackedEmbeddings,
        FlairEmbeddings,
        CharacterEmbeddings
    )
    from flair.models import SequenceTagger
    from flair.trainers import ModelTrainer
    
    # Define columns in the data file
    columns = {0: 'text', 1: 'pos'}  # WORD TAG
    
    # Load corpus
    print(f"\nLoading corpus from {data_dir}...")
    corpus: Corpus = ColumnCorpus(
        data_folder=data_dir,
        column_format=columns,
        train_file='train.txt',
        dev_file='dev.txt',
        test_file='test.txt'
    )
    
    print(f"  Train: {len(corpus.train)} sentences")
    print(f"  Dev:   {len(corpus.dev)} sentences")
    print(f"  Test:  {len(corpus.test)} sentences")
    
    # Create tag dictionary
    tag_dictionary = corpus.make_label_dictionary(label_type='pos')
    print(f"  Tags:  {len(tag_dictionary)} unique labels")
    
    # Create embeddings based on type
    print(f"\nCreating embeddings ({embeddings_type})...")
    
    embedding_types: List[TokenEmbeddings] = []
    
    if embeddings_type == 'flair':
        embedding_types = [
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
        ]
    elif embeddings_type == 'glove':
        embedding_types = [
            WordEmbeddings('glove'),
        ]
    elif embeddings_type == 'stacked':
        embedding_types = [
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward'),
            FlairEmbeddings('news-backward'),
            CharacterEmbeddings(),
        ]
    else:
        # Default: character embeddings only (lightweight)
        embedding_types = [
            CharacterEmbeddings(),
        ]
    
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    
    # Create sequence tagger
    print("\nCreating SequenceTagger model...")
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=hidden_size,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type='pos',
        use_crf=use_crf,
        rnn_layers=rnn_layers
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in tagger.parameters()):,}")
    
    # Create trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpus)
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    results = trainer.train(
        base_path=output_dir,
        learning_rate=learning_rate,
        mini_batch_size=mini_batch_size,
        max_epochs=epochs,
        checkpoint=True
    )
    
    return {
        'test_score': results['test_score'],
        'dev_score': results['dev_score_history'][-1] if results['dev_score_history'] else 0,
        'model_path': output_dir
    }


def main():
    """Запуск обучения Flair."""
    parser = argparse.ArgumentParser(
        description='Train Flair model for morphological analysis'
    )
    
    # Training arguments
    parser.add_argument('--epochs', '-e', type=int, default=DEFAULT_CONFIG['epochs'])
    parser.add_argument('--learning-rate', '-lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--batch-size', '-b', type=int, default=DEFAULT_CONFIG['mini_batch_size'])
    parser.add_argument('--hidden-size', type=int, default=DEFAULT_CONFIG['hidden_size'])
    parser.add_argument('--rnn-layers', type=int, default=DEFAULT_CONFIG['rnn_layers'])
    parser.add_argument('--no-crf', action='store_true', help='Disable CRF layer')
    parser.add_argument('--embeddings', type=str, default=DEFAULT_CONFIG['embeddings'],
                        choices=['flair', 'glove', 'char', 'stacked'])
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_CONFIG['output_dir'])
    
    # Data arguments
    parser.add_argument('--ewt-dir', type=str, default='services/datasets/english/UD_English-EWT')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "=" * 60)
    print("  Training: Flair (BiLSTM + CRF)")
    print(f"  Output:   {args.output_dir}")
    print("=" * 60)
    
    # Check flair installation
    if not check_flair_installed():
        sys.exit(1)
    
    # Prepare data
    data_dir = prepare_flair_data(args.ewt_dir, args.output_dir)
    
    # Train model
    results = train_flair_model(
        data_dir=data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        rnn_layers=args.rnn_layers,
        use_crf=not args.no_crf,
        embeddings_type=args.embeddings
    )
    
    print(f"\n✓ Flair training complete!")
    print(f"  Test F1: {results['test_score']:.4f}")
    print(f"  Model saved to: {args.output_dir}")
    
    return results


if __name__ == "__main__":
    main()
