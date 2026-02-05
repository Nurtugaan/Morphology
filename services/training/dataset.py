"""
Token Classification Dataset for English Morphological Analysis (TASK-004)

PyTorch Dataset для обучения моделей морфологического анализа.
Выполняет токенизацию и выравнивание меток (label alignment).
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.en_preprocessing import Sentence


class TokenClassificationDataset(Dataset):
    """
    PyTorch Dataset для задачи token classification.
    
    Выполняет:
    - Токенизацию с помощью HuggingFace tokenizer
    - Выравнивание меток (label alignment) для субтокенов
    - Паддинг и создание attention mask
    
    При субсловной токенизации:
    - Метка назначается только первому субтокену слова
    - Все последующие субтокены получают метку -100 (игнорируются при расчёте loss)
    - Специальные токены ([CLS], [SEP], [PAD]) также получают -100
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> dataset = TokenClassificationDataset(sentences, tokenizer, label2id)
        >>> item = dataset[0]
        >>> item.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """
    
    def __init__(
        self,
        sentences: List[Sentence],
        tokenizer: PreTrainedTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
        label_all_tokens: bool = False
    ):
        """
        Инициализация датасета.
        
        Args:
            sentences: Список предложений (объекты Sentence из en_preprocessing)
            tokenizer: HuggingFace токенизатор
            label2id: Словарь метка -> индекс
            max_length: Максимальная длина последовательности
            label_all_tokens: Если True, назначает метку всем субтокенам слова.
                              Если False (по умолчанию), только первому субтокену.
        """
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        
        # Precompute tokenized data for faster access
        self._tokenized_data = self._tokenize_all()
    
    def _tokenize_and_align_labels(self, sentence: Sentence) -> Dict[str, torch.Tensor]:
        """
        Токенизирует предложение и выравнивает метки.
        
        Args:
            sentence: Объект Sentence с токенами и метками
            
        Returns:
            Dict с input_ids, attention_mask, labels
        """
        # Get words and labels
        words = sentence.get_forms()
        labels = sentence.get_labels()
        
        # Tokenize with return_offsets_mapping for word alignment
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get word_ids to align labels with subtokens
        word_ids = tokenized.word_ids()
        
        # Align labels with subtokens
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subtoken of a word - assign the actual label
                label = labels[word_idx]
                label_id = self.label2id.get(label, self.label2id.get("O", 0))
                aligned_labels.append(label_id)
            else:
                # Subsequent subtokens of the same word
                if self.label_all_tokens:
                    # Option: label all subtokens with the same label
                    label = labels[word_idx]
                    label_id = self.label2id.get(label, self.label2id.get("O", 0))
                    aligned_labels.append(label_id)
                else:
                    # Default: ignore subsequent subtokens in loss calculation
                    aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def _tokenize_all(self) -> List[Dict[str, torch.Tensor]]:
        """
        Токенизирует все предложения заранее для быстрого доступа.
        
        Returns:
            Список словарей с токенизированными данными
        """
        tokenized_data = []
        for sentence in self.sentences:
            try:
                item = self._tokenize_and_align_labels(sentence)
                tokenized_data.append(item)
            except Exception as e:
                print(f"Warning: Failed to tokenize sentence: {e}")
                continue
        return tokenized_data
    
    def __len__(self) -> int:
        """Возвращает количество примеров в датасете."""
        return len(self._tokenized_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Возвращает токенизированный пример по индексу.
        
        Args:
            idx: Индекс примера
            
        Returns:
            Dict с ключами 'input_ids', 'attention_mask', 'labels'
        """
        return self._tokenized_data[idx]


def create_dataloaders(
    train_sentences: List[Sentence],
    val_sentences: List[Sentence],
    tokenizer: PreTrainedTokenizer,
    label2id: Dict[str, int],
    batch_size: int = 16,
    max_length: int = 128,
    num_workers: int = 0
) -> tuple:
    """
    Создаёт DataLoader'ы для обучения и валидации.
    
    Args:
        train_sentences: Список предложений для обучения
        val_sentences: Список предложений для валидации
        tokenizer: HuggingFace токенизатор
        label2id: Словарь метка -> индекс
        batch_size: Размер батча
        max_length: Максимальная длина последовательности
        num_workers: Количество воркеров для DataLoader
        
    Returns:
        Tuple (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = TokenClassificationDataset(
        sentences=train_sentences,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length
    )
    
    val_dataset = TokenClassificationDataset(
        sentences=val_sentences,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Demo usage
    print("TokenClassificationDataset Demo")
    print("=" * 50)
    
    from transformers import AutoTokenizer
    
    # Create mock data for testing
    from dataclasses import dataclass, field
    from typing import List as TypeList
    
    @dataclass
    class MockToken:
        form: str
        label: str
    
    @dataclass
    class MockSentence:
        tokens: TypeList[MockToken] = field(default_factory=list)
        
        def get_forms(self):
            return [t.form for t in self.tokens]
        
        def get_labels(self):
            return [t.label for t in self.tokens]
    
    # Create test sentence
    sentence = MockSentence(tokens=[
        MockToken("The", "DET|Definite=Def"),
        MockToken("cat", "NOUN|Number=Sing"),
        MockToken("sat", "VERB|Tense=Past"),
        MockToken(".", "PUNCT|_")
    ])
    
    # Create label vocabulary
    label2id = {
        "DET|Definite=Def": 0,
        "NOUN|Number=Sing": 1,
        "VERB|Tense=Past": 2,
        "PUNCT|_": 3,
        "O": 4
    }
    id2label = {v: k for k, v in label2id.items()}
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset
    print("Creating dataset...")
    dataset = TokenClassificationDataset(
        sentences=[sentence],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=32
    )
    
    # Get item
    item = dataset[0]
    
    print(f"\nInput tokens: {sentence.get_forms()}")
    print(f"Input labels: {sentence.get_labels()}")
    print(f"\nTokenized shape:")
    print(f"  input_ids: {item['input_ids'].shape}")
    print(f"  attention_mask: {item['attention_mask'].shape}")
    print(f"  labels: {item['labels'].shape}")
    
    # Decode for visualization
    tokens = tokenizer.convert_ids_to_tokens(item['input_ids'])
    labels = item['labels'].tolist()
    
    print(f"\nToken-Label alignment:")
    for tok, lab in zip(tokens[:15], labels[:15]):  # First 15 tokens
        if lab == -100:
            lab_str = "[IGNORED]"
        else:
            lab_str = id2label.get(lab, f"UNK({lab})")
        print(f"  {tok:15} -> {lab_str}")
    
    print("\n✓ Dataset created successfully!")
