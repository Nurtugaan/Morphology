"""
Training Configuration for English Morphological Analysis (TASK-004)

Конфигурация гиперпараметров для обучения трансформер-моделей.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    """
    Конфигурация для обучения модели морфологического анализа.
    
    Attributes:
        model_name: Название модели из HuggingFace (например, 'bert-base-uncased')
        output_dir: Директория для сохранения модели
        
        # Training hyperparameters
        learning_rate: Начальная скорость обучения
        batch_size: Размер батча
        epochs: Количество эпох обучения
        max_length: Максимальная длина последовательности
        
        # Optimizer settings
        warmup_ratio: Доля шагов для warmup
        weight_decay: Регуляризация весов
        gradient_accumulation_steps: Шаги накопления градиента
        max_grad_norm: Максимальная норма градиента
        
        # Other settings
        seed: Random seed для воспроизводимости
        fp16: Использовать mixed precision
        logging_steps: Частота логирования
        save_strategy: Стратегия сохранения чекпоинтов
        evaluation_strategy: Стратегия валидации
    """
    
    # Model
    model_name: str = "bert-base-uncased"
    output_dir: str = "services/models/en/bert-base"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 3
    max_length: int = 128
    
    # Optimizer settings
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Adam betas
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Dropout
    dropout: float = 0.1
    
    # Other settings
    seed: int = 42
    fp16: bool = False  # Set True if GPU supports it
    logging_steps: int = 100
    save_strategy: str = "epoch"  # "epoch" or "steps"
    evaluation_strategy: str = "epoch"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    
    def to_dict(self) -> dict:
        """Преобразует конфиг в словарь."""
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_length": self.max_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "dropout": self.dropout,
            "seed": self.seed,
            "fp16": self.fp16,
            "logging_steps": self.logging_steps,
            "save_strategy": self.save_strategy,
            "evaluation_strategy": self.evaluation_strategy,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_threshold": self.early_stopping_threshold,
            "device": self.device
        }
    
    def save(self, path: str):
        """Сохраняет конфиг в JSON файл."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Загружает конфиг из JSON файла."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


# Предустановленные конфигурации для разных моделей
BERT_BASE_CONFIG = TrainingConfig(
    model_name="bert-base-uncased",
    output_dir="services/models/en/bert-base",
    learning_rate=2e-5,
    batch_size=16,
    epochs=3
)

ROBERTA_BASE_CONFIG = TrainingConfig(
    model_name="roberta-base",
    output_dir="services/models/en/roberta-base",
    learning_rate=2e-5,
    batch_size=16,
    epochs=3
)

DISTILBERT_CONFIG = TrainingConfig(
    model_name="distilbert-base-uncased",
    output_dir="services/models/en/distilbert",
    learning_rate=3e-5,  # Slightly higher for smaller model
    batch_size=32,       # Can use larger batch
    epochs=4
)

ALBERT_BASE_CONFIG = TrainingConfig(
    model_name="albert-base-v2",
    output_dir="services/models/en/albert-base",
    learning_rate=2e-5,
    batch_size=32,  # ALBERT is memory efficient
    epochs=4
)


def get_config(model_name: str) -> TrainingConfig:
    """
    Возвращает конфигурацию для указанной модели.
    
    Args:
        model_name: Название модели (bert-base, roberta-base, distilbert, albert-base)
        
    Returns:
        TrainingConfig для указанной модели
    """
    configs = {
        "bert-base": BERT_BASE_CONFIG,
        "bert-base-uncased": BERT_BASE_CONFIG,
        "roberta-base": ROBERTA_BASE_CONFIG,
        "distilbert": DISTILBERT_CONFIG,
        "distilbert-base-uncased": DISTILBERT_CONFIG,
        "albert-base": ALBERT_BASE_CONFIG,
        "albert-base-v2": ALBERT_BASE_CONFIG,
    }
    
    if model_name.lower() in configs:
        return configs[model_name.lower()]
    
    # Return default config with custom model name
    return TrainingConfig(
        model_name=model_name,
        output_dir=f"services/models/en/{model_name.replace('/', '-')}"
    )
