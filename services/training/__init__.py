"""
Training module for English Morphological Analysis (TASK-004)

Structure:
    training/
    ├── config.py       # [SHARED] Configuration for all models
    ├── dataset.py      # [SHARED] PyTorch Dataset with label alignment
    ├── model.py        # [SHARED] MorphologyTagger wrapper
    ├── trainer.py      # [SHARED] Training loop, validation, checkpoints
    ├── train_base.py   # [SHARED] Base training function for Transformers
    └── models/         # [MODEL-SPECIFIC] Individual training scripts
        ├── train_bert.py
        ├── train_roberta.py
        ├── train_distilbert.py
        ├── train_albert.py
        └── train_flair.py  # Uses different library (flair, not transformers)
"""

from .config import TrainingConfig, get_config
from .dataset import TokenClassificationDataset, create_dataloaders
from .model import MorphologyTagger, create_model
from .trainer import Trainer, set_seed
from .train_base import train_model, load_data

__all__ = [
    'TrainingConfig',
    'get_config',
    'TokenClassificationDataset',
    'create_dataloaders',
    'MorphologyTagger',
    'create_model',
    'Trainer',
    'set_seed',
    'train_model',
    'load_data'
]
