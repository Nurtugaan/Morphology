"""
Morphology Tagger Model for English Morphological Analysis (TASK-004)

Модель для морфологического анализа на основе трансформеров.
Архитектура: Pre-trained Transformer Encoder → Dropout → Linear Classification Head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Any
from pathlib import Path
import json


class MorphologyTagger(nn.Module):
    """
    Модель для морфологического анализа на основе трансформеров.
    
    Архитектура:
    1. Pre-trained transformer encoder (BERT, RoBERTa, etc.)
    2. Dropout layer для регуляризации
    3. Linear classification head для предсказания меток
    
    Схема:
        Input -> Transformer -> Dropout -> Linear -> Output
        [B, L] -> [B, L, H]   -> [B, L, H] -> [B, L, C]
        
        где B = batch_size, L = seq_length, H = hidden_size, C = num_labels
    
    Example:
        >>> model = MorphologyTagger("bert-base-uncased", num_labels=500)
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss = outputs['loss']
        >>> logits = outputs['logits']
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Инициализация модели.
        
        Args:
            model_name: Название модели из HuggingFace (например, 'bert-base-uncased')
            num_labels: Количество классов (уникальных меток)
            dropout: Вероятность dropout
            freeze_encoder: Если True, замораживает веса encoder (только для feature extraction)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Get hidden size from config
        self.hidden_size = self.config.hidden_size
        
        # Optional: freeze encoder weights
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов классификатора."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass модели.
        
        Args:
            input_ids: Tensor [batch_size, seq_length] - ID токенов
            attention_mask: Tensor [batch_size, seq_length] - маска внимания
            labels: Optional Tensor [batch_size, seq_length] - метки для расчёта loss
            token_type_ids: Optional Tensor для BERT-подобных моделей
            
        Returns:
            Dict с ключами:
                - 'loss': Scalar tensor (если labels предоставлены)
                - 'logits': Tensor [batch_size, seq_length, num_labels]
        """
        # Prepare encoder inputs
        encoder_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add token_type_ids only if model supports it
        if token_type_ids is not None and hasattr(self.config, 'type_vocab_size'):
            encoder_kwargs['token_type_ids'] = token_type_ids
        
        # Get encoder outputs
        encoder_outputs = self.encoder(**encoder_kwargs)
        
        # Get last hidden state [batch_size, seq_length, hidden_size]
        sequence_output = encoder_outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Classification [batch_size, seq_length, num_labels]
        logits = self.classifier(sequence_output)
        
        # Prepare output
        output = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            # Reshape for loss calculation
            # logits: [batch_size * seq_length, num_labels]
            # labels: [batch_size * seq_length]
            loss = self.loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
            output['loss'] = loss
        
        return output
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Предсказание меток (без расчёта loss).
        
        Args:
            input_ids: Tensor [batch_size, seq_length]
            attention_mask: Tensor [batch_size, seq_length]
            
        Returns:
            Tensor [batch_size, seq_length] с предсказанными метками
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = outputs['logits'].argmax(dim=-1)
        return predictions
    
    def save_pretrained(self, save_path: str, label2id: Dict[str, int] = None):
        """
        Сохраняет модель и конфигурацию.
        
        Args:
            save_path: Путь для сохранения
            label2id: Словарь меток для сохранения
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save transformer encoder and tokenizer config
        self.encoder.save_pretrained(save_path)
        
        # Save classifier weights separately
        classifier_state = {
            'classifier': self.classifier.state_dict(),
            'dropout_prob': self.dropout_prob,
            'num_labels': self.num_labels,
            'hidden_size': self.hidden_size,
            'model_name': self.model_name
        }
        torch.save(classifier_state, save_path / 'classifier.pt')
        
        # Save label mapping
        if label2id is not None:
            with open(save_path / 'label2id.json', 'w', encoding='utf-8') as f:
                json.dump(label2id, f, ensure_ascii=False, indent=2)
            
            id2label = {v: k for k, v in label2id.items()}
            with open(save_path / 'id2label.json', 'w', encoding='utf-8') as f:
                json.dump(id2label, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cpu') -> "MorphologyTagger":
        """
        Загружает модель из сохранённого состояния.
        
        Args:
            load_path: Путь к сохранённой модели
            device: Устройство для загрузки ('cpu' или 'cuda')
            
        Returns:
            Загруженная модель MorphologyTagger
        """
        load_path = Path(load_path)
        
        # Load classifier config
        classifier_state = torch.load(load_path / 'classifier.pt', map_location=device)
        
        # Create model
        model = cls(
            model_name=str(load_path),  # Load from saved encoder
            num_labels=classifier_state['num_labels'],
            dropout=classifier_state['dropout_prob']
        )
        
        # Load classifier weights
        model.classifier.load_state_dict(classifier_state['classifier'])
        model.to(device)
        
        print(f"✓ Model loaded from {load_path}")
        return model
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """
        Возвращает количество параметров модели.
        
        Args:
            trainable_only: Если True, считает только обучаемые параметры
            
        Returns:
            Количество параметров
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    model_name: str,
    num_labels: int,
    dropout: float = 0.1,
    device: str = 'cpu'
) -> MorphologyTagger:
    """
    Создаёт и инициализирует модель MorphologyTagger.
    
    Args:
        model_name: Название модели из HuggingFace
        num_labels: Количество классов
        dropout: Вероятность dropout
        device: Устройство ('cpu' или 'cuda')
        
    Returns:
        Инициализированная модель
    """
    model = MorphologyTagger(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout
    )
    model.to(device)
    
    num_params = model.get_num_parameters()
    print(f"✓ Model created: {model_name}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Number of labels: {num_labels}")
    print(f"  Device: {device}")
    
    return model


if __name__ == "__main__":
    # Demo usage
    print("MorphologyTagger Demo")
    print("=" * 50)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        model_name="bert-base-uncased",
        num_labels=500,
        dropout=0.1,
        device=device
    )
    
    # Create dummy input
    batch_size = 2
    seq_length = 32
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    labels = torch.randint(0, 500, (batch_size, seq_length)).to(device)
    
    # Forward pass with labels
    print("\nForward pass with labels:")
    outputs = model(input_ids, attention_mask, labels=labels)
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Forward pass without labels (inference)
    print("\nForward pass without labels:")
    outputs = model(input_ids, attention_mask)
    print(f"  Logits shape: {outputs['logits'].shape}")
    
    # Prediction
    print("\nPrediction:")
    predictions = model.predict(input_ids, attention_mask)
    print(f"  Predictions shape: {predictions.shape}")
    
    print("\n✓ Model works correctly!")
