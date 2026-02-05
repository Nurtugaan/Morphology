"""
Trainer for English Morphological Analysis (TASK-004)

Класс для обучения и валидации модели морфологического анализа.
Включает training loop, валидацию, логирование и сохранение чекпоинтов.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

from .config import TrainingConfig
from .model import MorphologyTagger


class Trainer:
    """
    Класс для обучения и валидации модели морфологического анализа.
    
    Функции:
    - Training loop с gradient accumulation
    - Validation с вычислением метрик
    - Сохранение чекпоинтов (best model, last checkpoint)
    - Early stopping
    - Logging (console + JSON)
    
    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, config, label2id, id2label)
        >>> results = trainer.train()
        >>> print(results['best_f1'])
    """
    
    def __init__(
        self,
        model: MorphologyTagger,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        test_loader: Optional[DataLoader] = None
    ):
        """
        Инициализация Trainer.
        
        Args:
            model: Модель MorphologyTagger
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            config: Конфигурация обучения
            label2id: Словарь метка -> индекс
            id2label: Словарь индекс -> метка
            test_loader: Optional DataLoader для тестирования
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.label2id = label2id
        self.id2label = id2label
        
        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.early_stopping_counter = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'learning_rate': []
        }
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Output dir: {self.output_dir}")
    
    def _create_optimizer(self) -> AdamW:
        """Создаёт оптимизатор AdamW."""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon
        )
        
        return optimizer
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Обучение одной эпохи.
        
        Returns:
            Dict с train_loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Валидация модели.
        
        Args:
            data_loader: DataLoader для валидации (по умолчанию val_loader)
            
        Returns:
            Dict с метриками: loss, accuracy, precision, recall, f1
        """
        if data_loader is None:
            data_loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Validating", leave=False):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                num_batches += 1
                
                # Get predictions
                predictions = outputs['logits'].argmax(dim=-1)
                
                # Collect predictions and labels (ignore -100)
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    for p, l, m in zip(pred, label, mask):
                        if l.item() != -100 and m.item() == 1:
                            all_predictions.append(p.item())
                            all_labels.append(l.item())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        metrics = {
            'val_loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Полный цикл обучения.
        
        Returns:
            Dict с результатами обучения
        """
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['accuracy'].append(val_metrics['accuracy'])
            self.history['precision'].append(val_metrics['precision'])
            self.history['recall'].append(val_metrics['recall'])
            self.history['f1'].append(val_metrics['f1'])
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} completed in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  Precision:  {val_metrics['precision']:.4f}")
            print(f"  Recall:     {val_metrics['recall']:.4f}")
            print(f"  F1:         {val_metrics['f1']:.4f}")
            
            # Check for best model
            if val_metrics['f1'] > self.best_f1 + self.config.early_stopping_threshold:
                self.best_f1 = val_metrics['f1']
                self.best_epoch = epoch + 1
                self.early_stopping_counter = 0
                
                # Save best model
                self.save_checkpoint(self.output_dir / 'best_model')
                print(f"  ✓ New best model saved! F1: {self.best_f1:.4f}")
            else:
                self.early_stopping_counter += 1
                print(f"  Early stopping counter: {self.early_stopping_counter}/{self.config.early_stopping_patience}")
            
            # Save last checkpoint
            self.save_checkpoint(self.output_dir / 'last_checkpoint')
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Save training history
        self._save_history()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("Training Completed!")
        print("=" * 60)
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best F1: {self.best_f1:.4f} (Epoch {self.best_epoch})")
        print(f"Model saved to: {self.output_dir}")
        
        return {
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'history': self.history
        }
    
    def save_checkpoint(self, save_path: Path):
        """
        Сохраняет чекпоинт модели.
        
        Args:
            save_path: Путь для сохранения
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path, self.label2id)
        
        # Save training state
        trainer_state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_f1': self.best_f1,
            'best_epoch': self.best_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(trainer_state, save_path / 'trainer_state.pt')
        
        # Save config
        self.config.save(str(save_path / 'training_config.json'))
    
    def _save_history(self):
        """Сохраняет историю обучения в JSON."""
        history_path = self.output_dir / 'training_history.json'
        
        history_data = {
            'experiment': f"{self.config.model_name}-morphology",
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'history': self.history,
            'best_metrics': {
                'epoch': self.best_epoch,
                'f1': self.best_f1
            }
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training history saved to {history_path}")
    
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """
        Оценка модели на тестовом наборе.
        
        Args:
            test_loader: DataLoader для тестирования
            
        Returns:
            Dict с детальными метриками
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        if test_loader is None:
            raise ValueError("No test loader provided")
        
        print("\nEvaluating on test set...")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = outputs['logits'].argmax(dim=-1)
                
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    for p, l, m in zip(pred, label, mask):
                        if l.item() != -100 and m.item() == 1:
                            all_predictions.append(p.item())
                            all_labels.append(l.item())
        
        # Convert to label names
        pred_labels = [self.id2label.get(p, f"UNK_{p}") for p in all_predictions]
        true_labels = [self.id2label.get(l, f"UNK_{l}") for l in all_labels]
        
        # Classification report
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        
        # Print summary
        print("\nTest Results:")
        print(f"  Accuracy:  {report['accuracy']:.4f}")
        print(f"  Precision: {report['macro avg']['precision']:.4f}")
        print(f"  Recall:    {report['macro avg']['recall']:.4f}")
        print(f"  F1:        {report['macro avg']['f1-score']:.4f}")
        
        return report


def set_seed(seed: int = 42):
    """
    Фиксирует все источники случайности для воспроизводимости.
    
    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
