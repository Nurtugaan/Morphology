"""
Model Evaluator for Morphological Analysis (TASK-005)

Класс для оценки обученных моделей на тестовом наборе данных.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, compute_upos_accuracy, compute_all_metrics


@dataclass
class EvaluationResult:
    """Результат оценки модели."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    upos_accuracy: float
    total_samples: int
    num_labels: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """
    Класс для оценки моделей морфологического анализа.
    
    Поддерживает:
    - Загрузку сохранённых моделей
    - Оценку на тестовом наборе
    - Вычисление детальных метрик
    - Экспорт результатов
    """
    
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None
    ):
        """
        Инициализация Evaluator.
        
        Args:
            model: Модель для оценки (опционально)
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.results: List[EvaluationResult] = []
    
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        id2label: Dict[int, str],
        model_name: str = "unknown"
    ) -> EvaluationResult:
        """
        Оценивает модель на тестовом наборе.
        
        Args:
            model: Модель для оценки
            test_loader: DataLoader с тестовыми данными
            id2label: Маппинг индексов к меткам
            model_name: Название модели для отчёта
            
        Returns:
            EvaluationResult с метриками
        """
        model.eval()
        model.to(self.device)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                predictions = logits.argmax(dim=-1).cpu()
                
                # Собираем предсказания (игнорируем -100)
                for pred_seq, label_seq, mask_seq in zip(predictions, labels, attention_mask):
                    for p, l, m in zip(pred_seq, label_seq, mask_seq):
                        if l.item() != -100 and m.item() == 1:
                            pred_label = id2label.get(p.item(), 'UNK')
                            true_label = id2label.get(l.item(), 'UNK')
                            all_predictions.append(pred_label)
                            all_labels.append(true_label)
        
        # Вычисляем метрики
        metrics = compute_metrics(all_labels, all_predictions)
        upos_acc = compute_upos_accuracy(all_labels, all_predictions)
        
        result = EvaluationResult(
            model_name=model_name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            upos_accuracy=upos_acc,
            total_samples=len(all_labels),
            num_labels=len(set(all_labels))
        )
        
        self.results.append(result)
        return result
    
    def evaluate_from_logs(
        self,
        log_files: Dict[str, Path]
    ) -> List[EvaluationResult]:
        """
        Извлекает результаты из логов обучения.
        
        Args:
            log_files: Dict {model_name: log_file_path}
            
        Returns:
            List результатов оценки
        """
        results = []
        
        for model_name, log_path in log_files.items():
            result = self._parse_log_file(log_path, model_name)
            if result:
                results.append(result)
                self.results.append(result)
        
        return results
    
    def _parse_log_file(
        self,
        log_path: Path,
        model_name: str
    ) -> Optional[EvaluationResult]:
        """Парсит лог файл и извлекает метрики."""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Ищем Test Results секцию
            metrics = {}
            lines = content.split('\n')
            in_test_results = False
            
            for line in lines:
                if 'Test Results:' in line:
                    in_test_results = True
                    continue
                
                if in_test_results:
                    if 'Accuracy:' in line:
                        metrics['accuracy'] = float(line.split(':')[-1].strip())
                    elif 'Precision:' in line:
                        metrics['precision'] = float(line.split(':')[-1].strip())
                    elif 'Recall:' in line:
                        metrics['recall'] = float(line.split(':')[-1].strip())
                    elif 'F1:' in line:
                        metrics['f1'] = float(line.split(':')[-1].strip())
                        break
            
            if metrics:
                return EvaluationResult(
                    model_name=model_name,
                    accuracy=metrics.get('accuracy', 0.0),
                    precision=metrics.get('precision', 0.0),
                    recall=metrics.get('recall', 0.0),
                    f1=metrics.get('f1', 0.0),
                    upos_accuracy=0.0,  # Не доступно из логов
                    total_samples=0,
                    num_labels=0
                )
        except Exception as e:
            print(f"Error parsing {log_path}: {e}")
        
        return None
    
    def export_results(
        self,
        output_path: Path,
        format: str = 'json'
    ) -> None:
        """
        Экспортирует результаты в файл.
        
        Args:
            output_path: Путь для сохранения
            format: Формат ('json' или 'csv')
        """
        if format == 'json':
            data = [r.to_dict() for r in self.results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import csv
            if not self.results:
                return
            
            fieldnames = list(self.results[0].to_dict().keys())
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())
    
    def get_best_model(self, metric: str = 'f1') -> Optional[EvaluationResult]:
        """
        Возвращает лучшую модель по заданной метрике.
        
        Args:
            metric: Метрика для сравнения ('f1', 'accuracy', 'precision', 'recall')
            
        Returns:
            Лучший EvaluationResult или None
        """
        if not self.results:
            return None
        
        return max(self.results, key=lambda r: getattr(r, metric, 0.0))
    
    def get_ranking(self, metric: str = 'f1') -> List[EvaluationResult]:
        """
        Возвращает рейтинг моделей по метрике.
        
        Args:
            metric: Метрика для сортировки
            
        Returns:
            Отсортированный список результатов
        """
        return sorted(
            self.results,
            key=lambda r: getattr(r, metric, 0.0),
            reverse=True
        )
