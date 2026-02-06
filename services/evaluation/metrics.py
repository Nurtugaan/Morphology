"""
Metrics Module for Morphological Analysis Evaluation (TASK-005)

Вычисление метрик качества для token classification.
"""

from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    average: str = 'macro',
    zero_division: int = 0
) -> Dict[str, float]:
    """
    Вычисляет основные метрики для token classification.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        average: Тип усреднения ('macro', 'micro', 'weighted')
        zero_division: Значение при делении на ноль
        
    Returns:
        Dict с метриками: accuracy, precision, recall, f1
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        'recall': recall_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
        'f1': f1_score(
            y_true, y_pred, average=average, zero_division=zero_division
        ),
    }


def compute_upos_accuracy(
    y_true: List[str],
    y_pred: List[str]
) -> float:
    """
    Вычисляет точность только по UPOS тегам (игнорируя FEATS).
    
    Метки имеют формат "UPOS|FEATS", функция извлекает только UPOS часть.
    
    Args:
        y_true: Истинные метки в формате UPOS|FEATS
        y_pred: Предсказанные метки в формате UPOS|FEATS
        
    Returns:
        UPOS accuracy (0.0 - 1.0)
    """
    def extract_upos(label: str) -> str:
        """Извлекает UPOS из метки UPOS|FEATS."""
        if '|' in label:
            return label.split('|')[0]
        return label
    
    y_true_upos = [extract_upos(label) for label in y_true]
    y_pred_upos = [extract_upos(label) for label in y_pred]
    
    return accuracy_score(y_true_upos, y_pred_upos)


def compute_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    output_dict: bool = True
) -> Dict[str, Any]:
    """
    Вычисляет метрики для каждого класса отдельно.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        labels: Список меток для включения (опционально)
        output_dict: Возвращать как dict (True) или строку (False)
        
    Returns:
        Dict с метриками по классам или строка отчёта
    """
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=output_dict,
        zero_division=0
    )


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Вычисляет матрицу ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        labels: Список меток для включения
        
    Returns:
        Confusion matrix как numpy array
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_all_metrics(
    y_true: List[str],
    y_pred: List[str]
) -> Dict[str, Any]:
    """
    Вычисляет все доступные метрики.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        
    Returns:
        Dict со всеми метриками
    """
    basic_metrics = compute_metrics(y_true, y_pred)
    
    return {
        **basic_metrics,
        'upos_accuracy': compute_upos_accuracy(y_true, y_pred),
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class': compute_per_class_metrics(y_true, y_pred),
    }
