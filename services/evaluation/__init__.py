"""
Evaluation Module for English Morphological Analysis (TASK-005)

Модуль оценки качества моделей морфологического анализа.
"""

from .metrics import (
    compute_metrics,
    compute_upos_accuracy,
    compute_per_class_metrics,
)
from .evaluator import ModelEvaluator
from .report_generator import ReportGenerator

__all__ = [
    'compute_metrics',
    'compute_upos_accuracy', 
    'compute_per_class_metrics',
    'ModelEvaluator',
    'ReportGenerator',
]
