"""
Report Generator for Model Evaluation (TASK-005)

Генерация сравнительных отчётов для моделей морфологического анализа.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .evaluator import EvaluationResult


@dataclass
class ModelInfo:
    """Информация о модели."""
    name: str
    parameters: str
    training_time: str
    architecture: str


class ReportGenerator:
    """
    Генератор отчётов для сравнения моделей.
    
    Поддерживает:
    - Markdown таблицы
    - Рейтинг моделей
    - Рекомендации
    """
    
    # Информация о моделях
    MODEL_INFO = {
        'bert-base': ModelInfo('BERT-base', '109M', '38.6 мин', 'Transformer'),
        'roberta-base': ModelInfo('RoBERTa-base', '125M', '38.8 мин', 'Transformer'),
        'distilbert': ModelInfo('DistilBERT', '66M', '25.9 мин', 'Transformer'),
        'albert-base': ModelInfo('ALBERT-base', '12M', '49.0 мин', 'Transformer'),
        'flair': ModelInfo('Flair', '63M', '~30 мин', 'BiLSTM+CRF'),
    }
    
    def __init__(self, results: List[EvaluationResult]):
        """
        Инициализация генератора.
        
        Args:
            results: Список результатов оценки
        """
        self.results = results
    
    def generate_markdown_report(
        self,
        output_path: Optional[Path] = None,
        include_recommendations: bool = True
    ) -> str:
        """
        Генерирует Markdown отчёт.
        
        Args:
            output_path: Путь для сохранения (опционально)
            include_recommendations: Включить рекомендации
            
        Returns:
            Markdown строка
        """
        sections = [
            self._generate_header(),
            self._generate_overview(),
            self._generate_metrics_table(),
            self._generate_ranking(),
        ]
        
        if include_recommendations:
            sections.append(self._generate_recommendations())
        
        sections.append(self._generate_conclusions())
        
        report = '\n\n'.join(sections)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report
    
    def _generate_header(self) -> str:
        """Генерирует заголовок отчёта."""
        return f"""# TASK-005-EVALUATION-EN: Оценка качества моделей

**Дата:** {datetime.now().strftime('%Y-%m-%d')}
**Задача:** Морфологический анализ английского языка (Token Classification)
**Датасеты:** UD English EWT + GUM (~382k токенов)"""
    
    def _generate_overview(self) -> str:
        """Генерирует обзор."""
        return """## Обзор

В рамках TASK-004 были обучены 5 моделей на английских корпусах Universal Dependencies:
- 4 трансформер-модели (BERT, RoBERTa, DistilBERT, ALBERT)
- 1 RNN-модель (Flair с BiLSTM+CRF)

Все модели решают задачу token classification с ~400 уникальными метками формата `UPOS|FEATS`."""
    
    def _generate_metrics_table(self) -> str:
        """Генерирует таблицу метрик."""
        # Сортируем по F1
        sorted_results = sorted(self.results, key=lambda r: r.f1, reverse=True)
        
        rows = ["## Сравнительная таблица", ""]
        rows.append("| Модель | Accuracy | Precision | Recall | Macro-F1 | Параметры | Время |")
        rows.append("|--------|----------|-----------|--------|----------|-----------|-------|")
        
        for i, result in enumerate(sorted_results):
            model_key = result.model_name.lower().replace('-base', '-base').replace('_', '-')
            info = self.MODEL_INFO.get(model_key, ModelInfo(result.model_name, '?', '?', '?'))
            
            # Подсветка лучшей модели
            name = f"**{info.name}**" if i == 0 else info.name
            f1_str = f"**{result.f1:.2%}**" if i == 0 else f"{result.f1:.2%}"
            
            rows.append(
                f"| {name} | {result.accuracy:.2%} | {result.precision:.2%} | "
                f"{result.recall:.2%} | {f1_str} | {info.parameters} | {info.training_time} |"
            )
        
        return '\n'.join(rows)
    
    def _generate_ranking(self) -> str:
        """Генерирует рейтинг моделей."""
        sorted_results = sorted(self.results, key=lambda r: r.f1, reverse=True)
        
        rows = ["## Рейтинг моделей по Macro-F1", ""]
        
        for i, result in enumerate(sorted_results, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            rows.append(f"{medal} **{result.model_name}** — F1: {result.f1:.2%}")
        
        return '\n'.join(rows)
    
    def _generate_recommendations(self) -> str:
        """Генерирует рекомендации."""
        best = max(self.results, key=lambda r: r.f1)
        second = sorted(self.results, key=lambda r: r.f1, reverse=True)[1]
        
        return f"""## Рекомендации для казахского языка

### Основная модель: {best.model_name}

- ✅ Лучший Macro-F1 ({best.f1:.2%})
- ✅ Оптимальное соотношение качества и скорости обучения
- ✅ Меньший размер модели среди полноразмерных трансформеров

### Альтернатива: {second.model_name}

- Второй по F1 ({second.f1:.2%})
- Надёжный baseline для сравнения

> **Важно:** Для казахского языка будет использоваться **Kaz-RoBERTa**, но гиперпараметры и методика обучения от {best.model_name} применимы."""
    
    def _generate_conclusions(self) -> str:
        """Генерирует выводы."""
        best = max(self.results, key=lambda r: r.f1)
        
        return f"""## Выводы

1. **Лучшая модель:** {best.model_name} с F1 = {best.f1:.2%}
2. **Все трансформеры** показывают схожую accuracy (~94.8%)
3. **Низкий Macro-F1** (~55-59%) объясняется большим числом редких классов (~400 меток)
4. **Flair (BiLSTM+CRF)** уступает трансформерам, но предоставляет альтернативную архитектуру

### Чекпоинты TASK-005 ✅

- [x] Посчитаны Accuracy, Precision, Recall, Macro-F1
- [x] Построена сравнительная таблица
- [x] Выбраны 1-2 лучшие модели
- [x] Сделаны выводы для перехода к казахскому языку"""
    
    def generate_latex_table(self) -> str:
        """Генерирует LaTeX таблицу для диплома."""
        sorted_results = sorted(self.results, key=lambda r: r.f1, reverse=True)
        
        rows = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Сравнение моделей морфологического анализа (EN)}",
            r"\begin{tabular}{|l|c|c|c|c|c|}",
            r"\hline",
            r"\textbf{Модель} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Params} \\",
            r"\hline",
        ]
        
        for result in sorted_results:
            model_key = result.model_name.lower().replace('-base', '-base')
            info = self.MODEL_INFO.get(model_key, ModelInfo(result.model_name, '?', '?', '?'))
            
            rows.append(
                f"{info.name} & {result.accuracy:.2%} & {result.precision:.2%} & "
                f"{result.recall:.2%} & {result.f1:.2%} & {info.parameters} \\\\"
            )
        
        rows.extend([
            r"\hline",
            r"\end{tabular}",
            r"\label{tab:en-models}",
            r"\end{table}",
        ])
        
        return '\n'.join(rows)
