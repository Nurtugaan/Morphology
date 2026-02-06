"""
Тестовый скрипт для модуля evaluation (TASK-005)

Запуск:
    cd c:\Diploma\morphology\services
    python -m evaluation.test_evaluation
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import compute_metrics, compute_upos_accuracy, ModelEvaluator, ReportGenerator
from evaluation.evaluator import EvaluationResult


def test_metrics():
    """Тест функций метрик."""
    print("=" * 50)
    print("Тест: compute_metrics")
    print("=" * 50)
    
    y_true = [
        "NOUN|Number=Sing", 
        "VERB|Tense=Past", 
        "ADJ|Degree=Pos",
        "NOUN|Number=Sing",
        "PUNCT|_"
    ]
    y_pred = [
        "NOUN|Number=Sing",  # correct
        "VERB|Tense=Past",   # correct
        "ADJ|Degree=Cmp",    # wrong FEATS
        "VERB|Tense=Pres",   # wrong
        "PUNCT|_"            # correct
    ]
    
    metrics = compute_metrics(y_true, y_pred)
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1:        {metrics['f1']:.2%}")
    
    assert metrics['accuracy'] == 0.6, f"Expected 0.6, got {metrics['accuracy']}"
    print("  ✓ compute_metrics работает корректно")


def test_upos_accuracy():
    """Тест UPOS accuracy."""
    print("\n" + "=" * 50)
    print("Тест: compute_upos_accuracy")
    print("=" * 50)
    
    y_true = [
        "NOUN|Number=Sing", 
        "VERB|Tense=Past", 
        "ADJ|Degree=Pos",
    ]
    y_pred = [
        "NOUN|Number=Plur",  # UPOS correct, FEATS wrong
        "VERB|Tense=Pres",   # UPOS correct, FEATS wrong
        "ADV|_",             # UPOS wrong
    ]
    
    upos_acc = compute_upos_accuracy(y_true, y_pred)
    print(f"  UPOS Accuracy: {upos_acc:.2%}")
    
    assert upos_acc == 2/3, f"Expected 0.67, got {upos_acc}"
    print("  ✓ compute_upos_accuracy работает корректно")


def test_evaluator_from_logs():
    """Тест загрузки результатов из логов."""
    print("\n" + "=" * 50)
    print("Тест: ModelEvaluator.evaluate_from_logs")
    print("=" * 50)
    
    log_dir = Path(__file__).parent.parent.parent / "docs" / "logs"
    
    if not log_dir.exists():
        print(f"  ⚠ Директория логов не найдена: {log_dir}")
        return
    
    log_files = {
        'bert-base': log_dir / 'train_bert.txt',
        'roberta-base': log_dir / 'train_roberta.txt',
        'distilbert': log_dir / 'train_distilbert.txt',
        'albert-base': log_dir / 'train_albert.txt',
        'flair': log_dir / 'train_flair.txt',
    }
    
    # Проверяем существование файлов
    existing_logs = {k: v for k, v in log_files.items() if v.exists()}
    print(f"  Найдено логов: {len(existing_logs)}/{len(log_files)}")
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_from_logs(existing_logs)
    
    print(f"  Распознано результатов: {len(results)}")
    
    for r in results:
        print(f"    - {r.model_name}: F1={r.f1:.2%}")
    
    # Получаем лучшую модель
    best = evaluator.get_best_model(metric='f1')
    if best:
        print(f"\n  🥇 Лучшая модель: {best.model_name} (F1={best.f1:.2%})")
    
    print("  ✓ ModelEvaluator работает корректно")


def test_report_generator():
    """Тест генератора отчётов."""
    print("\n" + "=" * 50)
    print("Тест: ReportGenerator")
    print("=" * 50)
    
    # Создаём тестовые результаты
    results = [
        EvaluationResult("distilbert", 0.9489, 0.5992, 0.5932, 0.5884, 0.97, 25000, 400),
        EvaluationResult("bert-base", 0.9483, 0.5783, 0.5731, 0.5679, 0.96, 25000, 400),
        EvaluationResult("albert-base", 0.9464, 0.5771, 0.5611, 0.5609, 0.96, 25000, 400),
    ]
    
    generator = ReportGenerator(results)
    
    # Генерируем Markdown
    report = generator.generate_markdown_report()
    print(f"  Сгенерирован Markdown отчёт: {len(report)} символов")
    
    # Проверяем наличие ключевых секций
    assert "## Сравнительная таблица" in report
    assert "## Рейтинг" in report
    assert "DistilBERT" in report
    
    print("  ✓ ReportGenerator работает корректно")


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 50)
    print("  TASK-005: Тестирование модуля evaluation")
    print("=" * 50)
    
    test_metrics()
    test_upos_accuracy()
    test_evaluator_from_logs()
    test_report_generator()
    
    print("\n" + "=" * 50)
    print("  ✅ Все тесты пройдены!")
    print("=" * 50)


if __name__ == "__main__":
    main()
