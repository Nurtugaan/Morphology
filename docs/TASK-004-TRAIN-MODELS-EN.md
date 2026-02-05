# TASK-004: English Morphological Model Training Pipeline

## Обзор

Данный документ описывает техническое задание на реализацию пайплайна обучения трансформер-моделей для задачи морфологического анализа (token classification) на английском языке. Пайплайн использует предобработанные данные из TASK-003 и модели-кандидаты из TASK-001.

---

## Цели задачи

1. **Реализовать training loop** для обучения трансформер-моделей на задачу token classification
2. **Обучить выбранные модели** (BERT-base, RoBERTa-base, DistilBERT) на английском датасете
3. **Зафиксировать гиперпараметры** и обеспечить воспроизводимость экспериментов
4. **Получить baseline-результаты** для последующего сравнения с казахскими моделями

---

## Входные данные

### Источники данных

Данные поступают из TASK-003 (предобработка):

| Источник | Описание |
|----------|----------|
| `en_preprocessing.py` | Модуль загрузки и предобработки данных |
| `UD_English-EWT` | Основной обучающий датасет (~205k токенов) |
| `UD_English-GUM` | Дополнительный датасет для валидации (~135k токенов) |

### Формат данных

```python
{
    'tokens': ['The', 'cat', 'sat'],
    'labels': ['DET|Definite=Def', 'NOUN|Number=Sing', 'VERB|Tense=Past'],
    'label_ids': [5, 23, 45],
    'sentence_id': '...',
    'text': 'The cat sat'
}
```

### Словарь меток

- **label2id**: `Dict[str, int]` — метка → индекс
- **id2label**: `Dict[int, str]` — индекс → метка
- Ожидаемое количество меток: ~300-500 уникальных комбинаций `UPOS|FEATS`

---

## Модели для обучения

На основании TASK-001 выбраны следующие модели:

| Модель | Архитектура | Размер / Параметры | HuggingFace ID | Задачи | Почему выбрана |
|--------|-------------|-------------------|----------------|--------|----------------|
| **BERT-base** | BERT | base / 110M | `bert-base-uncased` | POS, Morphology, NER | Классическая базовая модель для точного анализа, проверена на английском |
| **RoBERTa-base** | RoBERTa | base / 125M | `roberta-base` | POS, Token Classification | Улучшенная предобученность, хороший контекст для token-level задач |
| **DistilBERT** | BERT | small / 66M | `distilbert-base-uncased` | POS, Token Classification | Быстрая и лёгкая, удобна для прототипов и сравнений скорости |
| **ALBERT-base** | ALBERT | base / 12M | `albert-base-v2` | POS, Morphology | Экономная по памяти, другой подход в архитектуре (factorized weights) |
| **Flair** | LSTM+Char | — | `flair` (библиотека) | POS, NER | Другая архитектура (RNN+char embeddings), хорошо для экспериментов и контраст с трансформерами |

> [!NOTE]
> Flair использует отдельную библиотеку `flair` и требует особого подхода к обучению (не HuggingFace Transformers).

---

## Архитектура модели

### Схема

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                         │
│           ["The", "cat", "sat", "on", "mat"]           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    Tokenizer                            │
│     (Subword tokenization + special tokens)             │
│   ["[CLS]", "The", "cat", "sat", "on", "mat", "[SEP]"] │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Transformer Encoder                        │
│         (BERT / RoBERTa / DistilBERT)                  │
│                                                         │
│   Выход: [batch_size, seq_len, hidden_size]            │
│          hidden_size = 768 (base models)                │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Dropout Layer (p=0.1)                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            Linear Classification Head                   │
│         (hidden_size → num_labels)                      │
│                                                         │
│   Выход: [batch_size, seq_len, num_labels]             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Output Logits                          │
│     → CrossEntropyLoss (ignore_index=-100)             │
│     → Predicted Labels                                  │
└─────────────────────────────────────────────────────────┘
```

### Выравнивание субтокенов (Label Alignment)

При субсловной токенизации необходимо:
- Назначить метку **только первому субтокену** слова
- Все последующие субтокены получают метку `-100` (игнорируются при расчёте loss)
- Специальные токены (`[CLS]`, `[SEP]`, `[PAD]`) также получают `-100`

```python
# Пример выравнивания
Word:       "unhappiness"
Subtokens:  ["un", "##happy", "##ness"]
Labels:     [LABEL_ID, -100, -100]
```

---

## Гиперпараметры

### Основные параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `learning_rate` | 2e-5 | Начальная скорость обучения |
| `batch_size` | 16 | Размер батча (на GPU) |
| `epochs` | 3-5 | Количество эпох обучения |
| `max_length` | 128 | Максимальная длина последовательности |
| `warmup_ratio` | 0.1 | Доля шагов для warmup |
| `weight_decay` | 0.01 | Регуляризация весов |
| `gradient_accumulation` | 2 | Шаги накопления градиента |

### Оптимизатор

- **Optimizer**: AdamW с bias correction
- **Scheduler**: Linear scheduler with warmup

```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

### Функция потерь

- **Loss Function**: CrossEntropyLoss с `ignore_index=-100`

```python
criterion = nn.CrossEntropyLoss(ignore_index=-100)
```

---

## Структура кода

### Файловая структура

```
services/
├── training/
│   ├── en_training.py          # Основной модуль обучения
│   ├── trainer.py              # Класс Trainer
│   ├── dataset.py              # PyTorch Dataset для токенов
│   ├── model.py                # Обёртка над трансформерами
│   ├── utils.py                # Вспомогательные функции
│   └── config.py               # Конфигурация гиперпараметров
│
├── models/
│   └── en/
│       ├── bert-base/          # Обученная модель BERT
│       ├── roberta-base/       # Обученная модель RoBERTa
│       ├── distilbert/         # Обученная модель DistilBERT
│       ├── albert-base/        # Обученная модель ALBERT
│       └── flair/              # Обученная модель Flair
│
└── logs/
    └── english/
        ├── training_logs.json  # Логи обучения
        └── tensorboard/        # TensorBoard логи
```

### Основные компоненты

#### 1. TokenClassificationDataset

```python
class TokenClassificationDataset(Dataset):
    """
    PyTorch Dataset для задачи token classification.
    
    Выполняет:
    - Токенизацию с помощью HuggingFace tokenizer
    - Выравнивание меток (label alignment)
    - Паддинг и создание attention mask
    """
    
    def __init__(
        self,
        sentences: List[Sentence],
        tokenizer: PreTrainedTokenizer,
        label2id: Dict[str, int],
        max_length: int = 128
    ):
        ...
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [max_length]
        """
        ...
```

#### 2. MorphologyTagger

```python
class MorphologyTagger(nn.Module):
    """
    Модель для морфологического анализа на основе трансформеров.
    
    Архитектура:
    - Pre-trained transformer encoder
    - Dropout layer
    - Linear classification head
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1
    ):
        ...
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            loss: Scalar tensor (if labels provided)
            logits: [batch_size, seq_len, num_labels]
        """
        ...
```

#### 3. Trainer

```python
class Trainer:
    """
    Класс для обучения и валидации модели.
    
    Функции:
    - Training loop с gradient accumulation
    - Validation с вычислением метрик
    - Сохранение чекпоинтов
    - Early stopping
    - Logging (console + TensorBoard + JSON)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: TrainingConfig
    ):
        ...
    
    def train_epoch(self) -> Dict[str, float]:
        """Обучение одной эпохи."""
        ...
    
    def validate(self) -> Dict[str, float]:
        """Валидация модели."""
        ...
    
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Полный цикл обучения."""
        ...
    
    def save_checkpoint(self, path: str):
        """Сохранение модели и состояния."""
        ...
```

---

## Training Loop

### Алгоритм обучения

```python
def training_step(batch, model, optimizer, scheduler, accumulation_steps):
    """
    Один шаг обучения с gradient accumulation.
    """
    model.train()
    
    # Forward pass
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )
    
    loss = outputs['loss'] / accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step (каждые accumulation_steps)
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return loss.item() * accumulation_steps
```

### Валидация

```python
def validation_step(model, val_loader, id2label):
    """
    Валидация модели и вычисление метрик.
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            total_loss += outputs['loss'].item()
            
            predictions = outputs['logits'].argmax(dim=-1)
            
            # Собираем предсказания (игнорируем -100)
            for pred, label, mask in zip(predictions, batch['labels'], batch['attention_mask']):
                for p, l, m in zip(pred, label, mask):
                    if l != -100 and m == 1:
                        all_predictions.append(id2label[p.item()])
                        all_labels.append(id2label[l.item()])
    
    # Вычисление метрик
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics
```

---

## Метрики

### Основные метрики

| Метрика | Описание | Формула |
|---------|----------|---------|
| **Accuracy** | Общая точность | `correct / total` |
| **Precision** | Точность (macro) | `avg(TP / (TP + FP))` |
| **Recall** | Полнота (macro) | `avg(TP / (TP + FN))` |
| **F1-score** | F1 мера (macro) | `2 * P * R / (P + R)` |

### Дополнительные метрики

- **UPOS Accuracy**: Точность только по UPOS-тегам (без FEATS)
- **Per-class F1**: F1 для каждого класса отдельно
- **Confusion Matrix**: Матрица ошибок для анализа

```python
from sklearn.metrics import classification_report, confusion_matrix

def compute_metrics(y_true, y_pred):
    """
    Вычисляет все метрики для token classification.
    """
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1': report['macro avg']['f1-score'],
        'report': report
    }
```

---

## Логирование и мониторинг

### Консольный вывод

```
Epoch 1/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.8234 | Val Loss: 0.5123
Accuracy: 0.8456 | F1: 0.7892

Epoch 2/3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Train Loss: 0.4521 | Val Loss: 0.4012
Accuracy: 0.8912 | F1: 0.8456
```

### JSON Logs

```json
{
    "experiment": "bert-base-en-morphology",
    "timestamp": "2026-02-06T00:00:00",
    "config": {
        "model": "bert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3
    },
    "history": [
        {
            "epoch": 1,
            "train_loss": 0.8234,
            "val_loss": 0.5123,
            "accuracy": 0.8456,
            "f1": 0.7892
        }
    ],
    "best_metrics": {
        "epoch": 3,
        "accuracy": 0.9123,
        "f1": 0.8756
    }
}
```

### TensorBoard

Логируются:
- `loss/train` — loss на обучении
- `loss/val` — loss на валидации
- `metrics/accuracy` — точность
- `metrics/f1` — F1-score
- `learning_rate` — текущий learning rate

---

## Чекпоинты

### Формат сохранения

```
services/models/en/bert-base/
├── config.json           # Конфигурация модели
├── pytorch_model.bin     # Веса модели
├── tokenizer_config.json # Конфигурация токенизатора
├── vocab.txt             # Словарь токенизатора
├── label2id.json         # Маппинг меток
├── training_args.json    # Гиперпараметры
└── trainer_state.json    # Состояние обучения
```

### Стратегия сохранения

- **Best Model**: Сохраняется модель с лучшим F1 на валидации
- **Last Checkpoint**: Последний чекпоинт для продолжения обучения
- **Every N epochs**: Опционально каждые N эпох

---

## Воспроизводимость

### Фиксация random seed

```python
def set_seed(seed: int = 42):
    """Фиксирует все источники случайности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Документирование эксперимента

Каждый эксперимент должен содержать:
1. Полную конфигурацию гиперпараметров
2. Версии библиотек (`requirements.txt`)
3. Информацию о hardware (GPU, память)
4. Seed для воспроизводимости

---

## Запуск обучения

### Командная строка

```bash
cd c:\Diploma\morphology\services\training

# Обучение BERT
python en_training.py --model bert-base-uncased --epochs 3

# Обучение RoBERTa
python en_training.py --model roberta-base --epochs 3

# Обучение DistilBERT
python en_training.py --model distilbert-base-uncased --epochs 3

# С кастомными параметрами
python en_training.py \
    --model bert-base-uncased \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --output_dir checkpoints/experiment1
```

### Python API

```python
from en_training import train_model

# Обучение модели
results = train_model(
    model_name="bert-base-uncased",
    train_data=train_sentences,
    val_data=val_sentences,
    label2id=label2id,
    config={
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5
    }
)

print(f"Best F1: {results['best_f1']:.4f}")
```

---

## Ожидаемые результаты

### Baseline метрики

На основе литературы и аналогичных экспериментов:

| Модель | Accuracy (ожидаемо) | F1 (ожидаемо) |
|--------|---------------------|---------------|
| BERT-base | 94-96% | 88-92% |
| RoBERTa-base | 95-97% | 89-93% |
| DistilBERT | 93-95% | 86-90% |

### Признаки успешного обучения

1. Loss монотонно убывает на обучении
2. Val loss не растёт (нет overfitting)
3. Метрики растут от эпохи к эпохе
4. Модель корректно предсказывает на примерах

---

## Зависимости

### Python packages

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
tensorboard>=2.14.0
tqdm>=4.65.0
numpy>=1.24.0
```

### Hardware

- **Минимум**: GPU с 8GB VRAM (RTX 3070 / T4)
- **Рекомендуется**: GPU с 16GB VRAM (RTX 4080 / A10)
- **Альтернатива**: Google Colab / Kaggle (бесплатные GPU)

---

## Чекпоинты выполнения

- [ ] Реализован `TokenClassificationDataset`
- [ ] Реализован `MorphologyTagger`
- [ ] Реализован `Trainer` с training loop
- [ ] Настроен логирование (console + TensorBoard + JSON)
- [ ] Обучен BERT-base, зафиксированы метрики
- [ ] Обучен RoBERTa-base, зафиксированы метрики
- [ ] Обучен DistilBERT, зафиксированы метрики
- [ ] Сохранены чекпоинты лучших моделей
- [ ] Документированы гиперпараметры и результаты
- [ ] Код воспроизводим (seed, requirements.txt)

---

## Связь с другими задачами

| Задача | Связь |
|--------|-------|
| TASK-001 | Использует выбранные модели |
| TASK-002 | Использует выбранные датасеты |
| TASK-003 | Использует предобработанные данные |
| **TASK-004** | **Текущая задача** |
| TASK-005 | Использует обученные модели для оценки |
| TASK-009 | Адаптирует пайплайн для казахского языка |

---

## Риски и митигация

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| OOM на GPU | Средняя | Уменьшить batch_size, использовать gradient accumulation |
| Overfitting | Средняя | Early stopping, dropout, weight decay |
| Долгое обучение | Низкая | Использовать DistilBERT для быстрых экспериментов |
| Низкие метрики | Низкая | Проверить данные, увеличить эпохи, тюнинг гиперпараметров |
