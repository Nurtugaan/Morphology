# TASK-004: Обучение моделей морфологического анализа на английском языке

**Дата:** 2026-02-06  
**Статус:** ✅ Выполнено  
**Задача:** Обучить выбранные модели на объединённом корпусе UD English EWT + GUM и зафиксировать метрики

---

## Цель

1. **Реализовать training loop** для обучения трансформер-моделей на задачу token classification
2. **Обучить 5 моделей** (BERT-base, RoBERTa-base, DistilBERT, ALBERT-base, Flair) на englsh датасете
3. **Зафиксировать гиперпараметры** и обеспечить воспроизводимость экспериментов
4. **Получить baseline-результаты** для последующего сравнения с казахскими моделями

---

## Входные данные

Данные поступают из TASK-003 (предобработка):

| Источник | Описание |
|----------|----------|
| `en_preprocessing.py` | Модуль загрузки и предобработки данных |
| `UD_English-EWT` | Основной обучающий датасет (204 577 токенов) |
| `UD_English-GUM` | Дополнительный датасет (177 410 токенов) |
| **Объединённый** | **381 987 токенов / 406 уникальных меток** |

---

## Модели для обучения

На основании TASK-001 выбраны следующие модели:

| Модель | Архитектура | Параметры | HuggingFace ID | Обоснование |
|--------|-------------|-----------|----------------|------------|
| **BERT-base** | BERT | 109M | `bert-base-uncased` | Классический baseline |
| **RoBERTa-base** | RoBERTa | 125M | `roberta-base` | Улучшенная предобученность |
| **DistilBERT** | BERT (distilled) | 66M | `distilbert-base-uncased` | Быстрая и лёгкая |
| **ALBERT-base** | ALBERT | 12M | `albert-base-v2` | Экономная по памяти |
| **Flair** | BiLSTM+CRF | 63M | `flair` (библиотека) | Альтернативная архитектура |

> **Примечание:** Flair использует отдельную библиотеку `flair` и требует особого подхода к обучению (не HuggingFace Transformers).

---

## Архитектура модели

### Схема (трансформеры)

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
│         (BERT / RoBERTa / DistilBERT / ALBERT)         │
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
│         (hidden_size → num_labels=406)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Output Logits                          │
│     → CrossEntropyLoss (ignore_index=-100)             │
│     → Predicted Labels                                  │
└─────────────────────────────────────────────────────────┘
```

### Схема (Flair)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Stacked Flair Embeddings (fwd + bwd)           │
│   LanguageModel(Embedding(300,100) → LSTM(100,2048))   │
│         Выход: [batch_size, seq_len, 4096]             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│        BiLSTM (4096 → 256×2, 2 layers, dropout=0.5)   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             Linear (512 → 326) + CRF                   │
│             ViterbiLoss / Viterbi Decode                │
└─────────────────────────────────────────────────────────┘
```

### Выравнивание субтокенов (Label Alignment)

При субсловной токенизации:
- Метка назначается **только первому субтокену** слова
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

### Трансформер-модели

| Параметр | BERT | RoBERTa | DistilBERT | ALBERT |
|----------|------|---------|------------|--------|
| `learning_rate` | 2e-5 | 2e-5 | 3e-5 | 2e-5 |
| `batch_size` | 32 | 32 | 32 | 64 |
| `epochs` | 5 | 5 | 6 | 6 |
| `max_length` | 128 | 128 | 128 | 128 |
| `warmup_ratio` | 0.1 | 0.1 | 0.1 | 0.1 |
| `weight_decay` | 0.01 | 0.01 | 0.01 | 0.01 |
| `gradient_accumulation` | 2 | 2 | 2 | 2 |
| `dropout` | 0.1 | 0.1 | 0.1 | 0.1 |
| `seed` | 42 | 42 | 42 | 42 |

### Flair

| Параметр | Значение |
|----------|----------|
| `learning_rate` | 0.1 (SGD) |
| `mini_batch_size` | 16 |
| `max_epochs` | 5 |
| `embeddings` | Flair forward + backward |
| `hidden_size` | 256 |
| `rnn_layers` | 2 |
| `use_crf` | True |

### Общие параметры

- **Optimizer**: AdamW (трансформеры) / SGD (Flair)
- **Scheduler**: Linear with warmup (трансформеры) / AnnealOnPlateau (Flair)
- **Loss**: CrossEntropyLoss с `ignore_index=-100` (трансформеры) / ViterbiLoss (Flair)
- **Device**: CUDA (GPU)

---

## Структура кода

```
services/training/
├── __init__.py           # Экспорты
├── config.py             # TrainingConfig — конфигурация гиперпараметров
├── dataset.py            # TokenClassificationDataset — PyTorch Dataset
├── model.py              # MorphologyTagger — обёртка над трансформерами
├── trainer.py            # Trainer — training loop, валидация, early stopping
├── train_base.py         # Скрипт запуска обучения
└── models/               # Скрипты обучения конкретных моделей
```

---

## Фактические результаты обучения

### Динамика обучения (val F1 по эпохам)

| Эпоха | BERT | RoBERTa | DistilBERT | ALBERT | Flair |
|-------|------|---------|------------|--------|-------|
| 1 | 0.3658 | 0.3448 | 0.3844 | 0.3413 | — |
| 2 | 0.4858 | 0.4755 | 0.4889 | 0.4729 | — |
| 3 | 0.5085 | 0.5169 | 0.5365 | 0.5097 | — |
| 4 | 0.5371 | 0.5291 | 0.5569 | 0.5195 | — |
| 5 | 0.5468 | 0.5306 | 0.5645 | 0.5357 | — |
| 6 | — | — | 0.5768 | 0.5463 | — |

### Итоговые результаты (Test Set)

| Модель | Accuracy | Precision | Recall | Macro-F1 | Время обучения |
|--------|----------|-----------|--------|----------|---------------|
| **DistilBERT** | 94.89% | 59.92% | 59.32% | **58.84%** | 25.9 мин |
| BERT-base | 94.83% | 57.83% | 57.31% | 56.79% | 38.6 мин |
| ALBERT-base | 94.64% | 57.71% | 56.11% | 56.09% | 49.0 мин |
| RoBERTa-base | 94.89% | 56.45% | 56.92% | 55.90% | 38.8 мин |
| Flair | 93.95% | 54.92% | 51.21% | 52.08% | ~30 мин |

### Динамика loss (лучшая модель — DistilBERT)

| Эпоха | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.9124 | 0.2901 |
| 2 | 0.2218 | 0.2227 |
| 3 | 0.1579 | 0.2062 |
| 4 | 0.1273 | 0.1997 |
| 5 | 0.1087 | 0.1994 |
| 6 | 0.0982 | 0.2005 |

> Наблюдается стабильное снижение train loss без значительного overfitting (val loss стабилизируется с эпохи 4).

---

## Сохранённые модели

Все обученные модели сохранены в `services/models/en/`:

```
services/models/en/
├── bert-base_model/       # BERT-base (best F1: 0.5468, test F1: 0.5679)
├── roberta-base_model/    # RoBERTa-base (best F1: 0.5306, test F1: 0.5590)
├── distilbert_model/      # DistilBERT (best F1: 0.5768, test F1: 0.5884)
├── albert-base_model/     # ALBERT-base (best F1: 0.5463, test F1: 0.5609)
└── flair_model/           # Flair (micro F1: 0.9395, macro F1: 0.5208)
```

Каждая директория содержит:
- Веса модели (`pytorch_model.bin` / `model.safetensors`)
- Конфигурацию (`config.json`)
- Токенизатор (`tokenizer_config.json`, `vocab.txt`)
- Маппинг меток (`label2id.json`)
- Историю обучения (`training_history.json`)

---

## Метрики

### Основные метрики

| Метрика | Описание | Формула |
|---------|----------|---------|
| **Accuracy** | Общая точность | `correct / total` |
| **Precision** | Точность (macro) | `avg(TP / (TP + FP))` |
| **Recall** | Полнота (macro) | `avg(TP / (TP + FN))` |
| **F1-score** | F1 мера (macro) | `2 * P * R / (P + R)` |

---

## Воспроизводимость

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

---

## Запуск обучения

```bash
cd c:\Diploma\morphology\services\training

# Обучение BERT
python train_base.py --model bert-base-uncased --epochs 5

# Обучение RoBERTa
python train_base.py --model roberta-base --epochs 5

# Обучение DistilBERT
python train_base.py --model distilbert-base-uncased --epochs 6

# Обучение ALBERT
python train_base.py --model albert-base-v2 --epochs 6 --batch_size 64
```

---

## Зависимости

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
tqdm>=4.65.0
numpy>=1.24.0
flair>=0.15.1
```

**Hardware**: GPU с CUDA (использован Google Colab / Kaggle)

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

## Чекпоинты ✅

- [x] Реализован `TokenClassificationDataset` (PyTorch Dataset)
- [x] Реализован `MorphologyTagger` (обёртка над трансформерами)
- [x] Реализован `Trainer` с training loop и gradient accumulation
- [x] Настроено логирование (console + JSON)
- [x] Обучен BERT-base — Test F1: 56.79%, время: 38.6 мин
- [x] Обучен RoBERTa-base — Test F1: 55.90%, время: 38.8 мин
- [x] Обучен DistilBERT — Test F1: 58.84%, время: 25.9 мин
- [x] Обучен ALBERT-base — Test F1: 56.09%, время: 49.0 мин
- [x] Обучен Flair (BiLSTM+CRF) — Test Macro-F1: 52.08%, время: ~30 мин
- [x] Сохранены чекпоинты лучших моделей
- [x] Документированы гиперпараметры и результаты
- [x] Код воспроизводим (seed=42, requirements.txt)
