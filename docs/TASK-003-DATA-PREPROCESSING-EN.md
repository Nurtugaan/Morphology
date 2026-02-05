# TASK-003: English Morphological Data Preprocessing Pipeline

## Обзор

Данный пайплайн предназначен для предобработки данных в задаче морфологического анализа (token classification) на английском языке. Пайплайн использует датасеты Universal Dependencies и подготавливает данные для обучения трансформер-моделей (BERT, RoBERTa, DistilBERT).

---

## Структура пайплайна

### 1. Парсинг CoNLL-U файлов

**Функция**: `parse_conllu_file(filepath)`

CoNLL-U — стандартный формат для аннотированных корпусов Universal Dependencies. Каждая строка содержит 10 полей, разделённых табуляцией:

| # | Поле    | Описание                            | Используется |
|---|---------|-------------------------------------|--------------|
| 0 | ID      | Индекс токена в предложении         | ✅ (фильтр)  |
| 1 | FORM    | Форма слова (as written)            | ✅           |
| 2 | LEMMA   | Лемма                               | ✅           |
| 3 | UPOS    | Универсальный POS-тег               | ✅           |
| 4 | XPOS    | Language-specific POS-тег           | ❌           |
| 5 | FEATS   | Морфологические признаки            | ✅           |
| 6 | HEAD    | Голова зависимости                  | ❌           |
| 7 | DEPREL  | Тип зависимости                     | ❌           |
| 8 | DEPS    | Enhanced dependencies               | ❌           |
| 9 | MISC    | Прочее                              | ❌           |

### 2. Фильтрация токенов

**Функция**: `is_valid_token_id(token_id)`

Игнорируются:
- **Мультислова** (ID вида "1-2", "19-20"): контрактные формы типа "don't" → "do" + "n't"
- **Пустые узлы** (ID вида "3.1", "5.2"): enhanced dependencies

```
19-20   don't   _       _       _       _       _       _       _       _   ← игнорируется
19      do      do      AUX     VBP     ...                                 ← используется
20      n't     not     PART    RB      ...                                 ← используется
```

### 3. Генерация меток

**Функция**: `generate_label(upos, feats)`

Формат метки: `UPOS|FEATS`

| UPOS   | FEATS                      | Результат                        |
|--------|----------------------------|----------------------------------|
| VERB   | Tense=Past\|Number=Sing    | VERB\|Tense=Past\|Number=Sing    |
| NOUN   | Number=Plur                | NOUN\|Number=Plur                |
| DET    | Definite=Def\|PronType=Art | DET\|Definite=Def\|PronType=Art  |
| PUNCT  | _                          | PUNCT\|_                         |
| ADJ    | (пусто)                    | ADJ\|_                           |

### 4. Создание словаря меток

**Функция**: `build_label_vocab(sentences)`

Строит словарь меток из обучающих данных:
- `label2id`: метка → индекс
- `id2label`: индекс → метка

### 5. Подготовка для трансформеров

**Функция**: `prepare_for_training(sentences, label2id)`

Формирует данные в формате, готовом для обучения:

```python
{
    'tokens': ['The', 'cat', 'sat'],
    'labels': ['DET|Definite=Def', 'NOUN|Number=Sing', 'VERB|Tense=Past'],
    'label_ids': [5, 23, 45],
    'sentence_id': '...',
    'text': 'The cat sat'
}
```

---

## Использование

### Базовый пример

```python
from en_preprocessing import (
    load_ud_dataset,
    build_label_vocab,
    prepare_for_training
)

# Загрузка датасета
dataset = load_ud_dataset("services/datasets/english/UD_English-EWT")

# Статистика
print(dataset.stats())

# Построение словаря меток
label2id, id2label = build_label_vocab(dataset.train)

# Подготовка данных
train_data = prepare_for_training(dataset.train, label2id)
dev_data = prepare_for_training(dataset.dev, label2id)
test_data = prepare_for_training(dataset.test, label2id)
```

### Полный пайплайн с экспортом

```python
from en_preprocessing import create_full_pipeline

# Создание полного пайплайна с экспортом
dataset, label2id, id2label = create_full_pipeline(
    ewt_dir="services/datasets/english/UD_English-EWT",
    gum_dir="services/datasets/english/UD_English-GUM",
    output_dir="output/preprocessed",
    combine=True  # Объединить датасеты
)
```

### Объединение датасетов

```python
from en_preprocessing import load_ud_dataset, combine_datasets

ewt = load_ud_dataset("services/datasets/english/UD_English-EWT")
gum = load_ud_dataset("services/datasets/english/UD_English-GUM")

# Объединение
combined = combine_datasets([ewt, gum], name="EWT+GUM")
print(combined.stats())
```

---

## Пример распарсенного предложения

Исходный текст:
> Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.

Распарсенные токены:

| Token     | Lemma     | UPOS   | FEATS                                      | Label                                        |
|-----------|-----------|--------|--------------------------------------------|----------------------------------------------|
| Al        | Al        | PROPN  | Number=Sing                                | PROPN\|Number=Sing                           |
| -         | -         | PUNCT  | _                                          | PUNCT\|_                                     |
| Zaman     | Zaman     | PROPN  | Number=Sing                                | PROPN\|Number=Sing                           |
| :         | :         | PUNCT  | _                                          | PUNCT\|_                                     |
| American  | American  | ADJ    | Degree=Pos                                 | ADJ\|Degree=Pos                              |
| forces    | force     | NOUN   | Number=Plur                                | NOUN\|Number=Plur                            |
| killed    | kill      | VERB   | Mood=Ind\|Number=Plur\|Person=3\|Tense=Past\|VerbForm=Fin | VERB\|Mood=Ind\|Number=Plur\|Person=3\|Tense=Past\|VerbForm=Fin |
| Shaikh    | Shaikh    | PROPN  | Number=Sing                                | PROPN\|Number=Sing                           |
| ...       | ...       | ...    | ...                                        | ...                                          |

---

## Статистика датасетов

### UD English EWT

| Split   | Предложений | Токенов  |
|---------|-------------|----------|
| Train   | ~12,500     | ~205,000 |
| Dev     | ~2,000      | ~25,000  |
| Test    | ~2,000      | ~25,000  |

### UD English GUM

| Split   | Предложений | Токенов  |
|---------|-------------|----------|
| Train   | ~6,500      | ~135,000 |
| Dev     | ~1,000      | ~18,000  |
| Test    | ~1,000      | ~18,000  |

---

## Выходной формат

### JSONL (рекомендуется)

Каждая строка — JSON-объект одного предложения:

```json
{"tokens": ["The", "cat", "sat"], "labels": ["DET|Definite=Def", "NOUN|Number=Sing", "VERB|Tense=Past"], "label_ids": [5, 23, 45]}
```

### JSON

Массив всех примеров:

```json
[
  {"tokens": [...], "labels": [...], "label_ids": [...]},
  {"tokens": [...], "labels": [...], "label_ids": [...]}
]
```

---

## Связь с обучением трансформеров

Подготовленные данные можно использовать с библиотеками:

- **Hugging Face Transformers**: `TokenClassificationDataset`
- **PyTorch**: кастомный `Dataset`
- **datasets library**: `load_dataset("json", data_files=...)`

Пример использования с Hugging Face:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Токенизация с выравниванием меток
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True
    )
    
    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # Special tokens
        elif word_idx != previous_word_idx:
            labels.append(example["label_ids"][word_idx])
        else:
            labels.append(-100)  # Subword tokens
        previous_word_idx = word_idx
    
    tokenized["labels"] = labels
    return tokenized
```

---

## Воспроизводимость

- **Seed**: 42 (для случайного разделения)
- **Кодировка**: UTF-8
- **Формат**: CoNLL-U 2.x

Все операции детерминированы при фиксированном seed.

---

## Файлы проекта

```
services/
├── preprocessing/
│   ├── en_preprocessing.py     # Основной модуль пайплайна
│   └── utils.py                # Общие утилиты (если нужны)
│
└── datasets/
    └── english/
        ├── UD_English-EWT/
        │   ├── en_ewt-ud-train.conllu
        │   ├── en_ewt-ud-dev.conllu
        │   └── en_ewt-ud-test.conllu
        │
        └── UD_English-GUM/
            ├── en_gum-ud-train.conllu
            ├── en_gum-ud-dev.conllu
            └── en_gum-ud-test.conllu
```

---

## Запуск демо

```bash
cd c:\Diploma\morphology\services\preprocessing
python en_preprocessing.py
```

Демо выводит:
1. Статистику загруженных датасетов
2. Пример распарсенного предложения
3. Примеры генерации меток
4. Словарь меток
5. Статистику по UPOS-тегам
6. Подготовленные данные для трансформеров
