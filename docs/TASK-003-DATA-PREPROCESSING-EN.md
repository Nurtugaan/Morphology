# TASK-003: Ағылшын тіліндегі морфологиялық талдау үшін деректерді алдын ала өңдеу

**Күні:** 2026-02-05  
**Мәртебесі:** ✅ Орындалды  
**Тапсырма:** Token classification тапсырмасы үшін деректерді өңдеу пайплайнын іске асыру

---

## Мақсаты

Морфологиялық талдау тапсырмасы бойынша трансформер-модельдерді (BERT, RoBERTa, DistilBERT) оқыту үшін Universal Dependencies деректерді алдын ала өңдеудің қайта жаңғыртылатын пайплайнын әзірлеу.

---

## Пайплайн құрылымы

**Функция**: `parse_conllu_file(filepath)`

### 1. CoNLL-U файлдарын парсингтеу

CoNLL-U — Universal Dependencies аннотацияланған корпустары үшін стандартты формат. Әр жолда табуляциямен бөлінген 10 өріс бар:

| # | Өріс    | Сипаттамасы                         | Қолданылады   |
|---|---------|-------------------------------------|-------------- | 
| 0 | ID      | Сөйлемдегі токен индексі            | ✅ (фильтр)  |
| 1 | FORM    | Сөз формасы (жазылуы бойынша)       | ✅           |
| 2 | LEMMA   | Лемма                               | ✅           |
| 3 | UPOS    | Универсалды POS-тег                 | ✅           |
| 4 | XPOS    | Тілге тән POS-тег                   | ❌           |
| 5 | FEATS   | Морфологиялық белгілер              | ✅           |
| 6 | HEAD    | Тәуелділік басы                     | ❌           |
| 7 | DEPREL  | Тәуелділік түрі                     | ❌           |
| 8 | DEPS    | Жақсартылған тәуелділіктер          | ❌           |
| 9 | MISC    | Басқалары                           | ❌           |

### 2. Токендерді сүзгілеу (фильтрация)

**Функция**: `is_valid_token_id(token_id)`

Келесілер ескерілмейді:
- **Мультисөздер** ("1-2", "19-20" түріндегі ID): "don't" → "do" + "n't" сияқты контрактты формалар
- **Бос түйіндер** ("3.1", "5.2" түріндегі ID): жақсартылған тәуелділіктер (enhanced dependencies)

```
19-20   don't   _       _       _       _       _       _       _       _   ← ескерілмейді
19      do      do      AUX     VBP     ...                                 ← қолданылады
20      n't     not     PART    RB      ...                                 ← қолданылады
```

### 3. Белгілерді (Label) генерациялау

**Функция**: `generate_label(upos, feats)`

Белгі форматы: `UPOS|FEATS`

| UPOS   | FEATS                      | Нәтиже                           |
|--------|----------------------------|----------------------------------|
| VERB   | Tense=Past\|Number=Sing    | VERB\|Tense=Past\|Number=Sing    |
| NOUN   | Number=Plur                | NOUN\|Number=Plur                |
| DET    | Definite=Def\|PronType=Art | DET\|Definite=Def\|PronType=Art  |
| PUNCT  | _                          | PUNCT\|_                         |
| ADJ    | (бос)                      | ADJ\|_                           |

### 4. Белгілер сөздігін жасау

**Функция**: `build_label_vocab(sentences)`

Оқыту деректерінен белгілер сөздігін құрастырады:
- `label2id`: белгі → индекс
- `id2label`: индекс → белгі

### 5. Трансформерлерге дайындау

**Функция**: `prepare_for_training(sentences, label2id)`

Деректерді оқытуға дайын форматта қалыптастырады:

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

## Пайдалану

### Базалық мысал

```python
from en_preprocessing import (
    load_ud_dataset,
    build_label_vocab,
    prepare_for_training
)

# Деректер жиынтығын жүктеу
dataset = load_ud_dataset("services/datasets/english/UD_English-EWT")

# Статистика
print(dataset.stats())

# Белгілер сөздігін құру
label2id, id2label = build_label_vocab(dataset.train)

# Деректерді дайындау
train_data = prepare_for_training(dataset.train, label2id)
dev_data = prepare_for_training(dataset.dev, label2id)
test_data = prepare_for_training(dataset.test, label2id)
```

### Деректер жиынтығын біріктірумен толық пайплайн

```python
from en_preprocessing import load_ud_dataset, combine_datasets

ewt = load_ud_dataset("services/datasets/english/UD_English-EWT")
gum = load_ud_dataset("services/datasets/english/UD_English-GUM")

# Біріктіру
combined = combine_datasets([ewt, gum], name="EWT+GUM")
print(combined.stats())
```

---

## Талданған сөйлем мысалы

Түпнұсқа мәтін:
> Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.

Талданған токендер:

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

## Деректер жиынтығының статистикасы (нақты)

### UD English EWT

| Бөлік (Split) | Сөйлемдер | Токеңдер |
|---------|-------------|---------|
| Train   | 12 544      | 204 577 |
| Dev     | 2 001       | 25 147  |
| Test    | 2 077       | 25 094  |
| **Барлығы** | **16 622** | **254 818** |

### UD English GUM

| Бөлік (Split) | Сөйлемдер | Токеңдер |
|---------|-------------|---------|
| Train   | 10 224      | 177 410 |
| Dev     | 1 575       | 28 119  |
| Test    | 1 464       | 28 397  |
| **Барлығы** | **13 263** | **233 926** |

### Біріктірілген деректер жиынтығы (EWT + GUM)

| Бөлік (Split) | Сөйлемдер | Токеңдер |
|---------|-------------|---------|
| Train   | 22 768      | 381 987 |
| Dev     | 3 576       | 53 266  |
| Test    | 3 541       | 53 491  |
| **Барлығы** | **29 885** | **488 744** |

**Бірегей белгілер (UPOS|FEATS):** 406

---

## Шығыс форматы

### JSONL (ұсынылады)

Әр жол — бір сөйлемнің JSON-объектісі:

```json
{"tokens": ["The", "cat", "sat"], "labels": ["DET|Definite=Def", "NOUN|Number=Sing", "VERB|Tense=Past"], "label_ids": [5, 23, 45]}
```

### JSON

Барлық мысалдар массиві:

```json
[
  {"tokens": [...], "labels": [...], "label_ids": [...]},
  {"tokens": [...], "labels": [...], "label_ids": [...]}
]
```

---

## Трансформерлерді оқытумен байланысы

Дайындалған деректерді келесі кітапханалармен пайдалануға болады:

- **Hugging Face Transformers**: `TokenClassificationDataset`
- **PyTorch**: кастомды `Dataset`
- **datasets library**: `load_dataset("json", data_files=...)`

Hugging Face-пен пайдалану мысалы:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Белгілерді теңестірумен токенизациялау
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

## Қайта жаңғыртылуы

- **Seed**: 42 (кездейсоқ бөлу үшін)
- **Кодтау**: UTF-8
- **Формат**: CoNLL-U 2.x

Барлық операциялар бекітілген seed кезінде детерминирленген.

---

## Жоба файлдары

```
services/
├── preprocessing/
│   └── en_preprocessing.py     # Пайплайнның негізгі модулі (~811 жол)
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

## Демо нұсқаны іске қосу

```bash
cd c:\Diploma\morphology\services\preprocessing
python en_preprocessing.py
```

Демо нұсқа келесілерді шығарады:
1. Жүктелген деректер жиынтығының статистикасы
2. Талданған сөйлем мысалы
3. Белгілерді генерациялау мысалдары
4. Белгілер сөздігі
5. UPOS-тегтер бойынша статистика
6. Трансформерлер үшін дайындалған деректер

---

## Бақылау нүктелері ✅

- [x] CoNLL-U парсингі іске асырылды және тестіленді
- [x] token / lemma / UPOS / FEATS алу
- [x] Бірыңғай белгіні қалыптастыру (UPOS|FEATS) — 406 бірегей белгі
- [x] Мультисөздер мен бос түйіндерді сүзгілеу
- [x] EWT + GUM деректер жиынтығын біріктіру
- [x] train / val / test-ке бөлу
- [x] Код қайта жаңғыртылатын (seed=42) және құжатталған
