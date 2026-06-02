# TASK-004: Ағылшын тіліндегі морфологиялық талдау модельдерін оқыту

**Күні:** 2026-02-06  
**Мәртебесі:** ✅ Орындалды  
**Тапсырма:** Таңдалған модельдерді UD English EWT + GUM біріктірілген корпусында оқыту және метрикаларды бекіту

---

## Мақсаты

1. **Training loop іске асыру** — трансформер-модельдерді token classification тапсырмасына оқыту үшін
2. **5 модельді оқыту** (BERT-base, RoBERTa-base, DistilBERT, ALBERT-base, Flair) ағылшын тіліндегі деректер жиынтығында
3. **Гиперпараметрлерді бекіту** және эксперименттердің қайта жаңғыртылуын қамтамасыз ету
4. **Baseline-нәтижелерді алу** — кейіннен қазақ тіліндегі модельдермен салыстыру үшін

---

## Кіріс деректері

Деректер TASK-003 (алдын ала өңдеу) тапсырмасынан алынады:

| Дереккөз | Сипаттамасы |
|----------|----------|
| `en_preprocessing.py` | Деректерді жүктеу және алдын ала өңдеу модулі |
| `UD_English-EWT` | Негізгі оқыту деректер жиынтығы (204 577 токен) |
| `UD_English-GUM` | Қосымша деректер жиынтығы (177 410 токен) |
| **Біріктірілген** | **381 987 токен / 406 бірегей белгі** |

---

## Оқытуға арналған модельдер

TASK-001 негізінде келесі модельдер таңдалды:

| Модель | Архитектура | Параметрлері | HuggingFace ID | Негіздеме |
|--------|-------------|-----------|----------------|------------|
| **BERT-base** | BERT | 109M | `bert-base-uncased` | Классикалық baseline |
| **RoBERTa-base** | RoBERTa | 125M | `roberta-base` | Жақсартылған алдын ала оқыту |
| **DistilBERT** | BERT (distilled) | 66M | `distilbert-base-uncased` | Жылдам және жеңіл |
| **ALBERT-base** | ALBERT | 12M | `albert-base-v2` | Жадқа үнемді |
| **Flair** | BiLSTM+CRF | 63M | `flair` (кітапхана) | Альтернативті архитектура |

> **Ескертпе:** Flair жеке `flair` кітапханасын пайдаланады және оқытуға ерекше тәсілді талап етеді (HuggingFace Transformers емес).

---

## Модель архитектурасы

### Схема (трансформерлер)
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
│   Шығыс: [batch_size, seq_len, hidden_size]            │
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

### Схема (Flair)
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Stacked Flair Embeddings (fwd + bwd)           │
│   LanguageModel(Embedding(300,100) → LSTM(100,2048))   │
│         Шығыс: [batch_size, seq_len, 4096]             │
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

### Субтокеңдерді теңестіру (Label Alignment)

Субсөздік токенизация кезінде:
- Белгі сөздің **тек бірінші субтокеніне** тағайындалады
- Барлық келесі субтокеңдер `-100` белгісін алады (loss есептеу кезінде ескерілмейді)
- Арнайы токеңдер де (`[CLS]`, `[SEP]`, `[PAD]`) `-100` алады

```python
# Теңестіру мысалы
Word:       "unhappiness"
Subtokens:  ["un", "##happy", "##ness"]
Labels:     [LABEL_ID, -100, -100]
```

---

## Гиперпараметрлер

### Трансформер-модельдер

| Параметр               | BERT | RoBERTa | DistilBERT | ALBERT |
|------------------------|------|---------|------------|--------|
| `learning_rate`        | 2e-5 | 2e-5    | 3e-5       | 2e-5   |
| `batch_size`           | 32   | 32      | 32         | 64     |
| `epochs`               | 5    | 5       | 6          | 6      |
| `max_length`           | 128  | 128     | 128        | 128    |
| `warmup_ratio`         | 0.1  | 0.1     | 0.1        | 0.1    |
| `weight_decay`         | 0.01 | 0.01    | 0.01       | 0.01   |
| `gradient_accumulation`| 2    | 2       | 2          | 2      |
| `dropout`              | 0.1  | 0.1     | 0.1        | 0.1    |
| `seed`                 | 42   | 42      | 42         | 42     |

### Flair

| Параметр               | Мәні                     |
|------------------------|--------------------------|
| `learning_rate`        | 0.1 (SGD)                |
| `mini_batch_size`      | 16                       |
| `max_epochs`           | 5                        |
| `embeddings`           | Flair forward + backward |
| `hidden_size`          | 256                      |
| `rnn_layers`           | 2                        |
| `use_crf`              | True                     |

### Жалпы параметрлер

- **Optimizer**: AdamW (трансформерлер) / SGD (Flair)
- **Scheduler**: Linear with warmup (трансформерлер) / AnnealOnPlateau (Flair)
- **Loss**: CrossEntropyLoss (`ignore_index=-100` индексімен) (трансформерлер) / ViterbiLoss (Flair)
- **Device**: CUDA (GPU)

---

## Код құрылымы

```
services/training/
├── __init__.py           # Экспорттар
├── config.py             # TrainingConfig — гиперпараметрлер конфигурациясы
├── dataset.py            # TokenClassificationDataset — PyTorch Dataset
├── model.py              # MorphologyTagger — трансформерлердің үстіндегі қабықша
├── trainer.py            # Trainer — training loop, валидация, early stopping
├── train_base.py         # Оқытуды бастау скрипті
└── models/               # Нақты модельдерді оқыту скрипттері
```

---

## Оқытудың нақты нәтижелері

### Оқыту динамикасы (эпохалар бойынша val F1)

| Эпоха | BERT | RoBERTa  | DistilBERT | ALBERT | Flair* |
|-------|------|----------|------------|--------|--------|
| 1     | 0.3658 | 0.3448 | 0.3844     | 0.3413 | 0.8573 |
| 2     | 0.4858 | 0.4755 | 0.4889     | 0.4729 | 0.9077 |
| 3     | 0.5085 | 0.5169 | 0.5365     | 0.5097 | 0.9234 |
| 4     | 0.5371 | 0.5291 | 0.5569     | 0.5195 | 0.9306 |
| 5     | 0.5468 | 0.5306 | 0.5645     | 0.5357 | 0.9366 |
| 6     | —      | —      | 0.5768     | 0.5463 | —      |

*\* Flair үшін оқыту кезінде Micro-F1 (Accuracy) бақыланды, 
ал басқа модельдер үшін Macro-F1 көрсетілген.*

### Loss динамикасы (ең жақсы модель — DistilBERT)

| Эпоха | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 1.9124 | 0.2901 |
| 2 | 0.2218 | 0.2227 |
| 3 | 0.1579 | 0.2062 |
| 4 | 0.1273 | 0.1997 |
| 5 | 0.1087 | 0.1994 |
| 6 | 0.0982 | 0.2005 |

Train loss-тың тұрақты төмендеуі байқалады, 
айтарлықтай overfitting жоқ (val loss 4-ші эпохадан бастап тұрақталады).

### Қорытынды нәтижелер (Test Set)

| Модель | Accuracy | Precision | Recall | Macro-F1 | Оқыту уақыты |
|--------|----------|-----------|--------|----------|---------------|
| **DistilBERT** | 94.89% | 59.92% | 59.32% | **58.84%** | 25.9 мин |
| BERT-base | 94.83% | 57.83% | 57.31% | 56.79% | 38.6 мин |
| ALBERT-base | 94.64% | 57.71% | 56.11% | 56.09% | 49.0 мин |
| RoBERTa-base | 94.89% | 56.45% | 56.92% | 55.90% | 38.8 мин |
| Flair | 93.95% | 54.92% | 51.21% | 52.08% | ~30 мин |

---

## Сақталған модельдер

Барлық оқытылған модельдер `services/models/en/` ішінде сақталған:

```
services/models/en/
├── bert-base_model/       # BERT-base (best F1: 0.5468, test F1: 0.5679)
├── roberta-base_model/    # RoBERTa-base (best F1: 0.5306, test F1: 0.5590)
├── distilbert_model/      # DistilBERT (best F1: 0.5768, test F1: 0.5884)
├── albert-base_model/     # ALBERT-base (best F1: 0.5463, test F1: 0.5609)
└── flair_model/           # Flair (micro F1: 0.9395, macro F1: 0.5208)
```

Әр директорияда мыналар бар:
- Модель салмақтары (`pytorch_model.bin` / `model.safetensors`)
- Конфигурация (`config.json`)
- Токенизатор (`tokenizer_config.json`, `vocab.txt`)
- Белгілер картасы (`label2id.json`)
- Оқыту тарихы (`training_history.json`)

---

## Метрикалар

### Негізгі метрикалар

| Метрика | Сипаттамасы | Формула |
|---------|----------|---------|
| **Accuracy** | Жалпы дәлдік | `correct / total` |
| **Precision** | Дәлдік (macro) | `avg(TP / (TP + FP))` |
| **Recall** | Толықтық (macro) | `avg(TP / (TP + FN))` |
| **F1-score** | F1 өлшемі (macro) | `2 * P * R / (P + R)` |

---

## Қайта жаңғыртылуы

```python
def set_seed(seed: int = 42):
    """Барлық кездейсоқ дереккөздерді бекітеді."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Оқытуды бастау

```bash
cd c:\Diploma\morphology\services\training

# BERT оқыту
python train_base.py --model bert-base-uncased --epochs 5

# RoBERTa оқыту
python train_base.py --model roberta-base --epochs 5

# DistilBERT оқыту
python train_base.py --model distilbert-base-uncased --epochs 6

# ALBERT оқыту
python train_base.py --model albert-base-v2 --epochs 6 --batch_size 64
```

---

## Тәуелділіктер

```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
tqdm>=4.65.0
numpy>=1.24.0
flair>=0.15.1
```

**Hardware**: CUDA қолдауы бар GPU (Google Colab / Kaggle пайдаланылды)

---

## Басқа тапсырмалармен байланысы

| Тапсырма | Байланысы |
|--------|-------|
| TASK-001 | Таңдалған модельдерді пайдаланады |
| TASK-002 | Таңдалған деректер жиынтығын пайдаланады |
| TASK-003 | Алдын ала өңделген деректерді пайдаланады |
| **TASK-004** | **Ағымдағы тапсырма** |
| TASK-005 | Бағалау үшін оқытылған модельдерді пайдаланады |
| TASK-009 | Пайплайнды қазақ тіліне бейімдейді |

---

## Бақылау нүктелері ✅

- [x] `TokenClassificationDataset` іске асырылды (PyTorch Dataset)
- [x] `MorphologyTagger` іске асырылды (трансформерлер негізінде)
- [x] `Trainer` іске асырылды (training loop және gradient accumulation)
- [x] Логикалық жазбалар реттелді (console + JSON)
- [x] BERT-base оқытылды — Test F1: 56.79%, уақыты: 38.6 мин
- [x] RoBERTa-base оқытылды — Test F1: 55.90%, уақыты: 38.8 мин
- [x] DistilBERT оқытылды — Test F1: 58.84%, уақыты: 25.9 мин
- [x] ALBERT-base оқытылды — Test F1: 56.09%, уақыты: 49.0 мин
- [x] Flair (BiLSTM+CRF) оқытылды — Test Macro-F1: 52.08%, уақыты: ~30 мин
- [x] Ең жақсы модельдердің чекпоинттері сақталды
- [x] Гиперпараметрлер мен нәтижелер құжатталды
- [x] Код қайта жаңғыртылатын (seed=42, requirements.txt)
