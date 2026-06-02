# TASK-008: Қазақ тіліндегі деректерді алдын ала өңдеу пайплайны

## Мақсаты

Token classification (UPOS|FEATS) тапсырмасы үшін қазақ тіліндегі морфологиялық деректерді алдын ала өңдеу пайплайны. 
Ағылшын тіліндегі пайплайнға (TASK-003) ұқсас, деректер жиынтығының 3 конфигурациясымен іске асырылды.

---

## Конфигурациялар

| Конфиг        | Атауы           | Train  | Dev   | Test  | Бірегей белгілер |
|---------------|-----------------|-------:|------:|------:|-----------------:|
| `gold_only`   | UD KTB          | 942    | 104   | 31    | 420              |
| `gold_silver` | KTB + KazDET→UD | 28,335 | 104   | 31    | 586              |
| `nlanu_only`  | KazDET NLANU    | 48,888 | 6,111 | 6,112 | 570              |

### Config 1: `gold_only`

Тек UD Kazakh KTB-нің "алтын стандарты". Dev жиынтығы train-нің 10%-ынан құрылады (seed=42).

- Train: `kk_ktb-ud-train.conllu` файлының 90%-ы
- Dev: `kk_ktb-ud-train.conllu` файлының 10%-ы
- Test: `kk_ktb-ud-test.conllu`

### Config 2: `gold_silver`

Алтын + күміс стандарт. Dev жиынтығы біріктірмес бұрын KTB train-нен бөлінеді.

- Train: KTB train (90%) + `kdt-NLANU-ud.conllu` (27,393 сөйлем)
- Dev: 10% KTB train (таза алтын)
- Test: `kk_ktb-ud-test.conllu` (таза алтын)

### Config 3: `nlanu_only`

KazDET бастапқы NLANU форматында. 80/10/10 автоматты бөлу (seed=42).

- Файл: `kdt-NLANU-0.01.connlu` (~65MB, 61,111 сөйлем)
- FEATS NLANU форматында (`vbTense`, `Poss`, `vbVcPass` және т.б.)

---

## Қолданылуы

```python
from kk_preprocessing import create_kk_pipeline

# Config 1: Тек алтын
dataset, label2id, id2label = create_kk_pipeline("gold_only")

# Config 2: Алтын + Күміс
dataset, label2id, id2label = create_kk_pipeline("gold_silver")

# Config 3: Тек NLANU
dataset, label2id, id2label = create_kk_pipeline("nlanu_only")

# JSONL форматына экспорттаумен
dataset, label2id, id2label = create_kk_pipeline(
    "gold_only", 
    output_dir="../../output/kk"
)
```

### Демо

```bash
python services/preprocessing/kk_preprocessing.py
```

---

## Деректер форматы

### Белгілер

Форматы: `UPOS|FEATS` (ағылшын пайплайнындағыдай).

Мысалдар (Config 1–2, UD форматы):
```
NOUN|Case=Gen
VERB|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin
PUNCT|_
```

Мысалдар (Config 3, NLANU форматы):
```
NOUN|Poss=3
VERB|vbType=Adv
CONJ|_
```

### Шығыс JSONL

```json
{
  "tokens": ["Экономиканың", "басқа", "салалары"],
  "labels": ["NOUN|Case=Gen", "ADJ|_", "NOUN|Case=Nom|Number=Plur|..."],
  "label_ids": [42, 1, 128],
  "sentence_id": "...",
  "text": "..."
}
```

---

## Жоба файлдары

| Файл | Сипаттамасы |
|------|----------|
| `services/preprocessing/kk_preprocessing.py` | Негізгі пайплайн |
| `services/preprocessing/en_preprocessing.py` | Базалық компоненттер (импортталады) |
| `services/datasets/kazakh/UD/kk_ktb-ud-train.conllu` | KTB train |
| `services/datasets/kazakh/UD/kk_ktb-ud-test.conllu` | KTB test |
| `services/datasets/kazakh/UD/kdt-NLANU-ud.conllu` | KazDET→UD (күміс) |
| `services/datasets/kazakh/NLANU/kdt-NLANU-0.01.connlu` | KazDET NLANU |

---

## Мәртебесі

- [x] `kk_preprocessing.py` іске асырылды
- [x] Деректер жиынтығының 3 конфигурациясы
- [x] Верификация: барлық сплиттер жүктеледі, tokens == labels
- [x] Құжаттама
