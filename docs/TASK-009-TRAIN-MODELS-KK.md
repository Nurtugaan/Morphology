# TASK-009: Қазақ тіліндегі морфологиялық талдау модельдерін оқыту

Қазақ тілі үшін оқыту пайплайнын (training pipeline) бейімдеу. Деректер жиынтығының 3 конфигурациясы бойынша модельді оқыту.

---

## 1. Модельдер мен конфигурациялар

Біз келесі модельдерді 3 түрлі деректер жиынтығында оқытамыз:

*   **Деректер жиынтықтары**:
    1.  `gold_only`: UD Kazakh KTB (алтын стандарт)
    2.  `gold_silver`: KTB + KazDET→UD (күміс стандарт)
    3.  `nlanu_only`: KazDET NLANU (бастапқы формат)

*   **Модельдер**:

| ID |     Атауы     | HuggingFace базалық моделі           |   Ерекшеліктері   |
|----|---------------|--------------------------------------|-------------------|
| 1 | **mBERT**      | `bert-base-multilingual-cased`       | Көптілді baseline |
| 3 | **DistilBERT** | `distilbert-base-multilingual-cased` | Жылдам және жеңіл |

---

## 2. Оқытуды бастау

Оқытуды бастау үшін `services/training/models/` ішіндегі скрипттерді пайдаланыңыз:

### mBERT
```bash
python -m services.training.models.train_kk_mbert --config gold_silver --epochs 6
```

### Kaz-RoBERTa
> [!NOTE]
> Жадты үнемдеу үшін `batch_size=16` және `gradient_accumulation=4` параметрлерін пайдаланады.

```bash
python -m services.training.models.train_kk_roberta --config gold_silver
```

### DistilBERT
```bash
python -m services.training.models.train_kk_distilbert --config gold_silver
```

### XLM-RoBERTa
```bash
python -m services.training.models.train_kk_xlmr --config gold_silver
```

### Flair
> [!WARNING]
> `flair` кітапханасының орнатылуын талап етеді (`pip install flair`). Деректердің басқа құрылымын пайдаланады.

```bash
python -m services.training.models.train_kk_flair --config gold_silver --embeddings flair
```

---

## 3. Нәтижелер мен метрикалар

Оқыту нәтижелері `services/models/kk/{model}-{config}/` ішінде сақталады.
Әр директорияда:
- `pytorch_model.bin` (салмақтар)
- `config.json`
- `tokenizer.json`
- `training_history.json` (эпохалар бойынша метрикалар журналы)
- `label2id.json`

| Модель \ Деректер жиынтығы | `gold_only` (F1) | `gold_silver` (F1) | `nlanu_only` (F1) |
|------------------|------------------|--------------------|-------------------|
| mBERT | Анықталуда | Анықталуда | Анықталуда |
| Kaz-RoBERTa | Анықталуда | Анықталуда | Анықталуда |
| DistilBERT | Анықталуда | Анықталуда | Анықталуда |
| XLM-RoBERTa | Анықталуда | Анықталуда | Анықталуда |
| Flair | Анықталуда | Анықталуда | Анықталуда |

---

## 4. Код құрылымы

*   `services/training/kk_config.py`: Гиперпараметрлер конфигурациялары.
*   `services/training/train_kk_base.py`: Оқытудың базалық логикасы (Transformers).
*   `services/training/models/train_kk_*.py`: Іске қосылатын скрипттер.
*   `services/preprocessing/kk_preprocessing.py`: Деректерді дайындау.
