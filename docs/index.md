## Пайплайн қадамдары

| № | Қадам                     | Функция                | Негізгі сипат                                           |
| - | ------------------------- | ---------------------- | ------------------------------------------------------- |
| 1 | CoNLL-U парсинг           | `parse_conllu_file`    | 10 өрістен тек токен, lemma, UPOS, FEATS пайдаланылады  |
| 2 | Токен сүзгілеу            | `is_valid_token_id`    | Мультисөздер (1-2) және бос түйіндер (3.1) ескерілмейді |
| 3 | Белгілер генерациясы      | `generate_label`       | Формат: `UPOSk/ FEATS`                                  |
| 4 | Белгілер сөздігі          | `build_label_vocab`    | `label2id` / `id2label` құрылады                        |
| 5 | Трансформерлерге дайындау | `prepare_for_training` | Токендер, label, label_ids, sentence_id, text           |

---

## Белгі генерациясы мысалы

| UPOS  | FEATS                     | Label                         |
| ----- | ------------------------- | ----------------------------- |
| VERB  | Tense=Past|Number=Sing    | VERB|Tense=Past|Number=Sing   |
| NOUN  | Number=Plur               | NOUN|Number=Plur              |
| DET   | Definite=Def|PronType=Art | DET|Definite=Def|PronType=Art |
| ADJ   | (бос)                     | ADJ|_                         |
| PUNCT | _                         | PUNCT|_                       |

---

## Деректер жиынтығы статистикасы

| Dataset     | Split | Сөйлем | Токен   |
| ----------- | ----- | ------ | ------- |
| **EWT**     | Train | 12 544 | 204 577 |
|             | Dev   | 2 001  | 25 147  |
|             | Test  | 2 077  | 25 094  |
| **GUM**     | Train | 10 224 | 177 410 |
|             | Dev   | 1 575  | 28 119  |
|             | Test  | 1 464  | 28 397  |
| **EWT+GUM** | Train | 22 768 | 381 987 |
|             | Dev   | 3 576  | 53 266  |
|             | Test  | 3 541  | 53 491  |

**Бірегей белгілер:** 406 (UPOS|FEATS)











