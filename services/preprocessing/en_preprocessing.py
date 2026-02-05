"""
English Morphological Data Preprocessing Pipeline (TASK-003)

Пайплайн предобработки данных для задачи морфологического анализа (token classification)
на английском языке с использованием датасетов Universal Dependencies.

Поддерживаемые датасеты:
- Universal Dependencies English EWT (English Web Treebank)
- Universal Dependencies English GUM (Georgetown University Multilayer)

Автор: Serik Nurtugan
Дата: 2026-02-05
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import random


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Token:
    """
    Представление токена из CoNLL-U файла.
    
    Attributes:
        form: Форма слова (как написано в тексте)
        lemma: Лемма (базовая форма)
        upos: Универсальный POS-тег (Universal Part-of-Speech)
        feats: Морфологические признаки в формате "Key=Value|Key=Value"
        label: Сгенерированная метка в формате "UPOS|FEATS"
    """
    form: str
    lemma: str
    upos: str
    feats: str
    label: str = field(default="")
    
    def __post_init__(self):
        """Автоматически генерирует метку при создании токена."""
        if not self.label:
            self.label = generate_label(self.upos, self.feats)


@dataclass  
class Sentence:
    """
    Представление предложения как списка токенов.
    
    Attributes:
        tokens: Список токенов предложения
        text: Оригинальный текст предложения (если доступен)
        sent_id: Идентификатор предложения
    """
    tokens: List[Token] = field(default_factory=list)
    text: str = ""
    sent_id: str = ""
    
    def get_forms(self) -> List[str]:
        """Возвращает список форм слов."""
        return [t.form for t in self.tokens]
    
    def get_labels(self) -> List[str]:
        """Возвращает список меток."""
        return [t.label for t in self.tokens]
    
    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class Dataset:
    """
    Представление датасета с разделением на train/dev/test.
    
    Attributes:
        name: Название датасета
        train: Обучающая выборка (список предложений)
        dev: Валидационная выборка
        test: Тестовая выборка
    """
    name: str
    train: List[Sentence] = field(default_factory=list)
    dev: List[Sentence] = field(default_factory=list)
    test: List[Sentence] = field(default_factory=list)
    
    def get_all_sentences(self) -> List[Sentence]:
        """Возвращает все предложения из всех split'ов."""
        return self.train + self.dev + self.test
    
    def stats(self) -> Dict[str, int]:
        """Возвращает статистику датасета."""
        return {
            "name": self.name,
            "train_sentences": len(self.train),
            "dev_sentences": len(self.dev),
            "test_sentences": len(self.test),
            "train_tokens": sum(len(s) for s in self.train),
            "dev_tokens": sum(len(s) for s in self.dev),
            "test_tokens": sum(len(s) for s in self.test),
        }


# =============================================================================
# LABEL GENERATION
# =============================================================================

def generate_label(upos: str, feats: str) -> str:
    """
    Генерирует единую метку для токена в формате UPOS|FEATS.
    
    Формат метки:
    - Если FEATS не пустой: "UPOS|Feat1=Val1|Feat2=Val2|..."
    - Если FEATS пустой ("_"): "UPOS|_"
    
    Args:
        upos: Универсальный POS-тег (например, "VERB", "NOUN")
        feats: Морфологические признаки (например, "Tense=Past|Number=Sing")
    
    Returns:
        Метка в формате "UPOS|FEATS"
        
    Examples:
        >>> generate_label("VERB", "Tense=Past|Number=Sing")
        'VERB|Tense=Past|Number=Sing'
        
        >>> generate_label("PUNCT", "_")
        'PUNCT|_'
        
        >>> generate_label("NOUN", "")
        'NOUN|_'
    """
    # Нормализация пустых FEATS
    if not feats or feats == "_" or feats.strip() == "":
        feats = "_"
    
    return f"{upos}|{feats}"


def parse_label(label: str) -> Tuple[str, str]:
    """
    Разбирает метку обратно на UPOS и FEATS.
    
    Args:
        label: Метка в формате "UPOS|FEATS"
        
    Returns:
        Кортеж (upos, feats)
        
    Example:
        >>> parse_label("VERB|Tense=Past|Number=Sing")
        ('VERB', 'Tense=Past|Number=Sing')
    """
    parts = label.split("|", 1)
    upos = parts[0]
    feats = parts[1] if len(parts) > 1 else "_"
    return upos, feats


# =============================================================================
# CONLL-U PARSING
# =============================================================================

def is_valid_token_id(token_id: str) -> bool:
    """
    Проверяет, является ли ID токена валидным (не мультислово и не пустой узел).
    
    Валидные ID:
    - Простые целые числа: "1", "2", "10"
    
    Невалидные ID (игнорируются):
    - Мультислова: "1-2", "19-20" (диапазон)
    - Пустые узлы: "3.1", "5.2" (с точкой)
    
    Args:
        token_id: Строка ID токена из CoNLL-U
        
    Returns:
        True если ID валидный, False иначе
        
    Examples:
        >>> is_valid_token_id("1")
        True
        >>> is_valid_token_id("1-2")
        False
        >>> is_valid_token_id("3.1")
        False
    """
    # Мультислово (диапазон)
    if "-" in token_id:
        return False
    
    # Пустой узел (enhanced dependencies)
    if "." in token_id:
        return False
    
    # Должен быть целым числом
    try:
        int(token_id)
        return True
    except ValueError:
        return False


def parse_conllu_line(line: str) -> Optional[Token]:
    """
    Парсит одну строку CoNLL-U формата в объект Token.
    
    Формат CoNLL-U строки (10 полей, разделённых табуляцией):
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
    
    Args:
        line: Строка из CoNLL-U файла
        
    Returns:
        Token объект или None если строка невалидна
        
    Example:
        >>> line = "7\\tkilled\\tkill\\tVERB\\tVBD\\tMood=Ind|Tense=Past\\t1\\tparataxis\\t_\\t_"
        >>> token = parse_conllu_line(line)
        >>> token.form
        'killed'
        >>> token.label
        'VERB|Mood=Ind|Tense=Past'
    """
    # Пропускаем пустые строки и комментарии
    if not line or line.startswith("#"):
        return None
    
    # Разбиваем на поля
    fields = line.strip().split("\t")
    
    # Проверяем количество полей (должно быть 10)
    if len(fields) < 10:
        return None
    
    token_id = fields[0]
    
    # Проверяем валидность ID
    if not is_valid_token_id(token_id):
        return None
    
    # Извлекаем нужные поля
    form = fields[1]    # FORM - форма слова
    lemma = fields[2]   # LEMMA - лемма
    upos = fields[3]    # UPOS - универсальный POS-тег
    feats = fields[5]   # FEATS - морфологические признаки
    
    return Token(form=form, lemma=lemma, upos=upos, feats=feats)


def parse_conllu_file(filepath: str) -> List[Sentence]:
    """
    Парсит CoNLL-U файл и возвращает список предложений.
    
    Args:
        filepath: Путь к .conllu файлу
        
    Returns:
        Список объектов Sentence
        
    Raises:
        FileNotFoundError: Если файл не найден
        
    Example:
        >>> sentences = parse_conllu_file("en_ewt-ud-train.conllu")
        >>> len(sentences)
        12543
    """
    sentences = []
    current_tokens = []
    current_text = ""
    current_sent_id = ""
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Пустая строка = конец предложения
            if not line:
                if current_tokens:
                    sentences.append(Sentence(
                        tokens=current_tokens,
                        text=current_text,
                        sent_id=current_sent_id
                    ))
                    current_tokens = []
                    current_text = ""
                    current_sent_id = ""
                continue
            
            # Комментарии с метаданными
            if line.startswith("#"):
                if line.startswith("# text = "):
                    current_text = line[9:]
                elif line.startswith("# sent_id = "):
                    current_sent_id = line[12:]
                continue
            
            # Парсим токен
            token = parse_conllu_line(line)
            if token:
                current_tokens.append(token)
        
        # Добавляем последнее предложение
        if current_tokens:
            sentences.append(Sentence(
                tokens=current_tokens,
                text=current_text,
                sent_id=current_sent_id
            ))
    
    return sentences


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_ud_dataset(
    dataset_dir: str,
    dataset_name: str = "UD_English-EWT"
) -> Dataset:
    """
    Загружает датасет Universal Dependencies с готовыми split'ами.
    
    Ожидаемая структура директории:
        dataset_dir/
        ├── en_ewt-ud-train.conllu
        ├── en_ewt-ud-dev.conllu
        └── en_ewt-ud-test.conllu
    
    Args:
        dataset_dir: Путь к директории с .conllu файлами
        dataset_name: Название датасета для идентификации
        
    Returns:
        Dataset объект с train/dev/test split'ами
        
    Example:
        >>> dataset = load_ud_dataset("datasets/english/UD_English-EWT")
        >>> dataset.stats()
        {'train_sentences': 12543, 'dev_sentences': 2002, 'test_sentences': 2077, ...}
    """
    # Находим файлы .conllu в директории
    files = os.listdir(dataset_dir)
    conllu_files = [f for f in files if f.endswith(".conllu")]
    
    train_sentences = []
    dev_sentences = []
    test_sentences = []
    
    for filename in conllu_files:
        filepath = os.path.join(dataset_dir, filename)
        sentences = parse_conllu_file(filepath)
        
        # Определяем split по имени файла
        if "train" in filename.lower():
            train_sentences = sentences
        elif "dev" in filename.lower():
            dev_sentences = sentences
        elif "test" in filename.lower():
            test_sentences = sentences
    
    return Dataset(
        name=dataset_name,
        train=train_sentences,
        dev=dev_sentences,
        test=test_sentences
    )


def combine_datasets(datasets: List[Dataset], name: str = "Combined") -> Dataset:
    """
    Объединяет несколько датасетов в один.
    
    Args:
        datasets: Список датасетов для объединения
        name: Название объединённого датасета
        
    Returns:
        Объединённый Dataset
        
    Example:
        >>> ewt = load_ud_dataset("datasets/UD_English-EWT")
        >>> gum = load_ud_dataset("datasets/UD_English-GUM")
        >>> combined = combine_datasets([ewt, gum], "EWT+GUM")
    """
    combined = Dataset(name=name)
    
    for ds in datasets:
        combined.train.extend(ds.train)
        combined.dev.extend(ds.dev)
        combined.test.extend(ds.test)
    
    return combined


def split_dataset(
    sentences: List[Sentence],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Sentence], List[Sentence], List[Sentence]]:
    """
    Разделяет список предложений на train/dev/test с фиксированным seed.
    
    Используется когда датасет не имеет готовых split'ов.
    
    Args:
        sentences: Список предложений для разделения
        train_ratio: Доля обучающей выборки (по умолчанию 0.8)
        dev_ratio: Доля валидационной выборки (по умолчанию 0.1)
        test_ratio: Доля тестовой выборки (по умолчанию 0.1)
        seed: Seed для воспроизводимости (по умолчанию 42)
        
    Returns:
        Кортеж (train, dev, test) списков предложений
        
    Example:
        >>> sentences = parse_conllu_file("data.conllu")
        >>> train, dev, test = split_dataset(sentences, seed=42)
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Устанавливаем seed для воспроизводимости
    random.seed(seed)
    
    # Перемешиваем копию списка
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    
    # Вычисляем границы
    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    
    train = shuffled[:train_end]
    dev = shuffled[train_end:dev_end]
    test = shuffled[dev_end:]
    
    return train, dev, test


# =============================================================================
# LABEL VOCABULARY
# =============================================================================

def build_label_vocab(
    sentences: List[Sentence],
    min_count: int = 1
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Строит словарь меток из списка предложений.
    
    Args:
        sentences: Список предложений для построения словаря
        min_count: Минимальное количество вхождений метки
        
    Returns:
        Кортеж (label2id, id2label)
        - label2id: словарь метка -> индекс
        - id2label: словарь индекс -> метка
        
    Example:
        >>> label2id, id2label = build_label_vocab(train_sentences)
        >>> label2id["VERB|Tense=Past"]
        15
        >>> id2label[15]
        'VERB|Tense=Past'
    """
    # Считаем частоту меток
    label_counts = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            label_counts[token.label] += 1
    
    # Фильтруем по минимальному количеству
    labels = [label for label, count in label_counts.items() if count >= min_count]
    
    # Сортируем для стабильности
    labels = sorted(labels)
    
    # Создаём маппинги
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def get_label_statistics(sentences: List[Sentence]) -> Dict[str, Any]:
    """
    Собирает статистику по меткам в датасете.
    
    Args:
        sentences: Список предложений
        
    Returns:
        Словарь со статистикой
    """
    label_counts = Counter()
    upos_counts = Counter()
    
    for sentence in sentences:
        for token in sentence.tokens:
            label_counts[token.label] += 1
            upos_counts[token.upos] += 1
    
    return {
        "total_tokens": sum(label_counts.values()),
        "unique_labels": len(label_counts),
        "unique_upos": len(upos_counts),
        "top_10_labels": label_counts.most_common(10),
        "upos_distribution": dict(upos_counts),
    }


# =============================================================================
# DATA PREPARATION FOR TRANSFORMERS
# =============================================================================

def prepare_for_training(
    sentences: List[Sentence],
    label2id: Optional[Dict[str, int]] = None
) -> List[Dict[str, Any]]:
    """
    Подготавливает данные для обучения трансформеров.
    
    Каждый пример содержит:
    - tokens: список форм слов
    - labels: список меток (той же длины)
    - label_ids: список индексов меток (если label2id предоставлен)
    
    Args:
        sentences: Список предложений
        label2id: Опциональный словарь label -> id
        
    Returns:
        Список словарей с подготовленными данными
        
    Example:
        >>> data = prepare_for_training(sentences, label2id)
        >>> data[0]
        {
            'tokens': ['The', 'cat', 'sat'],
            'labels': ['DET|Definite=Def', 'NOUN|Number=Sing', 'VERB|Tense=Past'],
            'label_ids': [5, 23, 45]
        }
    """
    prepared = []
    
    for sentence in sentences:
        tokens = sentence.get_forms()
        labels = sentence.get_labels()
        
        example = {
            "tokens": tokens,
            "labels": labels,
            "sentence_id": sentence.sent_id,
            "text": sentence.text,
        }
        
        # Добавляем label_ids если есть словарь
        if label2id:
            label_ids = []
            for label in labels:
                # Используем -100 для неизвестных меток (игнорируются в CrossEntropyLoss)
                label_ids.append(label2id.get(label, -100))
            example["label_ids"] = label_ids
        
        prepared.append(example)
    
    return prepared


def export_to_json(
    data: List[Dict[str, Any]],
    filepath: str
) -> None:
    """
    Экспортирует подготовленные данные в JSON файл.
    
    Args:
        data: Подготовленные данные
        filepath: Путь для сохранения
    """
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_to_jsonl(
    data: List[Dict[str, Any]],
    filepath: str
) -> None:
    """
    Экспортирует подготовленные данные в JSONL формат (по одному JSON на строку).
    
    Args:
        data: Подготовленные данные
        filepath: Путь для сохранения
    """
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


# =============================================================================
# DEMO AND EXAMPLES
# =============================================================================

def demo():
    """
    Демонстрация работы пайплайна на реальных данных.
    """
    print("=" * 80)
    print("ENGLISH MORPHOLOGICAL DATA PREPROCESSING PIPELINE - DEMO")
    print("=" * 80)
    
    # Определяем пути к датасетам
    base_path = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(base_path, "..", "datasets", "english")
    
    ewt_path = os.path.join(datasets_path, "UD_English-EWT")
    gum_path = os.path.join(datasets_path, "UD_English-GUM")
    
    print(f"\n📂 Datasets path: {datasets_path}")
    
    # Загружаем датасеты
    print("\n" + "-" * 40)
    print("1. LOADING DATASETS")
    print("-" * 40)
    
    if os.path.exists(ewt_path):
        ewt = load_ud_dataset(ewt_path, "UD_English-EWT")
        print(f"\n✅ UD English EWT loaded:")
        for key, value in ewt.stats().items():
            print(f"   {key}: {value}")
    else:
        print(f"\n❌ EWT dataset not found at: {ewt_path}")
        return
    
    if os.path.exists(gum_path):
        gum = load_ud_dataset(gum_path, "UD_English-GUM")
        print(f"\n✅ UD English GUM loaded:")
        for key, value in gum.stats().items():
            print(f"   {key}: {value}")
    else:
        print(f"\n⚠️ GUM dataset not found at: {gum_path}")
        gum = None
    
    # Пример распарсенного предложения
    print("\n" + "-" * 40)
    print("2. PARSED SENTENCE EXAMPLE")
    print("-" * 40)
    
    if ewt.train:
        sample_sentence = ewt.train[0]
        print(f"\n📝 Sentence ID: {sample_sentence.sent_id}")
        print(f"📝 Original text: {sample_sentence.text}")
        print(f"\n{'Token':<15} {'Lemma':<15} {'UPOS':<8} {'FEATS':<30} {'Label'}")
        print("-" * 90)
        
        for token in sample_sentence.tokens[:10]:  # Показываем первые 10 токенов
            feats_short = token.feats[:27] + "..." if len(token.feats) > 30 else token.feats
            print(f"{token.form:<15} {token.lemma:<15} {token.upos:<8} {feats_short:<30} {token.label}")
        
        if len(sample_sentence.tokens) > 10:
            print(f"... and {len(sample_sentence.tokens) - 10} more tokens")
    
    # Демонстрация генерации меток
    print("\n" + "-" * 40)
    print("3. LABEL GENERATION EXAMPLES")
    print("-" * 40)
    
    examples = [
        ("VERB", "Tense=Past|Number=Sing"),
        ("NOUN", "Number=Plur"),
        ("DET", "Definite=Def|PronType=Art"),
        ("PUNCT", "_"),
        ("ADJ", "Degree=Pos"),
    ]
    
    print(f"\n{'UPOS':<8} {'FEATS':<30} {'Generated Label'}")
    print("-" * 60)
    for upos, feats in examples:
        label = generate_label(upos, feats)
        print(f"{upos:<8} {feats:<30} {label}")
    
    # Построение словаря меток
    print("\n" + "-" * 40)
    print("4. LABEL VOCABULARY")
    print("-" * 40)
    
    label2id, id2label = build_label_vocab(ewt.train)
    print(f"\n📊 Unique labels in training set: {len(label2id)}")
    print("\nFirst 10 labels:")
    for i, (label, idx) in enumerate(list(label2id.items())[:10]):
        print(f"   {idx}: {label}")
    
    # Статистика по меткам
    print("\n" + "-" * 40)
    print("5. LABEL STATISTICS")
    print("-" * 40)
    
    stats = get_label_statistics(ewt.train)
    print(f"\n📊 Total tokens: {stats['total_tokens']}")
    print(f"📊 Unique labels: {stats['unique_labels']}")
    print(f"📊 Unique UPOS tags: {stats['unique_upos']}")
    
    print("\n🏷️ UPOS distribution:")
    for upos, count in sorted(stats['upos_distribution'].items(), key=lambda x: -x[1]):
        print(f"   {upos:<8}: {count:>8} ({count/stats['total_tokens']*100:.1f}%)")
    
    print("\n🏷️ Top 10 labels:")
    for label, count in stats['top_10_labels']:
        print(f"   {label:<40}: {count}")
    
    # Подготовка данных для трансформеров
    print("\n" + "-" * 40)
    print("6. PREPARED DATA FOR TRANSFORMERS")
    print("-" * 40)
    
    train_data = prepare_for_training(ewt.train[:3], label2id)
    
    print("\nExample prepared data (first 3 sentences):")
    for i, example in enumerate(train_data):
        print(f"\n📌 Example {i+1}:")
        print(f"   Tokens: {example['tokens'][:5]}..." if len(example['tokens']) > 5 else f"   Tokens: {example['tokens']}")
        print(f"   Labels: {example['labels'][:5]}..." if len(example['labels']) > 5 else f"   Labels: {example['labels']}")
        print(f"   Label IDs: {example['label_ids'][:5]}..." if len(example['label_ids']) > 5 else f"   Label IDs: {example['label_ids']}")
        print(f"   Lengths match: {len(example['tokens']) == len(example['labels'])}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


def create_full_pipeline(
    ewt_dir: str,
    gum_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    combine: bool = False
) -> Tuple[Dataset, Dict[str, int], Dict[int, str]]:
    """
    Создаёт полный пайплайн предобработки данных.
    
    Args:
        ewt_dir: Путь к директории UD English EWT
        gum_dir: Опциональный путь к директории UD English GUM
        output_dir: Опциональный путь для сохранения результатов
        combine: Объединять ли датасеты
        
    Returns:
        Кортеж (dataset, label2id, id2label)
    """
    # Загружаем EWT
    ewt = load_ud_dataset(ewt_dir, "UD_English-EWT")
    
    # Загружаем GUM если указан
    gum = None
    if gum_dir and os.path.exists(gum_dir):
        gum = load_ud_dataset(gum_dir, "UD_English-GUM")
    
    # Объединяем если нужно
    if combine and gum:
        dataset = combine_datasets([ewt, gum], "EWT+GUM")
    else:
        dataset = ewt
    
    # Строим словарь меток на обучающих данных
    label2id, id2label = build_label_vocab(dataset.train)
    
    # Экспортируем если указана директория
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        train_data = prepare_for_training(dataset.train, label2id)
        dev_data = prepare_for_training(dataset.dev, label2id)
        test_data = prepare_for_training(dataset.test, label2id)
        
        export_to_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
        export_to_jsonl(dev_data, os.path.join(output_dir, "dev.jsonl"))
        export_to_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
        
        # Сохраняем словарь меток
        import json
        with open(os.path.join(output_dir, "label2id.json"), "w") as f:
            json.dump(label2id, f, indent=2)
        
        print(f"✅ Data exported to {output_dir}")
    
    return dataset, label2id, id2label


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    demo()
