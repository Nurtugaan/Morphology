"""
Model-specific training scripts for English Morphological Analysis.

Этот пакет содержит отдельные скрипты обучения для каждой модели:

- train_bert.py      - BERT-base (110M params, Transformer)
- train_roberta.py   - RoBERTa-base (125M params, Transformer)
- train_distilbert.py - DistilBERT (66M params, Distilled Transformer)
- train_albert.py    - ALBERT-base (12M params, Factorized Transformer)
- train_flair.py     - Flair (BiLSTM + CRF, NOT Transformer)

Все Transformer-модели используют общий код из train_base.py.
Flair использует собственную библиотеку и полностью отдельную логику.

Запуск:
    # BERT
    python -m services.training.models.train_bert
    
    # RoBERTa
    python -m services.training.models.train_roberta
    
    # DistilBERT
    python -m services.training.models.train_distilbert
    
    # ALBERT
    python -m services.training.models.train_albert
    
    # Flair (requires: pip install flair)
    python -m services.training.models.train_flair
"""
