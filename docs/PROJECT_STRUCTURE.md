morpho_analyzer/
│
├── backend/
│   ├── app/
│   │   ├── api/                # REST API / FastAPI роуты
│   │   │   ├── en_analysis.py  # API endpoints для англ. моделей
│   │   │   ├── kk_analysis.py  # API endpoints для казахских моделей
│   │   │   └── own_model.py    # API для собственной модели
│   │   ├── core/               # Конфиги, логирование, константы
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── services/           # Взаимодействие с ML-сервисами
│   │   │   ├── inference.py    # Функции предсказания моделей
│   │   │   └── utils.py
│   │   ├── models/             # Pydantic / ORM модели
│   │   ├── db/                 # Скрипты для БД (если нужна)
│   │   └── main.py             # Точка входа FastAPI
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── components/         # React/Vue компоненты
│   │   ├── pages/              # Страницы приложения
│   │   ├── services/           # API-клиенты
│   │   └── App.jsx             # Главный компонент
│   ├── public/                 # Статика
│   ├── package.json
│   └── vite.config.js / webpack.config.js
│
├── services/
│   ├── datasets/
│   │   ├── english/            # EN датасеты (CoNLL-U, OntoNotes)
│   │   └── kazakh/             # KK датасеты (UD kk-ktb, собственный корпус)
│   │
│   ├── preprocessing/
│   │   ├── en_preprocessing.py # Подготовка EN датасетов
│   │   ├── kk_preprocessing.py # Подготовка KK датасетов
│   │   └── utils.py            # Общие функции (токенизация, split)
│   │
│   ├── training/
│   │   ├── en_training.py      # Обучение трансформеров на EN
│   │   ├── kk_training.py      # Обучение трансформеров на KK
│   │   └── utils.py            # Общие функции для тренировки
│   │
│   ├── evaluation/
│   │   ├── en_evaluation.py    # Метрики для EN моделей
│   │   ├── kk_evaluation.py    # Метрики для KK моделей
│   │   └── analysis.py         # Анализ ошибок, графики, таблицы
│   │
│   ├── transformers_models/    # Трансформеры (BERT, RoBERTa, Kaz-RoBERTa)
│   │   ├── en/                 # Англоязычные модели
│   │   └── kk/                 # Казахские модели
│   │
│   ├── own_model/              # Ваша собственная архитектура
│   │   ├── architecture.py     # BiLSTM/CRF/CNN/Hybrid
│   │   ├── train.py            # Скрипт обучения собственной модели
│   │   ├── evaluate.py         # Скрипт оценки собственной модели
│   │   └── utils.py            # Loss, optimizer, токенизация
│   │
│   ├── parsers/                # Сбор собственных данных
│   │   ├── web_parser.py       # Скрейпинг сайтов, новостей, Википедии
│   │   └── utils.py            # Очистка и нормализация текста
│   │
│   └── generator/              # Генерация собственного корпуса
│       ├── data_generator.py
│       └── augmentations.py
│
├── notebooks/                  # Jupyter ноутбуки для анализа, прототипирования
│
├── tests/                      # Юнит и интеграционные тесты
│   ├── backend/
│   ├── preprocessing/
│   └── training/
│
├── docs/                       # Документация, схемы, графики, отчёты
│
├── .gitignore
└── README.md
