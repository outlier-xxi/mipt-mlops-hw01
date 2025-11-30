# README

## Цель проекта

Построение воспроизводимого ML-пайплайна для предсказания качества вина с использованием DVC и MLflow.

- Датасет: Wine Quality Dataset: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data
- Модель: Logistic Regression

## Как запустить

```bash
# 1. Клонировать репозиторий
git clone https://github.com/outlier-xxi/mipt-mlops-hw01.git
cd mlops-hw01

# 2. Установить зависимости
uv install
# или
# pip install -r requirements.txt

# 3. Получить данные из DVC remote
# Я использовал локальную ФС
# Поэтому придется скачать данные вручную локально из: 
#   - https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data
#   - и сохранить в `data/raw/WineQT.csv`

# 4. Запустить pipeline
dvc repro

# 5. Просмотреть результаты в MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

UI будет доступен по ссылке: http://localhost:5000

## Краткое описание пайплайна

Пайплайн состоит из 2 стадий, описанных в `dvc.yaml`:

1. prepare:
   - Загружает сырые данные из `data/raw/WineQT.csv`
   - Удаляет дубликаты и пропуски
   - Разделяет на train/test (80/20) со стратификацией
   - Сохраняет в `data/processed/`: X_train, X_test, y_train, y_test

2. train:
   - Загружает обработанные данные
   - Применяет StandardScaler
   - Обучает LogisticRegression
   - Логирует в MLflow: параметры, метрики, модель (артефакт)
   - Сохраняет модель в `model.pkl`

Параметры пайплайна настраиваются в `params.yaml`.

## Где смотреть UI MLflow

После запуска `mlflow ui --backend-store-uri sqlite:///mlflow.db`:
- URL: http://localhost:5000


## Структура проекта

```
├── data/
│   ├── raw/              # Сырые данные (DVC)
│   └── processed/        # Обработанные данные (DVC pipeline output)
├── src/
│   ├── prepare.py        # Препроцессинг данных
│   ├── train.py          # Обучение модели
│   └── common/
│       └── settings.py   # Настройки проекта (pydantic_settings)
├── dvc.yaml              # DVC pipeline
├── params.yaml           # Параметры pipeline
├── requirements.txt      # Python dependencies
├── model.pkl             # Обученная модель (DVC output)
└── README.md             # Документация
```

## Воспроизводимость

Все эксперименты воспроизводимы через:
- DVC: версионирование данных и пайплайнов
- params.yaml: гиперпараметры
- dvc.lock: зависимости и хэши
- MLflow: трекинг экспериментов и моделей
```
