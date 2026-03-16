# AGENTS.md — Контекст для AI агентов

> Этот файл для Claude, Cursor, Copilot и любого другого AI-агента
> работающего в этом репозитории. Читай полностью перед началом работы.

---

## О ПРОЕКТЕ

**Erowid NLP v2** — современный ML/NLP анализ трип-репортов с Erowid.org.

Это форк репозитория [Monkeyanator/erowid-lsa](https://github.com/Monkeyanator/erowid-lsa),
оригинально написанного в **2016 году** с использованием LSA + sklearn.

Цель — переписать на современную архитектуру (трансформеры, SHAP, BERTopic)
и провести **сравнение результатов через 9 лет**.

---

## СТРУКТУРА ДАННЫХ — ДЕТАЛЬНО

### Папка `EROWID-2026/` — ИСТОРИЧЕСКИЙ АРХИВ

```
EROWID-2026/
├── core-experiences_2016/   ← "эталонные" репорты 2016
│   └── <substance>/         ← папки по веществам
│       └── *.txt            ← каждый файл = один трип-репорт
├── experiences_2016/        ← полный датасет 2016
│   └── <substance>/
│       └── *.txt
├── test/                    ← тестовые данные
├── analysis.py              ← ОРИГИНАЛЬНЫЙ анализ 2016 (LSA baseline)
├── erowid-scrape.py         ← скрапер (может нуждаться в обновлении)
└── stopwords_en.txt         ← English stopwords
```

**Формат .txt файлов:**
Каждый файл — один трип-репорт. Чистый текст, без метаданных внутри файла.
Метаданные (вещество) берутся из имени папки.

### Почему два набора данных критически важны:

| Набор | Дата | Назначение |
|-------|------|-----------|
| `core-experiences_2016` | 2016 | Высококачественная выборка |
| `experiences_2016` | 2016 | Полный датасет |
| `data/raw/` (создаётся) | 2025 | Свежие данные со скрапера |

Без `*_2016` папок — невозможно провести анализ "тогда vs сейчас".

---

## PIPELINE — КАК РАБОТАЕТ ПРОЕКТ

```
Этап 0: git clone + fork + исправить TODO
    ↓
Этап 1: Скрапер → data/raw/*.txt → dataset_raw.csv
    ↓
Этап 2: EDA — статистика, TF-IDF анализ, log-odds, статтесты
    ↓
Этап 3: Embeddings (all-mpnet-base-v2) → embeddings.npy
         → Классификатор вещества (distilbert fine-tune)
         → Классификатор тональности (positive/negative/bad trip)
    ↓
Этап 4: BERTopic → темы по веществам
         UMAP + HDBSCAN → семантические кластеры
         Cosine similarity матрица веществ
    ↓
Этап 5: Harm Reduction модуль
         → детектор "стоит ли беспокоиться"
         → предсказание исхода трипа
    ↓
Этап 6: SHAP → интерпретация классификатора
         → waterfall plots, beeswarm plots
    ↓
Этап 7: Сравнение 2016 vs 2025
         → метрики: LSA baseline vs distilbert
         → лексический дрейф
         → эволюция тем
    ↓
Этап 8: Gradio app → HuggingFace Spaces
         README → GitHub
```

---

## МОДЕЛИ И МЕТРИКИ

### Классификатор вещества
- **Input:** текст трип-репорта
- **Output:** топ-3 вещества + confidence scores
- **Метрики:** accuracy top-1 > 70%, top-3 > 90%, f1-macro > 0.70
- **Архитектуры:** LogReg+TF-IDF (baseline 2016) → LightGBM+embeddings → distilbert

### Классификатор тональности
- **Input:** текст трип-репорта
- **Output:** positive / negative / mixed / bad_trip
- **Лейблы:** из рейтингов авторов (готовые, не нужна разметка)
- **Особо важно:** recall класса bad_trip (для harm reduction)

### Harm Reduction модуль
- **Input:** описание эффекта + вещество
- **Output:** % пользователей с похожим опытом + схожие репорты
- **Алгоритм:** cosine similarity по embeddings → статистика по топ-N

---

## ВАЖНЫЕ ТЕХНИЧЕСКИЕ ДЕТАЛИ

### Работа с длинными текстами
Репорты могут быть очень длинными (> 512 токенов).
Для sentence-transformers брать первые 512 токенов ИЛИ усреднять по чанкам.

### Дисбаланс классов
Некоторые вещества имеют намного больше репортов (cannabis >> редкие).
Использовать stratified split и class_weight='balanced'.

### Temporal split
При сравнении 2016 vs 2025 — не смешивать в train/test.
Обучать на 2016, тестировать на 2025 (или наоборот) для честного сравнения.

### embeddings.npy
Большой файл (~несколько GB для полного датасета).
НЕ коммитить в git — добавить в .gitignore.
Хранить локально и на Google Drive / HuggingFace Hub.

---

## СОГЛАШЕНИЯ ПО КОДУ

```python
# Импорты: стандартная библиотека → third-party → local
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.embeddings import load_embeddings

# Константы — вверху файла, UPPER_SNAKE_CASE
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32

# Функции — документировать параметры и возвращаемые значения
def load_dataset(path: str, era: str = "all") -> pd.DataFrame:
    """
    Загружает датасет трип-репортов.
    
    Args:
        path: путь к CSV файлу
        era: "pre-2016", "post-2016" или "all"
    
    Returns:
        DataFrame с колонками: substance, text, date, rating, era
    """
    ...
```

---

## ЧАСТЫЕ ОШИБКИ — НЕ ДЕЛАТЬ

1. **Не удалять `*_2016` папки** — это единственная копия исторических данных
2. **Не коммитить `embeddings.npy`** — слишком большой файл
3. **Не смешивать данные 2016 и 2025** в одном train split при сравнительном анализе
5. **Не использовать `random_state` разные** в разных местах — фиксировать `SEED = 42`

---

## GIT WORKFLOW

```bash
# Маленькие атомарные коммиты
git commit -m "fix: resolve TODO - add error handling in scraper"
git commit -m "feat: add sentence-transformers embeddings pipeline"
git commit -m "feat: add LightGBM classifier on embeddings"
git commit -m "docs: update README with baseline comparison table"

# НЕ делать:
git commit -m "changes"  # слишком общо
git add .               # без проверки что добавляется
```

---

## РЕСУРСЫ

- Оригинал: https://github.com/Monkeyanator/erowid-lsa
- Erowid: https://www.erowid.org/experiences/
- HuggingFace модель: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- BERTopic docs: https://maartengr.github.io/BERTopic/
- SHAP docs: https://shap.readthedocs.io/
- UMAP docs: https://umap-learn.readthedocs.io/
