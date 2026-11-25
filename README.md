# Лабораторные работы по машинному обучению

Фокин Леонид Александрович, группа 401, вариант 20.

1. [K ближайших соседей](lab1.ipynb)
2. [Логистическая и линейная регрессия](lab2.ipynb)
3. [Решающее дерево](lab3.ipynb)
4. [Случайный лес](lab4.ipynb)
5. [Градиентный бустинг](lab5.ipynb)

[Итоговое сравнение всех моделей](summary.ipynb)

## Как запустить

Откройте ноутбуки в Jupyter или VS Code с ядром Python из виртуального окружения. Команды ниже выполняются из корня репозитория (проверено на Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_models.py
jupyter execute --timeout=600 lab1.ipynb lab2.ipynb lab3.ipynb lab4.ipynb lab5.ipynb summary.ipynb
```

Скачивать данные для запуска не нужно: CSV лежат в `data/`, [описание источников и отбора](data/README.md). Результаты ячеек сохранены в ноутбуках. `summary.ipynb` читает готовые оценки из `results/`; повторно обучать модели для просмотра итогов не нужно. Самописные модели находятся в `models/`.
