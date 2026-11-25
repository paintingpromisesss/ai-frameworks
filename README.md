# Лабораторные работы по машинному обучению

Фокин Леонид Александрович, группа 401.

[Градиентный бустинг](lab5.ipynb)

## Как запустить

Откройте ноутбуки в Jupyter или VS Code с ядром Python из виртуального окружения. Команды ниже выполняются из корня репозитория (проверено на Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python test_models.py
jupyter execute --timeout=600 lab5.ipynb
```

Датасеты уже в репозитории: [источники и подготовка](data/README.md).
