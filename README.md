# Практична робота 9.1 - IoT + Machine Learning (варіант 2: Industrial Equipment Anomaly Detection)

Дисципліна: Технології IoT та SMART-технології. Студент: Влад Рукасов, група Ік-42.

## Запуск
```
pip install -r requirements.txt
python simulator.py        # датасет data/motor_dataset.csv + графіки (figures/)
python preprocessing.py    # EDA, features, scaler (models/)
python train.py            # 4 моделі, таблиця data/model_comparison.csv
python inference.py        # real-time engine + бенчмарки (alerts.log)
```
Структура: `simulator.py`, `preprocessing.py`, `autoencoder.py`, `train.py`, `inference.py`, `data/`, `models/`, `figures/`, `report/`.

## Декларація використання ШІ
ШІ Claude допоміг з усіма перевірками коду на наявність всіх вимог у завданні. Також він допоміг з оформленням файлу README.md
