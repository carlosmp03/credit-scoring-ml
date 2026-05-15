# Credit Scoring ML

End-to-end модель кредитного скоринга на датасете Kaggle *Give Me Some Credit*: вероятность серьёзной просрочки (90+ дней) в течение двух лет.

## Что внутри

- **EDA + моделирование** в одном ноутбуке (`notebooks/eda_and_model.ipynb`):
  корректный сплит до любых трансформаций, заполнение пропусков по статистикам train (без утечки), feature engineering, подбор гиперпараметров CatBoost через Optuna, отбор признаков по importance, **isotonic-калибровка вероятностей**, **SHAP**, бизнес-метрики (Gini, KS, profit curve), **score-карта**.
- **REST API** (FastAPI, `api/app.py`): синхронные предсказания для одного клиента и батчей, OpenAPI-доки на `/docs`.
- **Дашборд** (Streamlit, `dashboard/app.py`): интерактивная форма ввода клиента, SHAP waterfall для объяснения каждого решения, графики качества модели.

## Метрики

| Метрика | Значение |
|---|---|
| ROC-AUC (test) | ~0.872 |
| Gini | ~0.744 |
| KS-статистика | ~0.55 |
| Признаков (после отбора) | 11 |

## Структура проекта

```
credit-scoring-ml/
├── data/                       # cs-training.csv, cs-test.csv (Kaggle)
├── notebooks/
│   └── eda_and_model.ipynb     # весь пайплайн
├── api/
│   ├── app.py                  # FastAPI
│   └── preprocessing.py        # общий препроцессинг (используется и в дашборде)
├── dashboard/
│   └── app.py                  # Streamlit
├── models/                     # генерируется ноутбуком
│   ├── cb_model.pkl                  # некалиброванная (нужна для SHAP)
│   ├── cb_model_calibrated.pkl       # isotonic — её зовёт API
│   ├── feature_names.json
│   ├── fill_values.json
│   ├── best_params.json
│   ├── metrics.json
│   ├── calibration_curve.png
│   ├── profit_curve.png
│   ├── score_distribution.png
│   └── shap_summary.png
├── requirements.txt
└── README.md
```

## Как запустить

### 1. Установка
```bash
pip install -r requirements.txt
```

### 2. Данные
Скачать Kaggle *Give Me Some Credit* и положить `cs-training.csv`, `cs-test.csv` в `data/`.

### 3. Обучение
Открыть и прогнать `notebooks/eda_and_model.ipynb`. На выходе появятся все артефакты в `models/`.

### 4. API
```bash
python -m uvicorn api.app:app --reload --port 8000
```
- Swagger UI: <http://localhost:8000/docs>
- Health:     <http://localhost:8000/health>

Пример запроса:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.4,
    "MonthlyIncome": 5400,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0
  }'
```

Ответ:
```json
{
  "probability_of_default": 0.04,
  "score": 712,
  "decision": "approve",
  "threshold_used": 0.18
}
```

### 5. Дашборд
```bash
streamlit run dashboard/app.py
```

## Методологические решения

### Сплит до любых трансформаций
Все статистики заполнения (медиана дохода, мода иждивенцев) считаются **только на `X_train`** и применяются к `X_val`/`X_test`. Иначе была бы утечка train-side информации в test через групповую статистику.

### Калибровка вероятностей
`auto_class_weights='Balanced'` оптимизирует ранжирование, но смещает вероятности (предсказывает дефолт чаще, чем он реально встречается). Для скоринга это критично — по PD считается ожидаемая потеря (EL = PD × LGD × EAD). Калибруем `CalibratedClassifierCV` с `method="isotonic"` на отдельной val-выборке, не подсматривая в test.

### Бизнес-порог
Дефолтный порог одобрения берётся не из F1, а из бизнес-симуляции (см. ноутбук, секция «Бизнес-метрики»): максимизируется ожидаемая прибыль с допущением `gain_good=+1, loss_bad=-5`. Под реальный продукт коэффициенты надо подкрутить.

### Score-карта
Перевод PD в скор по стандартной формуле: `score = offset + factor·ln((1-PD)/PD)`, где `factor = PDO/ln(2)`. Калибровка: 600 ↔ odds 50:1, PDO=20.
