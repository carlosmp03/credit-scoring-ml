"""Streamlit-дашборд для модели кредитного скоринга.

Запуск (из корня проекта):
    streamlit run dashboard/app.py

Что показывает:
    - Форма ввода всех фичей одного клиента
    - Вероятность дефолта + скоринговый балл + решение
    - SHAP-вклад каждой фичи в это конкретное предсказание (waterfall)
    - Метрики модели и графики из артефактов (calibration curve, profit curve, score distribution)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Делаем доступным импорт api.preprocessing из корня проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from api.preprocessing import load_artifacts, prepare, pd_to_score  # noqa: E402

MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Загрузка моделей и артефактов (кэшируется)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    calibrated = joblib.load(MODELS_DIR / "cb_model_calibrated.pkl")
    raw = joblib.load(MODELS_DIR / "cb_model.pkl")
    artifacts = load_artifacts(MODELS_DIR)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return calibrated, raw, artifacts, metrics


CALIBRATED, RAW_MODEL, ARTIFACTS, METRICS = load_models()


@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(RAW_MODEL)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Scoring", page_icon="💳", layout="wide")
st.title("💳 Credit Scoring Dashboard")
st.caption("CatBoost + isotonic calibration. Модель оценивает вероятность серьёзной просрочки (90+ дней) в течение двух лет.")

# Боковая панель — метрики модели
with st.sidebar:
    st.header("Метрики модели")
    if METRICS:
        st.metric("ROC-AUC (test)",   f"{METRICS.get('test_auc_calibrated', 0):.4f}")
        st.metric("Gini",              f"{METRICS.get('gini', 0):.4f}")
        st.metric("KS-статистика",    f"{METRICS.get('ks_statistic', 0):.4f}")
        st.metric("Признаков",         METRICS.get("n_features", "—"))
    else:
        st.info("metrics.json не найден — запустите ноутбук.")

    st.divider()
    st.subheader("Порог одобрения")
    default_thr = float(METRICS.get("best_threshold_business", 0.5))
    threshold = st.slider(
        "Одобряем клиентов с PD ниже порога:",
        min_value=0.05, max_value=0.95, value=default_thr, step=0.01,
        help="Из бизнес-симуляции в ноутбуке оптимальный порог автоматически подставлен.",
    )

# ---------------------------------------------------------------------------
# Форма ввода
# ---------------------------------------------------------------------------
st.subheader("Данные клиента")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Возраст", min_value=18, max_value=100, value=45)
    monthly_income = st.number_input("Месячный доход (оставьте 0 для 'не указан')", min_value=0.0, value=5400.0, step=100.0)
    income_provided = st.checkbox("Доход указан", value=True, help="Сам факт пропуска — сильный сигнал в скоринге")
    n_dependents = st.number_input("Число иждивенцев", min_value=0, max_value=20, value=0)

with col2:
    revolving = st.number_input("RevolvingUtilization (доля 0..1, иногда выше — выброс)", min_value=0.0, value=0.3, step=0.01)
    debt_ratio = st.number_input("DebtRatio", min_value=0.0, value=0.4, step=0.01)
    n_open = st.number_input("Открытых кредитных линий", min_value=0, max_value=80, value=8)
    n_real_estate = st.number_input("Ипотечных кредитов", min_value=0, max_value=60, value=1)

with col3:
    n_30_59 = st.number_input("Просрочек 30-59 дней", min_value=0, max_value=20, value=0)
    n_60_89 = st.number_input("Просрочек 60-89 дней", min_value=0, max_value=20, value=0)
    n_90 = st.number_input("Просрочек 90+ дней",  min_value=0, max_value=20, value=0)


def build_client_row() -> pd.DataFrame:
    """Собираем DataFrame с теми же колонками, что в обучающем CSV."""
    return pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": n_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": (monthly_income if income_provided else np.nan),
        "NumberOfOpenCreditLinesAndLoans": n_open,
        "NumberOfTimes90DaysLate": n_90,
        "NumberRealEstateLoansOrLines": n_real_estate,
        "NumberOfTime60-89DaysPastDueNotWorse": n_60_89,
        "NumberOfDependents": n_dependents,
    }])


if st.button("📊 Оценить клиента", type="primary"):
    raw_row = build_client_row()
    X = prepare(raw_row, ARTIFACTS)

    pd_value = float(CALIBRATED.predict_proba(X)[0, 1])
    score = pd_to_score(pd_value)
    decision = "✅ APPROVE" if pd_value < threshold else "❌ DENY"

    st.divider()
    st.subheader("Результат")
    c1, c2, c3 = st.columns(3)
    c1.metric("Вероятность дефолта", f"{pd_value:.2%}")
    c2.metric("Скоринговый балл", f"{score:.0f}")
    c3.metric("Решение", decision)

    # SHAP waterfall — вклад каждой фичи в конкретное решение
    st.subheader("Почему такое решение (SHAP)")
    explainer = get_explainer()
    shap_values = explainer.shap_values(X)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and not np.isscalar(expected_value):
        expected_value = float(np.array(expected_value).flatten()[0])

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=X.iloc[0].values,
        feature_names=X.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(explanation, show=False, max_display=10)
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.close()

    st.caption(
        "На графике вклад каждой фичи в логит модели. Красные стрелки увеличивают риск, "
        "синие — снижают. Если итоговая вероятность вас удивила — посмотрите, какие фичи "
        "перетянули решение."
    )

# ---------------------------------------------------------------------------
# Графики из артефактов
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Качество модели")
tabs = st.tabs(["Calibration curve", "Profit vs threshold", "Распределение скоров", "SHAP summary"])

for tab, name, caption in zip(
    tabs,
    ["calibration_curve.png", "profit_curve.png", "score_distribution.png", "shap_summary.png"],
    [
        "Чем ближе линия к диагонали, тем точнее предсказанные вероятности.",
        "Зависимость прибыли от порога одобрения. Допущения: +1 за хорошего клиента, -5 за дефолтника.",
        "Распределение скоров по реальным классам — насколько хорошо модель их разделяет.",
        "Глобальная важность и направление влияния фичей.",
    ],
):
    with tab:
        path = MODELS_DIR / name
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.caption(caption)
        else:
            st.info(f"Файл {name} не найден — запустите ноутбук, чтобы его сгенерировать.")
