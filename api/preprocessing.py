"""Препроцессинг входных данных для API и дашборда.

ВАЖНО: эти функции должны быть БИТ-в-БИТ идентичны тем, что в notebooks/eda_and_model.ipynb,
иначе предсказания на проде разъедутся с предсказаниями из ноутбука.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_artifacts(models_dir: Path | str) -> dict:
    """Загружаем артефакты, нужные для предсказания: статистики заполнения и список фичей."""
    models_dir = Path(models_dir)
    with open(models_dir / "fill_values.json") as f:
        fill_values = json.load(f)
    with open(models_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    return {
        "income_median": float(fill_values["income_median"]),
        "dependents_mode": int(fill_values["dependents_mode"]),
        "feature_names": list(feature_names),
    }


def fill_missing(df: pd.DataFrame, income_median: float, dependents_mode: int) -> pd.DataFrame:
    """Заполняем пропуски и создаём бинарные индикаторы пропуска.

    Индикаторы создаются ДО заполнения, иначе сигнал «не указал доход» теряется.
    """
    df = df.copy()
    df["MonthlyIncome_missing"] = df["MonthlyIncome"].isnull().astype(int)
    df["NumberOfDependents_missing"] = df["NumberOfDependents"].isnull().astype(int)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(dependents_mode)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляем produced features. Чистая функция, без утечек."""
    df = df.copy()
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["Has90DaysLate"] = (df["NumberOfTimes90DaysLate"] > 0).astype(int)
    df["MonthlyIncomePerPerson"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["AgeRisk"] = ((df["age"] < 25) | (df["age"] > 65)).astype(int)
    return df


def prepare(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Полный пайплайн: пропуски → фичи → отбор колонок в правильном порядке."""
    df = fill_missing(df, artifacts["income_median"], artifacts["dependents_mode"])
    df = add_features(df)
    return df[artifacts["feature_names"]]


def pd_to_score(pd_value: float, base_score: int = 600, base_odds: float = 50, pdo: int = 20) -> float:
    """PD → банковский скор (FICO-style)."""
    import numpy as np

    pd_value = float(np.clip(pd_value, 1e-6, 1 - 1e-6))
    odds = (1 - pd_value) / pd_value
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return float(offset + factor * np.log(odds))
