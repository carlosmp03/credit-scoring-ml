"""FastAPI-сервис для кредитного скоринга.

Запуск (из корня проекта):
    python -m uvicorn api.app:app --reload --port 8000

Endpoints:
    GET  /              — health check + метаданные модели
    GET  /health        — простой health check
    POST /predict       — предсказание для одного клиента
    POST /predict/batch — предсказание для списка клиентов
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from api.preprocessing import load_artifacts, prepare, pd_to_score

# ---------------------------------------------------------------------------
# Загрузка модели и артефактов на старте
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Калиброванная модель — её и используем в продакшене
_model_path = MODELS_DIR / "cb_model_calibrated.pkl"
if not _model_path.exists():
    # Fallback на некалиброванную, если калибровка не сделана
    _model_path = MODELS_DIR / "cb_model.pkl"
MODEL = joblib.load(_model_path)
ARTIFACTS = load_artifacts(MODELS_DIR)

_metrics_path = MODELS_DIR / "metrics.json"
METRICS = json.loads(_metrics_path.read_text()) if _metrics_path.exists() else {}

# Дефолтный порог одобрения берём из бизнес-симуляции, если есть, иначе 0.5
DEFAULT_THRESHOLD = float(METRICS.get("best_threshold_business", 0.5))

# ---------------------------------------------------------------------------
# Pydantic-схемы
# ---------------------------------------------------------------------------
class ClientFeatures(BaseModel):
    """Сырые фичи клиента — ровно те же колонки, что в исходном CSV (минус таргет)."""

    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Доля используемых необеспеченных кредитных линий")
    age: int = Field(..., ge=18, le=120, description="Возраст клиента")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime30-59DaysPastDueNotWorse", description="Просрочки 30-59 дней"
    )
    DebtRatio: float = Field(..., ge=0, description="Отношение долга к доходу")
    MonthlyIncome: Optional[float] = Field(None, ge=0, description="Месячный доход (может быть null)")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, description="Число открытых кредитных линий")
    NumberOfTimes90DaysLate: int = Field(..., ge=0, description="Просрочки 90+ дней")
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, description="Число ипотечных кредитов")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, alias="NumberOfTime60-89DaysPastDueNotWorse", description="Просрочки 60-89 дней"
    )
    NumberOfDependents: Optional[int] = Field(None, ge=0, description="Число иждивенцев (может быть null)")


class PredictionResponse(BaseModel):
    probability_of_default: float = Field(..., description="Калиброванная вероятность дефолта (0..1)")
    score: float = Field(..., description="Банковский скор (FICO-style, base=600, PDO=20)")
    decision: str = Field(..., description="approve | deny")
    threshold_used: float = Field(..., description="Порог, по которому принято решение")


class BatchRequest(BaseModel):
    clients: List[ClientFeatures]
    threshold: Optional[float] = Field(None, description="Порог одобрения; по умолчанию — бизнес-оптимальный")


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Scoring API",
    description="REST API для оценки вероятности дефолта по модели CatBoost (калиброванной isotonic).",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "service": "credit-scoring-api",
        "model_path": str(_model_path.name),
        "n_features": len(ARTIFACTS["feature_names"]),
        "default_threshold": DEFAULT_THRESHOLD,
        "metrics": METRICS,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _features_to_dataframe(clients: List[ClientFeatures]) -> pd.DataFrame:
    """Pydantic-объекты → DataFrame с теми же колонками, что в обучающем CSV."""
    rows = []
    for c in clients:
        # by_alias=True, чтобы получить колонки с дефисами как в датасете
        rows.append(c.model_dump(by_alias=True))
    return pd.DataFrame(rows)


def _predict(df_raw: pd.DataFrame, threshold: float) -> List[PredictionResponse]:
    try:
        X = prepare(df_raw, ARTIFACTS)
        probs = MODEL.predict_proba(X)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось обработать вход: {e}")

    out = []
    for p in probs:
        out.append(PredictionResponse(
            probability_of_default=float(p),
            score=pd_to_score(float(p)),
            decision="approve" if p < threshold else "deny",
            threshold_used=threshold,
        ))
    return out


@app.post("/predict", response_model=PredictionResponse)
def predict_one(client: ClientFeatures, threshold: Optional[float] = None):
    thr = float(threshold) if threshold is not None else DEFAULT_THRESHOLD
    df = _features_to_dataframe([client])
    return _predict(df, thr)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    thr = float(req.threshold) if req.threshold is not None else DEFAULT_THRESHOLD
    df = _features_to_dataframe(req.clients)
    return BatchResponse(predictions=_predict(df, thr))
