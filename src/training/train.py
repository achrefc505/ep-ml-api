"""Entraînement des modèles Random Forest.

Stratégie :
1. Modèle GLOBAL entraîné sur toutes les données (fallback)
2. Un modèle PAR TRIBUNAL si >= MIN_SAMPLES_PER_TRIBUNAL lignes

Chaque modèle est sauvegardé en .joblib avec ses métriques (MAE, RMSE, R², MAPE).
"""
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..config import settings
from ..data.loader import load_training_data
from ..features.build import (
    build_features, feature_matrix,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, FEATURES_VERSION,
)


GLOBAL_MODEL_NAME = "__global__"


@dataclass
class TrainingMetrics:
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float
    mape: float  # mean absolute percentage error

    @classmethod
    def compute(cls, y_true, y_pred):
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        return cls(n_train=0, n_test=len(y_true), mae=mae, rmse=rmse, r2=r2, mape=mape)


def _build_pipeline() -> Pipeline:
    """Pipeline sklearn : ColumnTransformer (OneHot pour catégorielles) + RandomForest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=settings.rf_n_estimators,
        max_depth=settings.rf_max_depth,
        min_samples_leaf=settings.rf_min_samples_leaf,
        random_state=settings.rf_random_state,
        n_jobs=-1,
    )
    return Pipeline([("prep", preprocessor), ("rf", model)])


def _slugify_tribunal(name: str) -> str:
    """Transforme 'TJ Paris' en 'tj_paris' (nom de fichier safe)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower()).strip("_")
    return s or "unknown"


def _train_one_model(df: pd.DataFrame, name: str) -> dict:
    """Entraîne et sauvegarde un modèle, retourne ses métriques."""
    X = feature_matrix(df)
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.rf_random_state
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = TrainingMetrics.compute(y_test, y_pred)
    metrics.n_train = len(X_train)
    metrics.n_test = len(X_test)

    artifact = {
        "name": name,
        "features_version": FEATURES_VERSION,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": asdict(metrics),
        "rf_params": {
            "n_estimators": settings.rf_n_estimators,
            "max_depth": settings.rf_max_depth,
            "min_samples_leaf": settings.rf_min_samples_leaf,
        },
        "pipeline": pipeline,
    }

    slug = _slugify_tribunal(name) if name != GLOBAL_MODEL_NAME else "global"
    file_path = settings.models_path / f"{slug}.joblib"
    joblib.dump(artifact, file_path)

    logger.info(
        "✓ Modèle {} sauvé ({:>5} train / {:>5} test) — MAE={:>10,.0f}€ RMSE={:>10,.0f}€ R²={:.3f} MAPE={:.1f}%",
        name, metrics.n_train, metrics.n_test, metrics.mae, metrics.rmse, metrics.r2, metrics.mape,
    )

    return {
        "name": name,
        "slug": slug,
        "file": str(file_path),
        **asdict(metrics),
    }


def train_all() -> dict:
    """Entraîne le modèle global + un modèle par tribunal éligible."""
    raw = load_training_data()
    df = build_features(raw, training=True)

    if len(df) < 50:
        raise RuntimeError(
            f"Pas assez de données pour entraîner ({len(df)} lignes). "
            "Minimum recommandé : 50. Génère du synthétique avec `python -m src.training.bootstrap`."
        )

    results = {"models": []}

    # 1. Modèle global (fallback)
    results["models"].append(_train_one_model(df, GLOBAL_MODEL_NAME))

    # 2. Modèles par tribunal éligibles
    tribunal_counts = df["tribunal"].fillna("Unknown").value_counts()
    eligible = tribunal_counts[tribunal_counts >= settings.min_samples_per_tribunal]
    logger.info(
        "Modèles par tribunal : {}/{} tribunaux atteignent le seuil de {} lignes",
        len(eligible), len(tribunal_counts), settings.min_samples_per_tribunal,
    )
    for tribunal in eligible.index:
        if tribunal == "Unknown":
            continue
        sub = df[df["tribunal"] == tribunal]
        results["models"].append(_train_one_model(sub, tribunal))

    # 3. Index des modèles
    index_path = settings.models_path / "index.json"
    index = {
        "features_version": FEATURES_VERSION,
        "trained_at": datetime.utcnow().isoformat(),
        "total_rows": len(df),
        "models": results["models"],
        "tribunal_to_slug": {
            tribunal: _slugify_tribunal(tribunal) for tribunal in eligible.index
        },
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Index écrit : {}", index_path)

    return index


if __name__ == "__main__":
    train_all()
