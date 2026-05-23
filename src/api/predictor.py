"""Couche d'inférence : charge les modèles depuis disk + sert une prédiction.

Stratégie :
- Pour chaque requête, on tente d'abord le modèle SPÉCIFIQUE au tribunal demandé
- Si absent → on retombe sur le modèle global
- Les modèles sont mis en cache (lazy load) au premier appel
"""
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from loguru import logger

from ..config import settings
from ..features.build import build_features, feature_matrix
from ..training.train import _slugify_tribunal, GLOBAL_MODEL_NAME


class ModelRegistry:
    """Charge les modèles à la demande, garde un cache en mémoire."""

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._index: Optional[dict] = None

    def _load_index(self) -> dict:
        if self._index is not None:
            return self._index
        path = settings.models_path / "index.json"
        if not path.exists():
            raise FileNotFoundError(
                "Aucun modèle entraîné. Lance d'abord :\n"
                "  1) python -m src.training.bootstrap   # génère les données synthétiques\n"
                "  2) python -m src.training.train       # entraîne les modèles"
            )
        self._index = json.loads(path.read_text(encoding="utf-8"))
        return self._index

    def list_tribunals(self) -> list[str]:
        return list(self._load_index().get("tribunal_to_slug", {}).keys())

    def get_metrics(self) -> dict:
        idx = self._load_index()
        return {
            "trained_at": idx.get("trained_at"),
            "total_rows": idx.get("total_rows"),
            "features_version": idx.get("features_version"),
            "models": [
                {
                    "name": m["name"],
                    "slug": m["slug"],
                    "n_train": m["n_train"],
                    "n_test": m["n_test"],
                    "mae": m["mae"],
                    "rmse": m["rmse"],
                    "r2": m["r2"],
                    "mape": m["mape"],
                }
                for m in idx.get("models", [])
            ],
        }

    def _load_artifact(self, slug: str) -> Optional[dict]:
        if slug in self._cache:
            return self._cache[slug]
        path = settings.models_path / f"{slug}.joblib"
        if not path.exists():
            return None
        artifact = joblib.load(path)
        self._cache[slug] = artifact
        return artifact

    def predict(self, payload: dict) -> dict:
        # Choix du modèle : tribunal-spécifique sinon global
        tribunal = payload.get("tribunal", "")
        index = self._load_index()
        slug = index.get("tribunal_to_slug", {}).get(tribunal)
        used = "tribunal"
        artifact = self._load_artifact(slug) if slug else None
        if artifact is None:
            artifact = self._load_artifact("global")
            used = "global"
        if artifact is None:
            raise FileNotFoundError("Aucun modèle global entraîné. Lance `python -m src.training.train`.")

        # Détection modèle obsolète (features changent entre versions)
        from ..features.build import FEATURES_VERSION
        stored_ver = artifact.get("features_version")
        if stored_ver != FEATURES_VERSION:
            raise RuntimeError(
                f"Modèle obsolète : features_version='{stored_ver}' mais le code attend '{FEATURES_VERSION}'. "
                f"Ré-entraîne avec : python -m src.cli train"
            )

        # Build features
        df = pd.DataFrame([payload])
        df = build_features(df, training=False)
        X = feature_matrix(df)

        # Prédiction + intervalle approximatif via les arbres du RF
        pipeline = artifact["pipeline"]
        y_pred = float(pipeline.predict(X)[0])

        rf = pipeline.named_steps["rf"]
        X_transformed = pipeline.named_steps["prep"].transform(X)
        tree_preds = [tree.predict(X_transformed)[0] for tree in rf.estimators_]
        p_low, p_high = float(np.percentile(tree_preds, 10)), float(np.percentile(tree_preds, 90))

        confidence = _confidence_from_spread(p_low, p_high, y_pred)

        return {
            "adjudicated_price_predicted": round(y_pred, 2),
            "low_estimate": round(p_low, 2),
            "high_estimate": round(p_high, 2),
            "confidence": confidence,
            "model_used": used,
            "model_name": artifact["name"],
            "features_version": artifact["features_version"],
            "model_metrics": {
                "mae": artifact["metrics"]["mae"],
                "r2": artifact["metrics"]["r2"],
                "mape": artifact["metrics"]["mape"],
            },
        }


def _confidence_from_spread(low: float, high: float, mean: float) -> int:
    """Heuristique : plus l'écart relatif est petit, plus la confiance est haute (0-100)."""
    if mean <= 0:
        return 0
    spread_rel = (high - low) / mean
    # spread_rel ~ 0  → confiance ~95
    # spread_rel ~ 1.0 → confiance ~45
    score = 100 - min(60, int(spread_rel * 60))
    return max(0, min(100, score))


# Imports tardifs pour éviter une dépendance dure pendant le bootstrap
import numpy as np  # noqa: E402

registry = ModelRegistry()
