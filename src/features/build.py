"""Feature engineering — fonction unique réutilisée par training ET inference.

⚠️ Toute modification ici impacte les modèles déjà entraînés.
   En cas de changement majeur, bumper FEATURES_VERSION pour invalider le cache.
"""
import pandas as pd
import numpy as np

FEATURES_VERSION = "v1"

# Colonnes numériques après transformation
NUMERIC_FEATURES = [
    "surface",
    "rooms",
    "initial_price",
    "price_per_sqm_initial",
    "year",
    "month",
]

# Colonnes catégorielles (encodées par one-hot dans le pipeline sklearn)
CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "region",
]

TARGET = "adjudicated_price"


def build_features(df: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    """Transforme un DataFrame brut en features prêtes pour le modèle.

    Args:
        df: doit contenir au minimum les colonnes requises (voir loader.REQUIRED_COLUMNS
            ou subset pour l'inférence)
        training: si True, valide la présence de la target
    """
    out = df.copy()

    # Nettoyage basique
    out["surface"] = out["surface"].astype(float).clip(lower=1)
    out["rooms"] = out["rooms"].fillna(0).astype(int).clip(lower=0, upper=20)
    out["initial_price"] = out["initial_price"].astype(float).clip(lower=1)

    # Features dérivées
    out["price_per_sqm_initial"] = out["initial_price"] / out["surface"]

    if "adjudication_date" in out.columns:
        out["adjudication_date"] = pd.to_datetime(out["adjudication_date"], errors="coerce")
        out["year"] = out["adjudication_date"].dt.year.fillna(2024).astype(int)
        out["month"] = out["adjudication_date"].dt.month.fillna(6).astype(int)
    else:
        from datetime import datetime
        out["year"] = datetime.utcnow().year
        out["month"] = datetime.utcnow().month

    # Normalisation catégorielle minimale (cohérence majuscules)
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str).str.strip()
        else:
            out[col] = "Unknown"

    if training:
        if TARGET not in out.columns:
            raise ValueError(f"Target manquante : {TARGET}")
        out[TARGET] = out[TARGET].astype(float).clip(lower=1)
        # Drop lignes aberrantes (target trop éloignée du raisonnable)
        before = len(out)
        out = out[(out[TARGET] >= 1000) & (out[TARGET] <= 50_000_000)]
        if before != len(out):
            from loguru import logger
            logger.info("Filtre target : {} → {} lignes", before, len(out))

    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait uniquement les colonnes features (sans la target)."""
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[cols]
