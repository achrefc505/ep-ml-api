"""Feature engineering — fonction unique réutilisée par training ET inference.

⚠️ Toute modification ici impacte les modèles déjà entraînés.
   En cas de changement majeur, bumper FEATURES_VERSION pour invalider le cache.

v2 (2026-05) : ajout postal_code + arrondissement Paris + lat/lng.
   Le code postal est le SIGNAL CRITIQUE pour Paris/Lyon/Marseille — un
   appartement au 16e n'a pas le même prix au m² qu'au 11e (~30% d'écart).
"""
import re
import pandas as pd
import numpy as np

FEATURES_VERSION = "v2"

NUMERIC_FEATURES = [
    "surface",
    "rooms",
    "initial_price",
    "price_per_sqm_initial",
    "year",
    "month",
    "latitude",
    "longitude",
    "arrondissement",  # 1-20 pour Paris, 1-9 pour Lyon, 1-16 pour Marseille, 0 ailleurs
]

CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "region",
    "postal_code",     # ← LA feature manquante : 75011 vs 75016 sont 2 mondes
]

TARGET = "adjudicated_price"


# Villes à arrondissements numérotés
_ARRONDISSEMENT_PREFIX = {
    "75": (1, 20),   # Paris 75001-75020
    "13": (1, 16),   # Marseille 13001-13016 (préfixe départemental)
    "69": (1, 9),    # Lyon 69001-69009
}


def _extract_arrondissement(postal_code: str) -> int:
    """Extrait l'arrondissement à partir du code postal (Paris/Lyon/Marseille).
    Retourne 0 si pas dans une ville à arrondissements.
    """
    if not postal_code:
        return 0
    pc = str(postal_code).strip()
    if len(pc) != 5 or not pc.isdigit():
        return 0
    prefix = pc[:2]
    rng = _ARRONDISSEMENT_PREFIX.get(prefix)
    if rng is None:
        return 0
    try:
        arr = int(pc[2:])
    except ValueError:
        return 0
    lo, hi = rng
    return arr if lo <= arr <= hi else 0


def build_features(df: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    """Transforme un DataFrame brut en features prêtes pour le modèle."""
    out = df.copy()

    # Numériques de base
    out["surface"] = out["surface"].astype(float).clip(lower=1)
    out["rooms"] = out["rooms"].fillna(0).astype(int).clip(lower=0, upper=20)
    out["initial_price"] = out["initial_price"].astype(float).clip(lower=1)

    # Feature dérivée : prix au m² de la mise à prix
    out["price_per_sqm_initial"] = out["initial_price"] / out["surface"]

    # Temporel
    if "adjudication_date" in out.columns:
        out["adjudication_date"] = pd.to_datetime(out["adjudication_date"], errors="coerce")
        out["year"] = out["adjudication_date"].dt.year.fillna(2024).astype(int)
        out["month"] = out["adjudication_date"].dt.month.fillna(6).astype(int)
    else:
        from datetime import datetime
        out["year"] = datetime.utcnow().year
        out["month"] = datetime.utcnow().month

    # Code postal → string normalisé d'abord (sert à imputer lat/lng)
    if "postal_code" in out.columns:
        out["postal_code"] = (
            out["postal_code"]
            .fillna("00000")
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # cas pandas 75011.0
            .str.zfill(5)
            .str[:5]
        )
    else:
        out["postal_code"] = "00000"

    out["arrondissement"] = out["postal_code"].apply(_extract_arrondissement).astype(int)

    # Géo : lat/lng — si absent, imputation depuis postal_code (centroïde connu)
    from .location import lookup_latlng

    for col in ("latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    needs_impute = out["latitude"].isna() | out["longitude"].isna()
    if needs_impute.any():
        imputed = out.loc[needs_impute, "postal_code"].apply(lookup_latlng)
        out.loc[needs_impute, "latitude"] = imputed.apply(
            lambda x: x[0] if x else np.nan
        )
        out.loc[needs_impute, "longitude"] = imputed.apply(
            lambda x: x[1] if x else np.nan
        )

    # Si toujours NaN après imputation : médiane (sinon RandomForest plante)
    for col in ("latitude", "longitude"):
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(out[col].median())

    # Autres catégorielles
    for col in ("property_type", "city", "region"):
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str).str.strip()
        else:
            out[col] = "Unknown"

    if training:
        if TARGET not in out.columns:
            raise ValueError(f"Target manquante : {TARGET}")
        out[TARGET] = out[TARGET].astype(float).clip(lower=1)
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
