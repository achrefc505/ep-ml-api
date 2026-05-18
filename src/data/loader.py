"""Chargement des données d'entraînement.

Deux sources interchangeables :
- CSV (rapide, déterministe, idéal POC)
- SQL Server EncheresPredict_Raw.adjudications (production)

Le DataFrame retourné a TOUJOURS le même schéma pour que la suite du
pipeline (features, training, prediction) soit agnostique de la source.
"""
import pandas as pd
from loguru import logger

from ..config import settings


REQUIRED_COLUMNS = [
    "tribunal",
    "city",
    "region",
    "property_type",
    "surface",
    "rooms",
    "initial_price",
    "adjudicated_price",
    "adjudication_date",
]


def load_training_data() -> pd.DataFrame:
    """Charge les données depuis la source configurée."""
    if settings.data_source == "sql":
        df = _load_from_sql()
    else:
        df = _load_from_csv()

    _validate_schema(df)
    logger.info("Données chargées : {} lignes depuis {}", len(df), settings.data_source)
    return df


def _load_from_csv() -> pd.DataFrame:
    path = settings.training_csv_full_path
    if not path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {path}. "
            "Lance d'abord `python -m src.training.bootstrap` pour générer un dataset synthétique."
        )
    df = pd.read_csv(path, parse_dates=["adjudication_date"])
    return df


def _load_from_sql() -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True)
    query = """
        SELECT
            tribunal, city, region, property_type,
            surface, rooms, initial_price, adjudicated_price,
            adjudication_date
        FROM dbo.adjudications
        WHERE adjudicated_price IS NOT NULL
          AND initial_price     IS NOT NULL
          AND surface           IS NOT NULL
          AND surface           > 5
          AND adjudicated_price > 0
    """
    df = pd.read_sql(query, engine, parse_dates=["adjudication_date"])
    return df


def _validate_schema(df: pd.DataFrame):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
