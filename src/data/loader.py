"""Chargement des données d'entraînement.

Sources interchangeables (via settings.data_source) :
- 'csv'      → CSV synthétique (bootstrap)
- 'sql'      → EncheresPredict_Raw.adjudications (production)
- 'hybrid'   → SQL + complète avec synthétique si < HYBRID_MIN_REAL lignes

Le DataFrame retourné a TOUJOURS le même schéma pour que la suite du
pipeline (features, training, prediction) soit agnostique de la source.
"""
import pandas as pd
from loguru import logger

from ..config import settings


def normalize_tribunal(name: str | None) -> str:
    """Mappe les variantes vers le nom canonical du tribunal.

    Exemples :
      'Tribunal Judiciaire de Paris' → 'TJ Paris'
      'TJ de Lyon'                   → 'TJ Lyon'
      'tj paris'                     → 'TJ Paris'
      'TGI Marseille' (ancien nom)   → 'TJ Marseille'
    """
    if not name or str(name).strip().lower() in {"", "unknown", "nan", "none"}:
        return "Unknown"
    import re as _re

    s = str(name).strip()
    # Normalise espaces
    s = _re.sub(r"\s+", " ", s)
    # Retire variantes "Tribunal Judiciaire de", "TGI", "TJ de"
    patterns = [
        (r"^Tribunal\s+Judiciaire\s+de\s+", "TJ "),
        (r"^Tribunal\s+Judiciaire\s+d['']", "TJ "),
        (r"^Tribunal\s+Judiciaire\s+", "TJ "),
        (r"^TGI\s+", "TJ "),                # ancienne dénomination
        (r"^TJ\s+de\s+", "TJ "),
        (r"^TJ\s+d['']", "TJ "),
        (r"^Cour\s+d['']?[Aa]ppel\s+de\s+", "CA "),
    ]
    for pat, repl in patterns:
        s = _re.sub(pat, repl, s, flags=_re.IGNORECASE)
    # Casse standardisée : 'TJ ' + Nom propre
    if s.lower().startswith("tj "):
        rest = s[3:].strip()
        # Titre case sur la partie ville
        s = "TJ " + rest[:1].upper() + rest[1:] if rest else "TJ Unknown"
    return s


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

# Si en mode hybride et qu'on a moins que ce seuil de lignes réelles, on
# complète avec du synthétique pour avoir un modèle utilisable
HYBRID_MIN_REAL = 200


# Mapping postal_code → région (fallback quand region est NULL)
POSTAL_TO_REGION = {
    "75": "Île-de-France", "77": "Île-de-France", "78": "Île-de-France",
    "91": "Île-de-France", "92": "Île-de-France", "93": "Île-de-France",
    "94": "Île-de-France", "95": "Île-de-France",
    "13": "PACA", "06": "PACA", "83": "PACA", "84": "PACA", "05": "PACA", "04": "PACA",
    "69": "Auvergne-Rhône-Alpes", "38": "Auvergne-Rhône-Alpes",
    "73": "Auvergne-Rhône-Alpes", "74": "Auvergne-Rhône-Alpes",
    "33": "Nouvelle-Aquitaine", "64": "Nouvelle-Aquitaine", "40": "Nouvelle-Aquitaine",
    "59": "Hauts-de-France", "62": "Hauts-de-France", "80": "Hauts-de-France",
    "44": "Pays de la Loire", "49": "Pays de la Loire", "85": "Pays de la Loire",
    "31": "Occitanie", "34": "Occitanie", "30": "Occitanie", "11": "Occitanie",
    "67": "Grand Est", "68": "Grand Est", "57": "Grand Est",
    "35": "Bretagne", "29": "Bretagne", "22": "Bretagne", "56": "Bretagne",
}


def load_training_data() -> pd.DataFrame:
    """Charge les données depuis la source configurée."""
    src = settings.data_source.lower()
    if src == "sql":
        df = _load_from_sql()
    elif src == "hybrid":
        df = _load_hybrid()
    else:
        df = _load_from_csv()

    df = _clean(df)
    _validate_schema(df)
    logger.info("Données chargées : {} lignes ({})", len(df), src)
    return df


def _load_from_csv() -> pd.DataFrame:
    path = settings.training_csv_full_path
    if not path.exists():
        raise FileNotFoundError(
            f"CSV introuvable : {path}. "
            "Lance d'abord `python -m src.training.bootstrap` pour générer un dataset synthétique."
        )
    df = pd.read_csv(path, parse_dates=["adjudication_date"])
    df["__source__"] = "csv"
    return df


def _load_from_sql() -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True)
    query = """
        SELECT
            tribunal, city, region, property_type,
            surface, rooms, initial_price, adjudicated_price,
            adjudication_date, postal_code, [address], floor,
            [description], latitude, longitude
        FROM dbo.adjudications
        WHERE adjudicated_price IS NOT NULL
          AND initial_price     IS NOT NULL
          AND surface           IS NOT NULL
          AND surface           > 5
          AND adjudicated_price > 0
    """
    df = pd.read_sql(query, engine, parse_dates=["adjudication_date"])
    df["__source__"] = "sql"
    return df


def _load_hybrid() -> pd.DataFrame:
    """Charge SQL en priorité, complète avec synthétique INTELLIGEMMENT.

    Stratégie par tribunal : on garde les vraies données pures pour les
    tribunaux qui ont déjà >= MIN_SAMPLES_PER_TRIBUNAL lignes réelles
    (sinon le synthétique pollue le signal local).

    Pour les tribunaux insuffisants : on ajoute du synthétique CIBLÉ
    (uniquement les villes manquantes) pour entraîner le modèle global.
    """
    from ..config import settings as _settings

    try:
        real = _load_from_sql()
    except Exception as e:
        logger.warning("SQL indisponible ({}). Fallback CSV complet.", e)
        return _load_from_csv()

    if len(real) >= HYBRID_MIN_REAL:
        logger.info("Hybride : {} lignes réelles ≥ seuil {} → pas de synthétique", len(real), HYBRID_MIN_REAL)
        return real

    # Détermine quels tribunaux ont assez de vraies données
    from .loader import normalize_tribunal as _norm
    real["tribunal"] = real["tribunal"].fillna("Unknown").astype(str).apply(_norm)
    real_counts = real["tribunal"].value_counts()
    well_covered = set(real_counts[real_counts >= _settings.min_samples_per_tribunal].index)
    if well_covered:
        logger.info(
            "Hybride : {} tribunaux bien couverts (>= {}) → exclus du synthétique : {}",
            len(well_covered), _settings.min_samples_per_tribunal,
            ", ".join(sorted(well_covered)),
        )

    try:
        synth = _load_from_csv()
    except FileNotFoundError:
        from ..training.bootstrap import generate
        logger.info("Hybride : génération synthétique à la volée...")
        synth = generate(n_rows=5000)
        synth["__source__"] = "csv"

    # Filtre le synthétique : on retire les tribunaux que les vraies données couvrent déjà
    if "tribunal" in synth.columns and well_covered:
        before = len(synth)
        synth["tribunal"] = synth["tribunal"].astype(str).apply(_norm)
        synth = synth[~synth["tribunal"].isin(well_covered)]
        logger.info("Hybride : synthétique filtré {} → {} lignes", before, len(synth))

    logger.info(
        "Hybride final : {} réel (purs sur tribunaux couverts) + {} synthétique",
        len(real), len(synth),
    )
    return pd.concat([real, synth], ignore_index=True)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage défensif : NULLs, outliers, fallbacks region."""
    if df.empty:
        return df

    # Ajout colonne postal_code si manquante (cas CSV)
    if "postal_code" not in df.columns:
        df["postal_code"] = None

    # Fallback region depuis postal_code
    region_missing = df["region"].isna() | (df["region"] == "")
    if region_missing.any() and "postal_code" in df.columns:
        df.loc[region_missing, "region"] = (
            df.loc[region_missing, "postal_code"]
              .astype(str).str[:2].map(POSTAL_TO_REGION)
        )

    # Defaults pour les colonnes critiques
    df["tribunal"] = df["tribunal"].fillna("Unknown").astype(str).str.strip()
    df["tribunal"] = df["tribunal"].apply(normalize_tribunal)
    df["city"] = df["city"].fillna("Unknown").astype(str).str.strip()
    df["region"] = df["region"].fillna("Unknown").astype(str).str.strip()
    df["property_type"] = df["property_type"].fillna("Appartement").astype(str).str.strip()
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce").fillna(0).clip(0, 30).astype(int)

    # Cast numériques
    for col in ("surface", "initial_price", "adjudicated_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop lignes inutilisables
    df = df.dropna(subset=["surface", "initial_price", "adjudicated_price"])
    df = df[(df["surface"] > 5) & (df["surface"] < 10000)]
    df = df[(df["initial_price"] > 100) & (df["initial_price"] < 50_000_000)]
    df = df[(df["adjudicated_price"] > 100) & (df["adjudicated_price"] < 50_000_000)]

    # Clip extremes (1% / 99%)
    if len(df) > 50:
        for col in ("initial_price", "adjudicated_price"):
            low, high = df[col].quantile([0.01, 0.99])
            df = df[(df[col] >= low) & (df[col] <= high)]

    # Sanity sur le ratio adjudication / mise à prix
    df["__ratio__"] = df["adjudicated_price"] / df["initial_price"]
    df = df[(df["__ratio__"] >= 0.3) & (df["__ratio__"] <= 10.0)]
    df = df.drop(columns=["__ratio__"])

    return df.reset_index(drop=True)


def _validate_schema(df: pd.DataFrame):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if len(df) == 0:
        raise ValueError(
            "0 ligne après nettoyage. "
            "Vérifie les données scrapées : initial_price, adjudicated_price, surface ne doivent pas être tous NULL."
        )


def db_quality_report() -> dict:
    """Diagnostic : compte les lignes en DB par qualité (utilisable pour ML ou non)."""
    from sqlalchemy import create_engine, text

    engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True)
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN adjudicated_price IS NOT NULL THEN 1 ELSE 0 END) AS with_adj_price,
                SUM(CASE WHEN initial_price IS NOT NULL THEN 1 ELSE 0 END)     AS with_init_price,
                SUM(CASE WHEN surface IS NOT NULL AND surface > 5 THEN 1 ELSE 0 END) AS with_surface,
                SUM(CASE WHEN tribunal IS NOT NULL THEN 1 ELSE 0 END)          AS with_tribunal,
                SUM(CASE WHEN city IS NOT NULL THEN 1 ELSE 0 END)              AS with_city,
                SUM(CASE
                    WHEN adjudicated_price IS NOT NULL AND initial_price IS NOT NULL
                     AND surface IS NOT NULL AND surface > 5
                     AND adjudicated_price > 0 THEN 1 ELSE 0 END) AS ml_usable
            FROM dbo.adjudications
        """)).first()

        per_tribunal = conn.execute(text("""
            SELECT TOP 20 tribunal, COUNT(*) AS n
            FROM dbo.adjudications
            WHERE adjudicated_price IS NOT NULL
              AND initial_price IS NOT NULL
              AND surface > 5
            GROUP BY tribunal
            ORDER BY n DESC
        """)).all()

    return {
        "total": result.total,
        "with_adjudicated_price": result.with_adj_price,
        "with_initial_price": result.with_init_price,
        "with_surface": result.with_surface,
        "with_tribunal": result.with_tribunal,
        "with_city": result.with_city,
        "ml_usable": result.ml_usable,
        "per_tribunal": [(r.tribunal or "(NULL)", r.n) for r in per_tribunal],
    }
