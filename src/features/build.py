"""Feature engineering — fonction unique réutilisée par training ET inference.

⚠️ Toute modification ici impacte les modèles déjà entraînés.
   En cas de changement majeur, bumper FEATURES_VERSION pour invalider le cache.

v3 (2026-05) : robustesse aux CP partiels + features depuis description.
   - Extraction arrondissement depuis la `city` quand postal_code est tronqué
     (cas Licitor : 'Paris 16e' + postal_code='75' → arr=16 reconstitué)
   - Nouvelle catégorie 'district_key' = city + arrondissement combinés
   - Parsing description : DPE, étage, balcon, parking, ascenseur, rénovation,
     occupation locative
   - Features dérivées : surface_per_room, log_initial_price

v2 (2026-05) : postal_code + arrondissement + lat/lng
v1 (2026-04) : version initiale
"""
import re
import pandas as pd
import numpy as np

FEATURES_VERSION = "v3"

NUMERIC_FEATURES = [
    "surface",
    "rooms",
    "initial_price",
    "price_per_sqm_initial",
    "log_initial_price",
    "surface_per_room",
    "year",
    "month",
    "latitude",
    "longitude",
    "arrondissement",
    "floor_num",
    "dpe_ordinal",
    "has_balcony",
    "has_parking",
    "has_elevator",
    "is_renovated",
    "is_rented",
]

CATEGORICAL_FEATURES = [
    "property_type",
    "city",
    "region",
    "postal_code",
    "district_key",   # ← v3 : clé combinée robuste
]

TARGET = "adjudicated_price"

_ARRONDISSEMENT_PREFIX = {
    "75": (1, 20),   # Paris
    "13": (1, 16),   # Marseille
    "69": (1, 9),    # Lyon
}

# Regex extraction arrondissement depuis nom de ville
# Match : "Paris 16e", "Paris 16ème", "Paris 16eme", "Lyon 6", "Marseille 8ème"
_ARRONDISSEMENT_FROM_CITY = re.compile(
    r"(paris|lyon|marseille)\s*(\d{1,2})\s*(?:e|er|ème|eme|ᵉ|nd|nde)?\b",
    re.IGNORECASE,
)


def _extract_arrondissement_from_postal(postal_code: str) -> int:
    """Renvoie l'arrondissement (1-20) si postal_code est un CP complet style 75011.
    Sinon 0.
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


def _extract_arrondissement_from_city(city: str) -> int:
    """Cherche '16e', '11ème'... dans une string ville (cas CP partiel)."""
    if not city:
        return 0
    m = _ARRONDISSEMENT_FROM_CITY.search(str(city))
    if not m:
        return 0
    base_city = m.group(1).lower()
    arr = int(m.group(2))
    bounds = {"paris": 20, "lyon": 9, "marseille": 16}[base_city]
    return arr if 1 <= arr <= bounds else 0


def _normalize_postal(pc) -> str:
    """Normalise un code postal en string 5 chars (padding 0 à droite si partiel)."""
    if pc is None or (isinstance(pc, float) and np.isnan(pc)):
        return "00000"
    s = str(pc).strip().replace(".0", "")
    if not s:
        return "00000"
    # Garde uniquement les chiffres
    s = re.sub(r"\D", "", s)
    if not s:
        return "00000"
    if len(s) >= 5:
        return s[:5]
    # Code partiel (ex: "75" ou "75011" tronqué) → on padde à droite avec 0
    # pour signaler "département connu, arrondissement inconnu"
    return s.ljust(5, "0")


def _build_district_key(city: str, arr: int, postal_code: str) -> str:
    """Clé robuste de localisation pour le one-hot.

    Stratégie :
    - Si arrondissement détecté : "paris-16", "lyon-06", "marseille-08"
    - Sinon si CP complet : "{cp}"
    - Sinon si CP département connu : "{city_norm}-dept{XX}"
    - Sinon : "{city_norm}"
    """
    city_norm = (city or "unknown").strip().lower()
    # Retire les numéros d'arrondissement pour normaliser la ville
    city_norm = re.sub(r"\s*\d{1,2}\s*(?:e|er|ème|eme|nd|nde)?\b", "", city_norm).strip()
    city_norm = city_norm.replace(" ", "_")

    if arr > 0 and city_norm in ("paris", "lyon", "marseille"):
        return f"{city_norm}-{arr:02d}"

    pc = postal_code or ""
    # CP complet (pas de padding ajouté → vraies 5 digits non-zéro à droite)
    if len(pc) == 5 and pc.isdigit() and not pc.endswith("000"):
        return pc

    if len(pc) >= 2:
        return f"{city_norm}-dept{pc[:2]}"

    return city_norm or "unknown"


# ---------------------------------------------------------------------------
# Parsing description (regex) — features quali extraites du texte
# ---------------------------------------------------------------------------
_RE_DPE = re.compile(r"\bDPE[\s:]*([A-G])\b", re.IGNORECASE)
_RE_FLOOR = re.compile(r"\b(\d{1,2})\s*(?:e|er|ème|eme)\s*(?:étage|etage)", re.IGNORECASE)
_RE_RDC = re.compile(r"\b(?:rez[-\s]de[-\s]chauss[ée]e|RDC|R\.D\.C\.?)\b", re.IGNORECASE)
_RE_BALCONY = re.compile(r"\b(?:balcon|terrasse|loggia)s?\b", re.IGNORECASE)
_RE_PARKING = re.compile(r"\b(?:parking|garage|box|stationnement)s?\b", re.IGNORECASE)
_RE_ELEVATOR = re.compile(r"\b(?:ascenseur|élévateur)s?\b", re.IGNORECASE)
_RE_RENOVATED = re.compile(r"\b(?:rénov[ée]e?s?|refait|neuf|neuve|moderne|moderniser?)\b", re.IGNORECASE)
_RE_RENTED = re.compile(r"\b(?:lou[ée]|occup[ée]e?|locataire|bail|baux|en\s+place)\b", re.IGNORECASE)

_DPE_TO_ORDINAL = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}


def _parse_floor(text: str, fallback_floor: str | None = None) -> int:
    """0 = RDC, 1-30 = étage. Si pas trouvé, fallback à -1."""
    if fallback_floor:
        s = str(fallback_floor).lower()
        if "rdc" in s or "rez" in s:
            return 0
        m = re.search(r"\d+", s)
        if m:
            return min(int(m.group()), 30)
    if not text:
        return -1
    if _RE_RDC.search(text):
        return 0
    m = _RE_FLOOR.search(text)
    if m:
        try:
            return min(int(m.group(1)), 30)
        except ValueError:
            pass
    return -1


def _parse_dpe(text: str) -> int:
    """Renvoie 1-7 (G à A) ou 0 si pas trouvé."""
    if not text:
        return 0
    m = _RE_DPE.search(text)
    if not m:
        return 0
    return _DPE_TO_ORDINAL.get(m.group(1).upper(), 0)


def _has_keyword(text: str, regex) -> int:
    return 1 if (text and regex.search(text)) else 0


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, *, training: bool = True) -> pd.DataFrame:
    out = df.copy()

    # Numériques de base
    out["surface"] = out["surface"].astype(float).clip(lower=1)
    out["rooms"] = out["rooms"].fillna(0).astype(int).clip(lower=0, upper=20)
    out["initial_price"] = out["initial_price"].astype(float).clip(lower=1)

    # Dérivées numériques
    out["price_per_sqm_initial"] = out["initial_price"] / out["surface"]
    out["log_initial_price"] = np.log1p(out["initial_price"])
    # rooms peut être 0 (studio/local) → division safe
    out["surface_per_room"] = out["surface"] / out["rooms"].replace(0, 1)

    # Temporel
    if "adjudication_date" in out.columns:
        out["adjudication_date"] = pd.to_datetime(out["adjudication_date"], errors="coerce")
        out["year"] = out["adjudication_date"].dt.year.fillna(2024).astype(int)
        out["month"] = out["adjudication_date"].dt.month.fillna(6).astype(int)
    else:
        from datetime import datetime
        out["year"] = datetime.utcnow().year
        out["month"] = datetime.utcnow().month

    # Code postal normalisé
    if "postal_code" not in out.columns:
        out["postal_code"] = None
    out["postal_code"] = out["postal_code"].apply(_normalize_postal)

    # City normalisée pour features et district_key
    if "city" not in out.columns:
        out["city"] = "Unknown"
    out["city"] = out["city"].fillna("Unknown").astype(str).str.strip()

    # Arrondissement : on tente le CP d'abord (le plus fiable), puis fallback city
    arr_from_pc = out["postal_code"].apply(_extract_arrondissement_from_postal)
    arr_from_city = out["city"].apply(_extract_arrondissement_from_city)
    out["arrondissement"] = arr_from_pc.where(arr_from_pc > 0, arr_from_city).astype(int)

    # District key combinée (city + arrondissement OU city + département)
    out["district_key"] = out.apply(
        lambda r: _build_district_key(r["city"], r["arrondissement"], r["postal_code"]),
        axis=1,
    )

    # Catégorielles restantes
    for col in ("property_type", "region"):
        if col in out.columns:
            out[col] = out[col].fillna("Unknown").astype(str).str.strip()
        else:
            out[col] = "Unknown"

    # Description : extraction features quali
    description = out["description"] if "description" in out.columns else pd.Series([None] * len(out))
    description = description.fillna("").astype(str)

    floor_col = out["floor"] if "floor" in out.columns else pd.Series([None] * len(out))
    out["floor_num"] = [
        _parse_floor(desc, fb) for desc, fb in zip(description, floor_col)
    ]
    out["dpe_ordinal"] = description.apply(_parse_dpe)
    out["has_balcony"] = description.apply(lambda t: _has_keyword(t, _RE_BALCONY))
    out["has_parking"] = description.apply(lambda t: _has_keyword(t, _RE_PARKING))
    out["has_elevator"] = description.apply(lambda t: _has_keyword(t, _RE_ELEVATOR))
    out["is_renovated"] = description.apply(lambda t: _has_keyword(t, _RE_RENOVATED))
    out["is_rented"] = description.apply(lambda t: _has_keyword(t, _RE_RENTED))

    # Géo lat/lng — imputation depuis postal_code (full) ou centroïde dept
    from .location import lookup_latlng

    for col in ("latitude", "longitude"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = np.nan

    needs_impute = out["latitude"].isna() | out["longitude"].isna()
    if needs_impute.any():
        # Si on a un CP complet on l'utilise, sinon clé combinée avec arrondissement
        def _lookup_row(r):
            # Priorité 1 : full CP
            pc = r["postal_code"]
            if len(pc) == 5 and pc.isdigit() and pc[2:] != "000":
                res = lookup_latlng(pc)
                if res:
                    return res
            # Priorité 2 : reconstruire CP depuis city+arrondissement
            arr = r["arrondissement"]
            city_l = r["city"].lower()
            if arr > 0:
                if "paris" in city_l:
                    res = lookup_latlng(f"750{arr:02d}")
                    if res:
                        return res
                elif "lyon" in city_l:
                    res = lookup_latlng(f"690{arr:02d}")
                    if res:
                        return res
                elif "marseille" in city_l:
                    res = lookup_latlng(f"130{arr:02d}")
                    if res:
                        return res
            # Priorité 3 : centroïde département
            return lookup_latlng(pc)

        imputed = out[needs_impute].apply(_lookup_row, axis=1)
        out.loc[needs_impute, "latitude"] = imputed.apply(lambda x: x[0] if x else np.nan)
        out.loc[needs_impute, "longitude"] = imputed.apply(lambda x: x[1] if x else np.nan)

    for col in ("latitude", "longitude"):
        if out[col].isna().all():
            out[col] = 0.0
        else:
            out[col] = out[col].fillna(out[col].median())

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
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[cols]
