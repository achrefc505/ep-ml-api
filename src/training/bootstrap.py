"""Génère un dataset synthétique réaliste pour bootstrap l'API.

Permet de tester tout le pipeline (training → API → /predict) sans avoir
scrapé Licitor. Les distributions sont calibrées sur des ordres de grandeur
plausibles du marché immobilier français des ventes judiciaires.

Quand le scraping aura produit assez de données, on bascule DATA_SOURCE=sql
dans .env et on relance `python -m src.training.train`.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import settings


# Profil tribunal → (multiplicateur prix marché, écart-type relatif)
# Calibré sur tendances réelles (Paris > Lyon > Marseille > etc.)
TRIBUNAL_PROFILES = {
    "TJ Paris":        (10500, 0.18, "Île-de-France",         "Paris"),
    "TJ Versailles":   (6800,  0.15, "Île-de-France",         "Versailles"),
    "TJ Nanterre":     (7200,  0.16, "Île-de-France",         "Nanterre"),
    "TJ Lyon":         (4800,  0.14, "Auvergne-Rhône-Alpes",  "Lyon"),
    "TJ Bordeaux":     (4500,  0.15, "Nouvelle-Aquitaine",    "Bordeaux"),
    "TJ Marseille":    (3500,  0.20, "PACA",                  "Marseille"),
    "TJ Nice":         (5200,  0.17, "PACA",                  "Nice"),
    "TJ Lille":        (3200,  0.14, "Hauts-de-France",       "Lille"),
    "TJ Nantes":       (3800,  0.13, "Pays de la Loire",      "Nantes"),
    "TJ Toulouse":     (3600,  0.13, "Occitanie",             "Toulouse"),
    "TJ Montpellier":  (3400,  0.14, "Occitanie",             "Montpellier"),
    "TJ Strasbourg":   (3000,  0.13, "Grand Est",             "Strasbourg"),
    "TJ Rennes":       (3200,  0.12, "Bretagne",              "Rennes"),
    "TJ Rouen":        (2400,  0.13, "Normandie",             "Rouen"),
    "TJ Reims":        (2300,  0.12, "Grand Est",             "Reims"),
    "TJ Dijon":        (2100,  0.12, "Bourgogne-FC",          "Dijon"),
    "TJ Limoges":      (1500,  0.13, "Nouvelle-Aquitaine",    "Limoges"),
    "TJ Brest":        (2200,  0.13, "Bretagne",              "Brest"),
    "TJ Angers":       (2400,  0.12, "Pays de la Loire",      "Angers"),
    "TJ Grenoble":     (3200,  0.13, "Auvergne-Rhône-Alpes",  "Grenoble"),
}

# Profil type de bien → (surface_moy, ecart, rooms_moy, ratio_adj/marche)
PROPERTY_PROFILES = {
    "Appartement":          (60,  25,  3, 0.78),   # adjudication ~78% du marché
    "Maison":               (110, 40,  4, 0.74),
    "Studio":               (25,  8,   1, 0.82),
    "Local commercial":     (120, 50,  0, 0.68),
    "Terrain":              (500, 300, 0, 0.65),
    "Immeuble de rapport":  (300, 150, 0, 0.70),
}

PROPERTY_WEIGHTS = [0.45, 0.25, 0.10, 0.08, 0.07, 0.05]


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tribunals = list(TRIBUNAL_PROFILES.keys())
    properties = list(PROPERTY_PROFILES.keys())

    today = datetime.utcnow().date()
    horizon_days = 365 * 3  # 3 ans d'historique synthétique

    rows = []
    for _ in range(n_rows):
        tribunal = rng.choice(tribunals)
        price_per_sqm_market, sigma_rel, region, city = TRIBUNAL_PROFILES[tribunal]

        property_type = rng.choice(properties, p=PROPERTY_WEIGHTS)
        surf_mean, surf_std, rooms_med, adj_ratio = PROPERTY_PROFILES[property_type]

        surface = max(8.0, float(rng.normal(surf_mean, surf_std)))
        rooms = max(0, int(rng.normal(rooms_med, 1)))

        # Prix marché estimé pour ce bien
        market_value = surface * price_per_sqm_market * float(rng.normal(1.0, sigma_rel))

        # Prix d'adjudication ~= market * adj_ratio + bruit
        adjudicated = market_value * adj_ratio * float(rng.normal(1.0, 0.10))
        adjudicated = max(1000.0, adjudicated)

        # Mise à prix = adjudicated * ratio (le marteau monte du startPrice à l'adjudication)
        start_ratio = float(rng.uniform(0.45, 0.85))
        initial = adjudicated * start_ratio

        # Date dans les 3 dernières années
        days_back = int(rng.integers(0, horizon_days))
        adj_date = today - timedelta(days=days_back)

        rows.append({
            "tribunal": tribunal,
            "city": city,
            "region": region,
            "property_type": property_type,
            "surface": round(surface, 2),
            "rooms": rooms,
            "initial_price": round(initial, 2),
            "adjudicated_price": round(adjudicated, 2),
            "adjudication_date": adj_date,
        })

    df = pd.DataFrame(rows)
    return df


def main():
    out = settings.training_csv_full_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_rows=5000)
    df.to_csv(out, index=False)
    logger.info("✓ Dataset synthétique généré : {} ({} lignes)", out, len(df))
    logger.info("Aperçu :")
    print(df.head(5).to_string(index=False))
    print("\nDistribution par tribunal :")
    print(df["tribunal"].value_counts().to_string())


if __name__ == "__main__":
    main()
