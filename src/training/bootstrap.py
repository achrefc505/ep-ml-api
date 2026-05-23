"""Génère un dataset synthétique réaliste pour bootstrap l'API.

v2 (2026-05) : génère postal_code + latitude/longitude par arrondissement
   pour entraîner les features de localisation fine.
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from ..config import settings


# Profil tribunal → (multiplicateur prix marché de référence, écart-type relatif, région, ville)
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

PROPERTY_PROFILES = {
    "Appartement":          (60,  25,  3, 0.78),
    "Maison":               (110, 40,  4, 0.74),
    "Studio":               (25,  8,   1, 0.82),
    "Local commercial":     (120, 50,  0, 0.68),
    "Terrain":              (500, 300, 0, 0.65),
    "Immeuble de rapport":  (300, 150, 0, 0.70),
}
PROPERTY_WEIGHTS = [0.45, 0.25, 0.10, 0.08, 0.07, 0.05]


# --------------------------------------------------------------------------
# Profil arrondissement : (multiplicateur vs prix moyen ville, lat, lng)
# Calibré sur https://www.meilleursagents.com (ordres de grandeur réels 2024)
# --------------------------------------------------------------------------
PARIS_DISTRICTS = {
    1:  (1.30, 48.8606, 2.3376),   # Louvre — premium
    2:  (1.20, 48.8678, 2.3414),
    3:  (1.25, 48.8627, 2.3601),   # Marais
    4:  (1.30, 48.8546, 2.3576),   # Marais / Ile St-Louis
    5:  (1.30, 48.8447, 2.3477),   # Quartier latin
    6:  (1.45, 48.8488, 2.3327),   # St-Germain — top
    7:  (1.50, 48.8556, 2.3120),   # 7e — top
    8:  (1.40, 48.8721, 2.3120),   # Champs-Élysées
    9:  (1.15, 48.8767, 2.3370),
    10: (0.95, 48.8772, 2.3593),
    11: (1.00, 48.8593, 2.3789),
    12: (0.92, 48.8404, 2.3892),
    13: (0.85, 48.8281, 2.3550),
    14: (0.98, 48.8323, 2.3263),
    15: (1.05, 48.8423, 2.2918),
    16: (1.35, 48.8638, 2.2761),   # 16e — premium
    17: (1.10, 48.8836, 2.3214),
    18: (0.85, 48.8923, 2.3445),   # Montmartre / Goutte d'Or
    19: (0.75, 48.8830, 2.3826),
    20: (0.80, 48.8633, 2.3990),
}

LYON_DISTRICTS = {
    1: (1.05, 45.7691, 4.8345),
    2: (1.15, 45.7549, 4.8312),   # Presqu'île
    3: (1.00, 45.7560, 4.8521),
    4: (1.10, 45.7790, 4.8210),   # Croix-Rousse
    5: (0.95, 45.7616, 4.8198),   # Vieux Lyon
    6: (1.25, 45.7714, 4.8527),   # 6e — top
    7: (0.90, 45.7470, 4.8430),
    8: (0.85, 45.7388, 4.8716),
    9: (0.80, 45.7787, 4.8093),
}

MARSEILLE_DISTRICTS = {
    1:  (0.90, 43.2980, 5.3781),
    2:  (0.75, 43.3074, 5.3640),
    3:  (0.65, 43.3098, 5.3833),
    4:  (0.85, 43.3060, 5.4030),
    5:  (0.80, 43.2920, 5.4030),
    6:  (1.05, 43.2876, 5.3835),
    7:  (1.20, 43.2812, 5.3550),   # Vauban / Roucas Blanc — premium
    8:  (1.25, 43.2606, 5.3683),   # 8e — top (Prado)
    9:  (0.95, 43.2466, 5.4181),
    10: (0.80, 43.2734, 5.4173),
    11: (0.75, 43.2728, 5.4630),
    12: (0.85, 43.2962, 5.4470),
    13: (0.75, 43.3247, 5.4150),
    14: (0.65, 43.3402, 5.3973),
    15: (0.60, 43.3479, 5.3691),
    16: (0.70, 43.3621, 5.3490),
}

# Centroïdes des autres villes (lat, lng approx)
CITY_CENTROIDS = {
    "Versailles":   (48.8049, 2.1204),
    "Nanterre":     (48.8924, 2.2069),
    "Bordeaux":     (44.8378, -0.5792),
    "Nice":         (43.7102, 7.2620),
    "Lille":        (50.6292, 3.0573),
    "Nantes":       (47.2184, -1.5536),
    "Toulouse":     (43.6047, 1.4442),
    "Montpellier":  (43.6109, 3.8772),
    "Strasbourg":   (48.5734, 7.7521),
    "Rennes":       (48.1173, -1.6778),
    "Rouen":        (49.4431, 1.0993),
    "Reims":        (49.2583, 4.0317),
    "Dijon":        (47.3220, 5.0415),
    "Limoges":      (45.8336, 1.2611),
    "Brest":        (48.3904, -4.4861),
    "Angers":       (47.4784, -0.5632),
    "Grenoble":     (45.1885, 5.7245),
}

CITY_POSTAL_BASE = {
    "Versailles": 78000, "Nanterre": 92000, "Bordeaux": 33000,
    "Nice": 6000, "Lille": 59000, "Nantes": 44000, "Toulouse": 31000,
    "Montpellier": 34000, "Strasbourg": 67000, "Rennes": 35000,
    "Rouen": 76000, "Reims": 51100, "Dijon": 21000, "Limoges": 87000,
    "Brest": 29200, "Angers": 49000, "Grenoble": 38000,
}


def _location_for(city: str, rng: np.random.Generator) -> tuple[float, str, float, float, int]:
    """Renvoie (multiplicateur prix, postal_code, lat, lng, arrondissement)."""
    if city == "Paris":
        arr = int(rng.integers(1, 21))
        mult, lat, lng = PARIS_DISTRICTS[arr]
        lat += float(rng.normal(0, 0.003))
        lng += float(rng.normal(0, 0.003))
        return mult, f"750{arr:02d}", lat, lng, arr
    if city == "Lyon":
        arr = int(rng.integers(1, 10))
        mult, lat, lng = LYON_DISTRICTS[arr]
        lat += float(rng.normal(0, 0.003))
        lng += float(rng.normal(0, 0.003))
        return mult, f"690{arr:02d}", lat, lng, arr
    if city == "Marseille":
        arr = int(rng.integers(1, 17))
        mult, lat, lng = MARSEILLE_DISTRICTS[arr]
        lat += float(rng.normal(0, 0.003))
        lng += float(rng.normal(0, 0.003))
        return mult, f"130{arr:02d}", lat, lng, arr

    lat, lng = CITY_CENTROIDS.get(city, (48.8566, 2.3522))
    lat += float(rng.normal(0, 0.005))
    lng += float(rng.normal(0, 0.005))
    pc_base = CITY_POSTAL_BASE.get(city, 75000)
    return 1.0, f"{pc_base:05d}", lat, lng, 0


def _generate_description(rng, property_type, has_balcony, has_parking, has_elevator,
                           is_renovated, is_rented, floor, dpe):
    """Génère une description plausible incluant les mots-clés cibles."""
    parts = [f"{property_type.lower()}"]
    if floor == 0:
        parts.append("au rez-de-chaussée")
    elif floor > 0:
        parts.append(f"au {floor}e étage")
    if has_elevator:
        parts.append("avec ascenseur")
    if has_balcony:
        parts.append("avec balcon")
    if has_parking:
        parts.append(rng.choice(["et parking", "avec garage", "et box"]))
    if is_renovated:
        parts.append(rng.choice(["entièrement rénové", "refait à neuf", "moderne"]))
    if is_rented:
        parts.append(rng.choice(["loué", "occupé par un locataire", "avec bail en cours"]))
    if dpe:
        parts.append(f"DPE {dpe}")
    return " ".join(parts) + "."


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tribunals = list(TRIBUNAL_PROFILES.keys())
    properties = list(PROPERTY_PROFILES.keys())

    today = datetime.utcnow().date()
    horizon_days = 365 * 3

    rows = []
    for _ in range(n_rows):
        tribunal = rng.choice(tribunals)
        base_ppsqm, sigma_rel, region, city = TRIBUNAL_PROFILES[tribunal]

        loc_mult, postal_code, lat, lng, arr = _location_for(city, rng)

        property_type = rng.choice(properties, p=PROPERTY_WEIGHTS)
        surf_mean, surf_std, rooms_med, adj_ratio = PROPERTY_PROFILES[property_type]

        surface = max(8.0, float(rng.normal(surf_mean, surf_std)))
        rooms = max(0, int(rng.normal(rooms_med, 1)))

        # Features quali synthétiques avec un effet sur le prix
        has_balcony = int(rng.random() < 0.3)
        has_parking = int(rng.random() < 0.35)
        has_elevator = int(rng.random() < 0.5)
        is_renovated = int(rng.random() < 0.25)
        is_rented = int(rng.random() < 0.25)
        floor = int(rng.integers(-1, 8))  # -1 = inconnu, 0=RDC, 1-7
        dpe = rng.choice(["A", "B", "C", "D", "E", "F", "G", ""], p=[0.05,0.08,0.15,0.2,0.2,0.15,0.07,0.1])

        # Multiplicateurs (effets typiques marché immobilier français)
        # Sources d'inspiration : études MeilleursAgents 2024
        quality_mult = 1.0
        quality_mult *= (1.06 if has_balcony else 1.0)
        quality_mult *= (1.07 if has_parking else 1.0)
        quality_mult *= (1.05 if has_elevator else 1.0)
        quality_mult *= (1.10 if is_renovated else 1.0)
        quality_mult *= (0.85 if is_rented else 1.0)  # bien occupé = décote ~15%
        # Étage : 0 = RDC -5%, 1-3 neutre, 4+ +3%
        if floor == 0:
            quality_mult *= 0.95
        elif floor >= 4:
            quality_mult *= 1.03
        # DPE
        dpe_mult = {"A": 1.10, "B": 1.07, "C": 1.03, "D": 1.0, "E": 0.97, "F": 0.92, "G": 0.85, "": 1.0}
        quality_mult *= dpe_mult[dpe]

        effective_ppsqm = base_ppsqm * loc_mult * quality_mult
        market_value = surface * effective_ppsqm * float(rng.normal(1.0, sigma_rel))

        adjudicated = market_value * adj_ratio * float(rng.normal(1.0, 0.10))
        adjudicated = max(1000.0, adjudicated)

        start_ratio = float(rng.uniform(0.45, 0.85))
        initial = adjudicated * start_ratio

        adj_date = today - timedelta(days=int(rng.integers(0, horizon_days)))

        description = _generate_description(
            rng, property_type, has_balcony, has_parking, has_elevator,
            is_renovated, is_rented, floor, dpe
        )

        # Simule le cas du user : 30% des lignes ont postal_code tronqué à 2 digits
        # (mais city contient '11e' pour qu'on puisse reconstruire)
        if city in ("Paris", "Lyon", "Marseille") and rng.random() < 0.30:
            short_postal = postal_code[:2]
            display_city = f"{city} {arr}e"
            stored_postal = short_postal
            stored_city = display_city
        else:
            stored_postal = postal_code
            stored_city = city

        rows.append({
            "tribunal": tribunal,
            "city": stored_city,
            "region": region,
            "property_type": property_type,
            "surface": round(surface, 2),
            "rooms": rooms,
            "initial_price": round(initial, 2),
            "adjudicated_price": round(adjudicated, 2),
            "adjudication_date": adj_date,
            "postal_code": stored_postal,
            "latitude": round(lat, 6),
            "longitude": round(lng, 6),
            "description": description,
            "floor": str(floor) if floor >= 0 else None,
        })

    return pd.DataFrame(rows)


def main():
    out = settings.training_csv_full_path
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(n_rows=5000)
    df.to_csv(out, index=False)
    logger.info("✓ Dataset synthétique généré : {} ({} lignes)", out, len(df))
    print(df.head(5).to_string(index=False))
    print("\nPrix moyen au m² par arrondissement parisien (sanity check) :")
    paris = df[df["city"] == "Paris"].copy()
    paris["ppsqm_adj"] = paris["adjudicated_price"] / paris["surface"]
    print(paris.groupby("postal_code")["ppsqm_adj"].mean().round(0).sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
