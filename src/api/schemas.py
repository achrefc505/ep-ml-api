"""Schémas Pydantic pour les endpoints API."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """Entrée d'une prédiction de prix d'adjudication."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tribunal": "TJ Paris",
            "city": "Paris",
            "region": "Île-de-France",
            "property_type": "Appartement",
            "surface": 68,
            "rooms": 3,
            "initial_price": 185000,
            "adjudication_date": "2026-06-15",
            "postal_code": "75011",
            "latitude": 48.8593,
            "longitude": 2.3789,
            "address": "14 rue de la Roquette, 75011 Paris"
        }
    })

    tribunal: str = Field(..., description="Tribunal Judiciaire (ex: 'TJ Paris')")
    city: str = Field(..., description="Ville du bien")
    region: Optional[str] = Field(None, description="Région administrative")
    property_type: str = Field(..., description="Type de bien (Appartement, Maison, ...)")
    surface: float = Field(..., gt=0, description="Surface en m²")
    rooms: int = Field(0, ge=0, le=20, description="Nombre de pièces")
    initial_price: float = Field(..., gt=0, description="Mise à prix (€)")
    adjudication_date: Optional[date] = Field(None, description="Date prévue de la vente")

    # Localisation fine (v2) — CRITIQUE pour Paris/Lyon/Marseille
    postal_code: Optional[str] = Field(
        None, description="Code postal (ex: '75011'). Discrimine fortement les prix intra-ville."
    )
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    address: Optional[str] = Field(
        None, description="Adresse complète (informatif, peut servir au géocodage côté serveur si lat/lng absents)"
    )


class PredictResponse(BaseModel):
    adjudicated_price_predicted: float
    low_estimate: float
    high_estimate: float
    confidence: int = Field(..., description="Score de confiance 0-100")
    model_used: str = Field(..., description="'tribunal' ou 'global'")
    model_name: str
    features_version: str
    model_metrics: dict


class ModelInfo(BaseModel):
    name: str
    slug: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float
    mape: float


class ModelsResponse(BaseModel):
    trained_at: Optional[str] = None
    total_rows: Optional[int] = None
    features_version: Optional[str] = None
    models: list[ModelInfo] = []


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    tribunals_count: int
