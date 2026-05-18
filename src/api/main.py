"""Point d'entrée FastAPI."""
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ..config import settings
from .predictor import registry
from .schemas import (
    PredictRequest, PredictResponse,
    ModelsResponse, HealthResponse,
)


# ---------------------------------------------------------------------------
# Logging global
# ---------------------------------------------------------------------------
logger.remove()
logger.add(sys.stderr, level=settings.log_level)

app = FastAPI(
    title="EncheresPredict — ML API",
    description="Prédiction du prix d'adjudication immobilier par Random Forest (modèle par tribunal + fallback global).",
    version="1.0.0",
)

# CORS — autorise le frontend Angular (et le backend .NET si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://localhost:5000",
        "https://localhost:7001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        tribunals = registry.list_tribunals()
        return HealthResponse(status="ok", models_loaded=True, tribunals_count=len(tribunals))
    except Exception as e:
        logger.warning("Health : modèles non chargés : {}", e)
        return HealthResponse(status="ok", models_loaded=False, tribunals_count=0)


@app.get("/models", response_model=ModelsResponse)
def models():
    try:
        return registry.get_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/tribunals", response_model=list[str])
def tribunals():
    """Liste des tribunaux pour lesquels un modèle dédié existe."""
    try:
        return registry.list_tribunals()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        payload = req.model_dump(mode="python")
        result = registry.predict(payload)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Erreur prédiction")
        raise HTTPException(status_code=400, detail=str(e))
