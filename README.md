# ep-ml-api

API Python de **prédiction du prix d'adjudication immobilier** pour EnchèresPredict.
Random Forest avec **un modèle par tribunal** + fallback global automatique.

## Stack
- FastAPI + Pydantic v2
- scikit-learn (RandomForestRegressor)
- pandas / numpy
- SQLAlchemy + pyodbc (lecture de `EncheresPredict_Raw`)
- joblib (sérialisation des modèles)

## Architecture

```
ep-ml-api/
├── src/
│   ├── config.py                  # settings via .env
│   ├── data/
│   │   └── loader.py              # CSV ou SQL Server (au choix)
│   ├── features/
│   │   └── build.py               # feature engineering partagé train/inference
│   ├── training/
│   │   ├── bootstrap.py           # générateur dataset synthétique (5000 lignes)
│   │   └── train.py               # entraînement RF par tribunal + global
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   ├── predictor.py           # cache modèles + logique prédiction
│   │   └── schemas.py             # Pydantic request/response
│   └── models/store/              # .joblib + index.json (généré)
├── tests/
└── data/training.csv              # (généré) dataset bootstrap
```

## Installation (Windows / Linux / Mac)

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

## 🚀 Démarrage automatique en local (Windows)

**Une seule commande** depuis le repo cloné :

```powershell
.\scripts\setup-local.ps1
```

Le script crée le `.venv`, installe les deps, configure `.env`, vérifie SQL Server,
entraîne les modèles et démarre l'API. Voir [`scripts/README.md`](scripts/README.md)
pour les options.

## Démarrage manuel (en 3 commandes)

```bash
# 1. Génère 5000 lignes de données synthétiques réalistes (calibrées par tribunal)
python -m src.training.bootstrap

# 2. Entraîne le modèle global + 1 modèle par tribunal éligible
python -m src.training.train

# 3. Lance l'API (Swagger UI auto sur /docs)
uvicorn src.api.main:app --reload
```

Ouvrir http://localhost:8000/docs

## Exemple de prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tribunal": "TJ Paris",
    "city": "Paris",
    "region": "Île-de-France",
    "property_type": "Appartement",
    "surface": 68,
    "rooms": 3,
    "initial_price": 185000,
    "adjudication_date": "2026-06-15"
  }'
```

Réponse :
```json
{
  "adjudicated_price_predicted": 376365.65,
  "low_estimate": 321288.46,
  "high_estimate": 447147.10,
  "confidence": 80,
  "model_used": "tribunal",
  "model_name": "TJ Paris",
  "features_version": "v1",
  "model_metrics": { "mae": 97558.89, "r2": 0.93, "mape": 14.47 }
}
```

## Endpoints

| Méthode | Path | Description |
|---------|------|-------------|
| GET  | `/health`     | Statut + nombre de modèles chargés |
| GET  | `/models`     | Métriques de tous les modèles entraînés |
| GET  | `/tribunals`  | Liste des tribunaux ayant un modèle dédié |
| POST | `/predict`    | Prédiction (voir schéma) |
| GET  | `/docs`       | Swagger UI |

## Brancher sur la vraie base scrapée

Une fois ton `ep-licitor-scraper` a alimenté `EncheresPredict_Raw.adjudications` :

```bash
# .env
DATA_SOURCE=sql
DB_SERVER=(localdb)\mssqllocaldb
DB_NAME=EncheresPredict_Raw
DB_TRUSTED=yes
```

Puis ré-entraîne :
```bash
python -m src.training.train
```

Les anciens modèles dans `src/models/store/` sont **écrasés** automatiquement.

## Performances sur le dataset synthétique (5000 lignes, seed=42)

| Modèle | Train | Test | MAE | RMSE | R² | MAPE |
|--------|------:|-----:|----:|-----:|----:|-----:|
| **Global** | 4000 | 1000 | 48 195 € | 121 683 € | 0.944 | 14.0% |
| TJ Paris | 196 | 49 | 97 559 € | 130 922 € | 0.932 | 14.5% |
| TJ Lyon | 189 | 48 | 46 620 € | 73 403 € | 0.961 | 15.3% |
| TJ Marseille | 200 | 50 | 41 458 € | 86 364 € | 0.953 | 15.7% |
| TJ Bordeaux | 218 | 55 | 44 639 € | 66 430 € | 0.971 | 11.9% |
| ... | | | | | | |

> Les MAPE 10-16% sur synthétique sont attendus ; sur vraies données Licitor, on devrait converger vers le même ordre voire mieux grâce à de meilleures features (géocodage, prix DVF, etc.)

## Configuration

Toutes les variables dans `.env` :

```env
DATA_SOURCE=csv                    # 'csv' ou 'sql'
TRAINING_CSV_PATH=data/training.csv
MIN_SAMPLES_PER_TRIBUNAL=30        # seuil pour modèle dédié
RF_N_ESTIMATORS=200
RF_MAX_DEPTH=20
RF_MIN_SAMPLES_LEAF=3
API_HOST=0.0.0.0
API_PORT=8000
```

## Intégration avec le backend .NET

Côté EncheresPredict.Api, créer un service `MlApiClient` qui appelle `/predict`.
Lors du seed/import d'une nouvelle enchère depuis Raw → app, calculer `aiEstimate`
en appelant cette API au lieu de le hardcoder.

## Tests

```bash
pytest tests/
```

## TODO

- [ ] Ajouter `geocoded` (lat/lng) comme feature après géocodage
- [ ] Ajouter `dvf_price_per_sqm` comme feature (prix marché DVF)
- [ ] Endpoint `/feedback` pour collecter les vraies adjudications post-vente → retraining
- [ ] Versioning des modèles (MLflow ou stockage S3)
- [ ] Dockerfile + healthcheck
- [ ] Métriques Prometheus

## Licence

MIT.
