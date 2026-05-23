# Scripts de démarrage local

## 🚀 Une seule commande pour tout faire

```powershell
.\scripts\setup-local.ps1
```

Le script :

1. Crée le `.venv` Python si absent
2. Installe `requirements.txt`
3. Copie `.env.example` → `.env` (si absent), force `DATA_SOURCE=hybrid`
4. Lance `check-db` pour vérifier la connectivité SQL Server
5. Génère `data/training.csv` synthétique (si absent)
6. Entraîne tous les modèles (RandomForest global + un par tribunal)
7. Fait un smoke-test d'inférence (`TJ Paris 68m² T3 → prédiction`)
8. Démarre l'API sur http://localhost:8000

## Variantes

| Commande | Effet |
|----------|-------|
| `.\scripts\setup-local.ps1` | Tout (default `hybrid`) |
| `.\scripts\setup-local.ps1 -Source sql` | Force `DATA_SOURCE=sql` |
| `.\scripts\setup-local.ps1 -Source csv` | Force synthétique pur |
| `.\scripts\setup-local.ps1 -SkipInstall` | Re-run rapide (skip pip) |
| `.\scripts\setup-local.ps1 -SkipTraining` | Skip re-train (juste démarrer l'API) |
| `.\scripts\setup-local.ps1 -NoServe` | Train mais ne démarre pas l'API |
| `.\scripts\setup-local.ps1 -Port 8001` | Port custom |

## Comportement défensif

| Si... | Alors... |
|-------|---------|
| Pas de connexion SQL | `check-db` échoue → bascule auto sur `DATA_SOURCE=csv` |
| `< 200` adjudications scrapées | Mode `hybrid` → SQL + synthétique en backfill |
| `>= 200` adjudications | Mode `hybrid` → SQL pur (synthétique ignoré) |
| Pas assez de tribunaux à seuil | Modèle global seulement, fallback automatique à l'inférence |

## Commandes manuelles utiles

```powershell
# Diagnostic DB
python -m src.cli check-db

# Régénérer le synthétique
python -m src.cli bootstrap

# Re-train (sans toucher au .env)
python -m src.cli train --source hybrid

# Démarrer l'API seule
python -m src.cli serve --port 8000
```

## Premier setup typique pour Paris uniquement

Si tu n'as scrapé QUE Paris et que tu veux que le modèle apprenne ce que tu as déjà :

```powershell
# 1. Lancer en mode hybride (SQL + synthétique en backfill)
.\scripts\setup-local.ps1 -Source hybrid
```

Le modèle global sera entraîné sur **tes données Paris + synthétique des autres villes**, et un modèle dédié `TJ Paris` apparaîtra dès que tu as ≥ 20 adjudications réelles.

Plus tard quand tu auras scrapé tout :

```powershell
.\scripts\setup-local.ps1 -Source sql -SkipInstall
```
