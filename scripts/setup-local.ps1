# =============================================================================
# setup-local.ps1
#
# Installation + entraînement + démarrage de ep-ml-api en LOCAL.
# Configuré par défaut pour SQL Server LocalDB + données scrapées Paris.
#
# Usage :
#   .\scripts\setup-local.ps1                      # tout faire et démarrer l'API
#   .\scripts\setup-local.ps1 -SkipInstall         # skip pip install (re-run rapide)
#   .\scripts\setup-local.ps1 -SkipTraining        # skip re-train
#   .\scripts\setup-local.ps1 -Source sql          # force DATA_SOURCE=sql
#   .\scripts\setup-local.ps1 -NoServe             # train mais ne démarre pas l'API
# =============================================================================

param(
    [switch]$SkipInstall = $false,
    [switch]$SkipTraining = $false,
    [switch]$NoServe = $false,
    [ValidateSet("csv","sql","hybrid")]
    [string]$Source = "hybrid",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# --- Aller à la racine du projet (parent de scripts/) ---
Set-Location (Join-Path $PSScriptRoot "..")

function Write-Step($n, $msg) {
    Write-Host ""
    Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "  Étape $n : $msg" -ForegroundColor Cyan
    Write-Host "──────────────────────────────────────────────────────────────" -ForegroundColor Cyan
}

# ── Étape 1 : venv ─────────────────────────────────────────────────────────
Write-Step 1 "Environnement Python"

if (-not (Test-Path ".venv")) {
    Write-Host "  Création du venv..."
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1

if (-not $SkipInstall) {
    Write-Host "  Installation des dépendances..."
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    Write-Host "  ✓ Dépendances installées" -ForegroundColor Green
} else {
    Write-Host "  Skip install (-SkipInstall)"
}

# ── Étape 2 : .env ─────────────────────────────────────────────────────────
Write-Step 2 "Configuration .env"

if (-not (Test-Path ".env")) {
    Copy-Item .env.example .env
    Write-Host "  ✓ .env créé depuis .env.example" -ForegroundColor Green
}

# Force DATA_SOURCE selon paramètre
$envContent = Get-Content .env -Raw
$envContent = $envContent -replace "(?m)^DATA_SOURCE=.*$", "DATA_SOURCE=$Source"
Set-Content -Path .env -Value $envContent -NoNewline
Write-Host "  DATA_SOURCE=$Source"

# ── Étape 3 : check-db ─────────────────────────────────────────────────────
Write-Step 3 "Vérification DB (uniquement si SQL utilisé)"

if ($Source -ne "csv") {
    python -m src.cli check-db
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ check-db a échoué — bascule sur synthétique pour démarrer" -ForegroundColor Yellow
        $envContent = Get-Content .env -Raw
        $envContent = $envContent -replace "(?m)^DATA_SOURCE=.*$", "DATA_SOURCE=csv"
        Set-Content -Path .env -Value $envContent -NoNewline
        $Source = "csv"
    }
} else {
    Write-Host "  Skip (mode csv pur)"
}

# ── Étape 4 : bootstrap synthétique si nécessaire ──────────────────────────
Write-Step 4 "Dataset synthétique (utilisé en mode csv ou hybrid)"

if ($Source -in @("csv", "hybrid") -and -not (Test-Path "data/training.csv")) {
    python -m src.cli bootstrap
    Write-Host "  ✓ data/training.csv généré" -ForegroundColor Green
} else {
    Write-Host "  Skip (déjà présent ou inutile)"
}

# ── Étape 5 : entraînement ─────────────────────────────────────────────────
Write-Step 5 "Entraînement des modèles"

if (-not $SkipTraining) {
    python -m src.cli train --source $Source
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "✗ Entraînement KO — fallback sur csv pur" -ForegroundColor Yellow
        python -m src.cli train --source csv
    }
    Write-Host "  ✓ Modèles entraînés (src/models/store/)" -ForegroundColor Green
} else {
    Write-Host "  Skip (-SkipTraining)"
}

# ── Étape 6 : test rapide de l'inférence ───────────────────────────────────
Write-Step 6 "Test prédiction (smoke test)"

python -c @"
import json
from src.api.predictor import registry
res = registry.predict({
    'tribunal': 'TJ Paris', 'city': 'Paris',
    'region': 'Île-de-France', 'property_type': 'Appartement',
    'surface': 68, 'rooms': 3, 'initial_price': 185000,
})
print('Prediction OK :', json.dumps(res, ensure_ascii=False, indent=2, default=str))
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Smoke test KO" -ForegroundColor Red
    exit 1
}

# ── Étape 7 : démarrage API ─────────────────────────────────────────────────
if (-not $NoServe) {
    Write-Step 7 "Démarrage API"
    Write-Host "  Swagger UI : http://localhost:$Port/docs" -ForegroundColor Green
    Write-Host "  Ctrl+C pour arrêter."
    Write-Host ""
    python -m src.cli serve --port $Port --reload
} else {
    Write-Host ""
    Write-Host "✓ Setup terminé. Pour démarrer l'API : python -m src.cli serve" -ForegroundColor Green
}
