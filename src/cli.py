"""CLI principal de ep-ml-api : check-db, train, serve."""
import sys

import click
from loguru import logger

from .config import settings


def _setup_logging():
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)


def _fix_console_encoding():
    """Force UTF-8 on Windows consoles that default to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


@click.group()
def cli():
    """ep-ml-api — prédiction prix d'adjudication."""
    _fix_console_encoding()
    _setup_logging()


@cli.command(name="check-db")
def check_db():
    """Diagnostique la connectivité SQL Server + qualité des données scrapées."""
    click.echo(f"Server   : {settings.db_server}")
    click.echo(f"Database : {settings.db_name}")
    click.echo("")

    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(settings.sqlalchemy_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1")).first()
        click.secho("✓ Connexion SQL Server OK", fg="green")
    except Exception as e:
        click.secho(f"✗ Connexion KO : {e}", fg="red")
        click.echo("\nVérifications :")
        click.echo("  1. SQL Server LocalDB démarré ? → sqllocaldb start mssqllocaldb")
        click.echo("  2. ODBC Driver 17 installé ? → https://learn.microsoft.com/sql/connect/odbc/")
        click.echo("  3. Base EncheresPredict_Raw existe ?")
        sys.exit(1)

    try:
        from .data.loader import db_quality_report
        r = db_quality_report()
    except Exception as e:
        click.secho(f"✗ Lecture table adjudications KO : {e}", fg="red")
        click.echo("  La table dbo.adjudications existe-t-elle ?")
        sys.exit(1)

    click.echo("")
    click.echo("─── Qualité des données ──────────────────────────")
    click.echo(f"  Total adjudications        : {r['total']:>6}")
    click.echo(f"  avec adjudicated_price     : {r['with_adjudicated_price']:>6}")
    click.echo(f"  avec initial_price         : {r['with_initial_price']:>6}")
    click.echo(f"  avec surface > 5           : {r['with_surface']:>6}")
    click.echo(f"  avec tribunal              : {r['with_tribunal']:>6}")
    click.echo(f"  avec city                  : {r['with_city']:>6}")
    click.echo(f"  → ML-usable (toutes conds) : {r['ml_usable']:>6}")
    click.echo("")
    click.echo("─── Répartition par tribunal ─────────────────────")
    if not r["per_tribunal"]:
        click.echo("  (vide)")
    else:
        for tribunal, n in r["per_tribunal"]:
            marker = "✓" if n >= settings.min_samples_per_tribunal else " "
            click.echo(f"  {marker} {tribunal:<30} {n:>5}")
        click.echo(f"\n  ✓ = ≥ {settings.min_samples_per_tribunal} (modèle dédié)")

    # Recommandation
    click.echo("")
    click.echo("─── Recommandation ───────────────────────────────")
    if r["ml_usable"] < 50:
        click.secho(
            "⚠ Pas assez de données scrapées (<50). Recommandation : DATA_SOURCE=csv pour démarrer.",
            fg="yellow",
        )
    elif r["ml_usable"] < 200:
        click.secho(
            f"  {r['ml_usable']} lignes — mode hybride recommandé : DATA_SOURCE=hybrid",
            fg="cyan",
        )
    else:
        click.secho(
            f"✓ {r['ml_usable']} lignes — DATA_SOURCE=sql (modèles entraînés sur tes vraies données)",
            fg="green",
        )


@cli.command()
@click.option("--source", type=click.Choice(["csv", "sql", "hybrid"]), default=None,
              help="Surcharge DATA_SOURCE de .env le temps du run")
def bootstrap(source):
    """Génère un dataset synthétique (5000 lignes) calibré sur 20 tribunaux."""
    if source:
        import os
        os.environ["DATA_SOURCE"] = source
    from .training.bootstrap import main as bootstrap_main
    bootstrap_main()


@cli.command()
@click.option("--source", type=click.Choice(["csv", "sql", "hybrid"]), default=None,
              help="Surcharge DATA_SOURCE de .env le temps du run")
@click.option("--min-samples", type=int, default=None,
              help="Surcharge MIN_SAMPLES_PER_TRIBUNAL")
def train(source, min_samples):
    """Entraîne le modèle global + 1 modèle par tribunal éligible."""
    if source:
        import os
        os.environ["DATA_SOURCE"] = source
        # Recharger les settings après mutation env
        from importlib import reload
        from . import config as _config_mod
        reload(_config_mod)

    if min_samples:
        settings.min_samples_per_tribunal = min_samples

    from .training.train import train_all
    train_all()


@cli.command()
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--reload", "reload_", is_flag=True)
def serve(host, port, reload_):
    """Démarre l'API FastAPI."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload_,
    )


if __name__ == "__main__":
    cli()
