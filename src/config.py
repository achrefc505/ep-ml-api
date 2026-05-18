from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_source: str = "csv"  # 'csv' | 'sql'
    training_csv_path: str = "data/training.csv"

    db_server: str = "(localdb)\\mssqllocaldb"
    db_name: str = "EncheresPredict_Raw"
    db_trusted: str = "yes"
    db_user: str | None = None
    db_password: str | None = None

    models_dir: str = "src/models/store"
    min_samples_per_tribunal: int = 30

    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 3
    rf_random_state: int = 42

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    @property
    def models_path(self) -> Path:
        p = ROOT_DIR / self.models_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def training_csv_full_path(self) -> Path:
        return ROOT_DIR / self.training_csv_path

    @property
    def odbc_connection_string(self) -> str:
        if self.db_trusted.lower() in {"yes", "true", "1"}:
            return (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={self.db_server};DATABASE={self.db_name};"
                f"Trusted_Connection=yes;TrustServerCertificate=yes;"
            )
        return (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={self.db_server};DATABASE={self.db_name};"
            f"UID={self.db_user};PWD={self.db_password};TrustServerCertificate=yes;"
        )

    @property
    def sqlalchemy_url(self) -> str:
        from urllib.parse import quote_plus
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(self.odbc_connection_string)}"


settings = Settings()
