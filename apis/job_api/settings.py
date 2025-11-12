from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    # API Config
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    # --- ADD THESE LINES ---
    REDIS_HOST : str
    REDIS_PORT : int
    REDIS_DB  : str
    # -----------------------
    # Postgres Config
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    
    # --- NEW: RabbitMQ & Queue Config ---
    BROKER_URL: str
    QUEUE_JOB_EMBEDDER: str

    MONITORING_EXCHANGE_NAME: str
    # Database URL for SQLAlchemy + psycopg
    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()