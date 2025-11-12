from pydantic_settings import BaseSettings
from typing import Optional # <-- Import this

class Settings(BaseSettings):
    API_PORT: int = 8002

    # Postgres
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # RabbitMQ
    BROKER_URL: str

    # The *first* queue in the scoring pipeline
    QUEUE_SCORING_ENGINE: str

    # --- ADD THESE REDIS SETTINGS ---
    # These are needed to read from the dashboard cache
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # --- ADD THIS MONITORING SETTING ---
    # This is needed to publish events
    MONITORING_EXCHANGE_NAME: str
    # -----------------------------------

    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()