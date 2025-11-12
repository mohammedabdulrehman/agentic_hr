from pydantic_settings import BaseSettings
import os
class Settings(BaseSettings):
    # RabbitMQ
    BROKER_URL: str

    # Postgres
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    # Queues
    QUEUE_EXTRACTOR: str             # Listen to this
    QUEUE_INGESTION_AGGREGATOR: str  # Send to this
    
    # AI Config
    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-4.1-mini"
    MONITORING_EXCHANGE_NAME : str
    # Database URL
    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()