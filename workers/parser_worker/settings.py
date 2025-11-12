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

    # MinIO
    MINIO_HOST: str
    MINIO_PORT: int
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_BUCKET_NAME: str
    MINIO_USE_SSL: bool = False

    # Other Service URLs
    UNSTRUCTURED_API_URL: str
    
    # --- UPDATED QUEUES ---
    QUEUE_PARSER: str      # Listen to this
    QUEUE_EMBEDDER: str    # Send to this (Parallel 1)
    QUEUE_EXTRACTOR: str   # Send to this (Parallel 2)
    # ---
    MONITORING_EXCHANGE_NAME: str
    # Database URL
    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def MINIO_ENDPOINT_URL(self) -> str:
        protocol = "https" if self.MINIO_USE_SSL else "http"
        return f"{protocol}://{self.MINIO_HOST}:{self.MINIO_PORT}"

settings = Settings()