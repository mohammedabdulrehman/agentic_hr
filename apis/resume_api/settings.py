from pydantic_settings import BaseSettings
import os
class Settings(BaseSettings):
    API_PORT: int = 8001 # We'll map this to 8001

    # Postgres
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    # --- ADD THESE LINES ---
    REDIS_HOST : str
    REDIS_PORT : int
    REDIS_DB  : str
    # RabbitMQ
    BROKER_URL: str

    # MinIO
    MINIO_HOST: str
    MINIO_PORT: int
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str
    MINIO_BUCKET_NAME: str
    MINIO_USE_SSL: bool = False

    # The *first* queue in the ingestion pipeline
    QUEUE_PARSER: str
    MONITORING_EXCHANGE_NAME: str
    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def MINIO_ENDPOINT_URL(self) -> str:
        return f"http://{self.MINIO_HOST}:{self.MINIO_PORT}"

settings = Settings()