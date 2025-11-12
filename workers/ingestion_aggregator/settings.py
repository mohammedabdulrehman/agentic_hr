from pydantic_settings import BaseSettings

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
    QUEUE_INGESTION_AGGREGATOR: str # Listen to this
    
    # Database URL
    @property
    def DATABASE_URL_PSYCOPG(self) -> str:
        return f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

settings = Settings()