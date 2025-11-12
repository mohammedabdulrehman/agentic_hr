from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RABBITMQ_USER : str
    RABBITMQ_PASS : str
    RABBITMQ_HOST : str
    RABBITMQ_PORT : int
    
    REDIS_HOST : str
    REDIS_PORT : int
    REDIS_DB : int

    MONITORING_EXCHANGE_NAME : str
    MONITORING_QUEUE_NAME : str
    MONITORING_ROUTING_KEY : str

settings = Settings()