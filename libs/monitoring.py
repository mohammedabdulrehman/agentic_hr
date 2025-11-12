# /libs/monitoring.py
import json
from kombu import Exchange, Producer
from settings import settings # <-- This will import from the *local* settings.py

# Define the exchange (it will get the name from the local settings)
monitoring_exchange = Exchange(
    settings.MONITORING_EXCHANGE_NAME, 
    type='topic', 
    durable=True
)

def publish_event(celery_app_instance, routing_key: str, data: dict):
    """
    Publishes a monitoring event using the app's existing connection pool.
    """
    try:
        print(f"[Monitor Publish] Sending to exchange={settings.MONITORING_EXCHANGE_NAME}, routing_key={routing_key}")

        with celery_app_instance.producer_pool.acquire(block=True) as producer:
            producer.publish(
                body=json.dumps(data, default=str), # Use default=str for safety
                routing_key=routing_key,
                exchange=monitoring_exchange,
                declare=[monitoring_exchange],
                retry=True,
                retry_policy={'max_retries': 3}
            )
        print(f"[Monitor Publish] Sent event: {routing_key}")
    except Exception as e:
        print(f"CRITICAL: Failed to publish monitoring event {routing_key}. Error: {e}")