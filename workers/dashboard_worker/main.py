import asyncio
import aio_pika
import json
import redis.asyncio as redis
from settings import settings
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
# --- Redis Client Initialization ---
redis_client: redis.Redis | None = None


async def get_redis_client() -> redis.Redis:
    global redis_client
    if redis_client is None:
        print(f"[Aggregator] Connecting to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}...")
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
    return redis_client


# --- RabbitMQ Callback ---
async def process_monitoring_message(message: aio_pika.IncomingMessage) -> None:
    async with message.process(requeue=False):
        try:
            body = message.body
            routing_key = message.routing_key
            message_data = json.loads(body.decode())

            print(f"[Aggregator] Received event: {routing_key}")

            # --- Determine Redis Key ---
            batch_id = message_data.get("batch_id")
            job_id = message_data.get("job_id")

            if batch_id:
                key = f"batch:{batch_id}"
            elif job_id:
                key = f"job:{job_id}"
            else:
                print(f"[Aggregator] Skipping event (no batch_id/job_id): {routing_key}")
                return

            r = await get_redis_client()
            pipe = r.pipeline()

            # === Job Ingestion Events ===
            if routing_key == "job.created":
                pipe.hset(key, mapping={
                    "status": "CREATED",
                    "title": message_data.get("title", "N/A"),
                    "type": "job_ingestion"
                })
            elif routing_key == "job.embedding.started":
                pipe.hset(key, "status", "EMBEDDING")
            elif routing_key == "job.embedding.completed":
                pipe.hset(key, mapping={
                    "status": "EMBEDDING_COMPLETE",
                    "vectors_created": message_data.get("vectors_created", 0)
                })
            elif routing_key == "job.embedding.failed":
                pipe.hset(key, mapping={
                    "status": "EMBEDDING_FAILED",
                    "error": message_data.get("error", "Unknown error")
                })

            # === Resume Ingestion Pipeline Events ===
            elif routing_key == "resume.batch.created":
                pipe.hset(key, mapping={
                    "total_resumes": message_data.get("total_resumes", 0),
                    "status": "SUBMITTED",
                    "type": "resume_ingestion"
                })
            elif routing_key == "resume.parsing.started":
                pipe.hincrby(key, "parsing_started", 1)
            elif routing_key == "resume.parsing.completed":
                pipe.hincrby(key, "parsing_completed", 1)

            # === Matching Pipeline Events ===
            elif routing_key == "match.batch.created":
                pipe.hset(key, mapping={
                    "total_resumes": message_data.get("total_resumes", 0),
                    "job_id": message_data.get("job_id"),
                    "status": "SUBMITTED",
                    "type": "matching"
                })
            elif routing_key == "score.llm.completed":
                pipe.hincrby(key, "llm_completed", 1)
            elif routing_key == "score.aggregator.completed":
                pipe.hincrby(key, "total_complete", 1)

            # Expire the key after 7 days
            pipe.expire(key, 60 * 60 * 24 * 7)
            await pipe.execute()

        except Exception as e:
            print(f"[Aggregator] Error processing message: {e}")
            await message.reject(requeue=False)


# --- Main Async Execution ---
async def main() -> None:
    print("[Aggregator] Service starting...")
    await get_redis_client()  # Initialize Redis connection

    while True:
        connection = None
        try:
            mq_url = (
                f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASS}"
                f"@{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}/"
            )
            print(f"[Aggregator] Connecting to RabbitMQ at {settings.RABBITMQ_HOST}...")
            connection = await aio_pika.connect_robust(mq_url, timeout=10)
            print("[Aggregator] Connected to RabbitMQ.")

            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=100)

                monitoring_exchange = await channel.declare_exchange(
                    settings.MONITORING_EXCHANGE_NAME,
                    aio_pika.ExchangeType.TOPIC,
                    durable=True,
                )

                queue = await channel.declare_queue(
                    settings.MONITORING_QUEUE_NAME, durable=True
                )

                await queue.bind(
                    exchange=monitoring_exchange,
                    routing_key=settings.MONITORING_ROUTING_KEY,
                )

                print(
                    f"[Aggregator] Bound queue '{settings.MONITORING_QUEUE_NAME}' "
                    f"to exchange '{settings.MONITORING_EXCHANGE_NAME}'"
                )
                print("[Aggregator] Waiting for monitoring events...")
                await queue.consume(process_monitoring_message)
                await asyncio.Future()  # keep running forever

        except Exception as e:
            print(f"[Aggregator] Error: {e}. Retrying in 5 seconds...")
            if connection:
                await connection.close()
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Aggregator] Main process terminated.")
