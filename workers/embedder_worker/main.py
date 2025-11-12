# /workers/embedder_worker/main.py

import asyncio
import threading
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from qdrant_client import AsyncQdrantClient, models
from sentence_transformers import SentenceTransformer

from libs.monitoring import publish_event
from schemas.models import Resume, ScoreLog
from settings import settings

# --------------------------------------------------------------------------
# Celery Configuration
# --------------------------------------------------------------------------

app = Celery("embedder_worker", broker=settings.BROKER_URL, backend="rpc://")

app.conf.task_default_queue = settings.QUEUE_EMBEDDER
app.conf.task_queues = {
    settings.QUEUE_EMBEDDER: {
        "exchange": settings.QUEUE_EMBEDDER,
        "routing_key": settings.QUEUE_EMBEDDER,
    }
}

# --------------------------------------------------------------------------
# Async Resource Management
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    qdrant_client = None
    embedding_model = None

resources: AsyncResources = None
loop_thread = None
loop = None


def start_loop_in_thread():
    """Run a dedicated event loop forever in a background thread."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_forever()


@worker_process_init.connect
def on_worker_init(**kwargs):
    """Initialize async resources once when the Celery worker starts."""
    global resources, loop_thread, loop
    resources = AsyncResources()

    print("[EmbedderWorker] Loading embedding model...")
    resources.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    print("[EmbedderWorker] Embedding model loaded.")

    # Start background async loop
    loop_thread = threading.Thread(target=start_loop_in_thread, daemon=True)
    loop_thread.start()

    # Wait until the loop is ready
    while loop is None or not loop.is_running():
        asyncio.sleep(0.1)

    # Initialize async clients on that loop
    fut = asyncio.run_coroutine_threadsafe(setup_async_resources(resources), loop)
    fut.result()
    print("[EmbedderWorker] Async loop and resources initialized.")


async def setup_async_resources(res: AsyncResources):
    """Async initialization for DB and Qdrant."""
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    res.qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    # Ensure Qdrant collection exists
    try:
        exists = await res.qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME)
        if not exists:
            print(f"[EmbedderWorker] Creating Qdrant collection: {settings.QDRANT_COLLECTION_NAME}")
            await res.qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=res.embedding_model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
        print("[EmbedderWorker] Qdrant collection ready.")
    except Exception as e:
        print(f"[EmbedderWorker] Qdrant setup error: {e}")


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Clean shutdown for async resources and loop."""
    global loop, loop_thread
    if not resources or not loop:
        return

    print("[EmbedderWorker] Shutting down async resources...")
    fut = asyncio.run_coroutine_threadsafe(shutdown_async_resources(resources), loop)
    fut.result()

    loop.call_soon_threadsafe(loop.stop)
    if loop_thread and loop_thread.is_alive():
        loop_thread.join(timeout=5)
    print("[EmbedderWorker] Shutdown complete.")


async def shutdown_async_resources(res: AsyncResources):
    """Async cleanup."""
    try:
        if res.qdrant_client:
            await res.qdrant_client.close()
        if res.db_engine:
            await res.db_engine.dispose()
    except Exception as e:
        print(f"[EmbedderWorker] Error during shutdown: {e}")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async DB session."""
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized.")
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# Embedding Logic
# --------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    tokens = text.split()
    return [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size - overlap)]


async def async_embed_resume(resume_id: int):
    """Main async embedding pipeline."""
    print(f"[Resume {resume_id}] Starting embedding...")

    batch_id = None
    try:
        # --- 1. Fetch resume ---
        async with get_db_session() as db:
            resume = await db.get(Resume, resume_id)
            if not resume or not resume.parsed_text:
                raise Exception(f"Resume {resume_id} missing parsed_text.")

            batch_id = str(resume.batch_id) if resume.batch_id else None
            resume.status = "EMBEDDING"
            db.add(ScoreLog(
                resume_id=resume_id,
                batch_id=batch_id,
                worker_name="embedder_worker",
                status="STARTED",
                message="Embedding started."
            ))
            await db.commit()

        publish_event(app, "resume.embedding.started", {"resume_id": resume_id, "batch_id": batch_id})

        # --- 2. Chunk + embed ---
        text_chunks = chunk_text(resume.parsed_text)
        if not text_chunks:
            raise Exception("No text chunks created.")

        print(f"[Resume {resume_id}] Embedding {len(text_chunks)} chunks...")
        vectors = await asyncio.to_thread(resources.embedding_model.encode, text_chunks)

        # --- 3. Upsert to Qdrant ---
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"resume_id": resume_id, "text_chunk": chunk},
            )
            for chunk, vector in zip(text_chunks, vectors)
        ]

        await resources.qdrant_client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=points,
            wait=True,
        )
        print(f"[Resume {resume_id}] Saved {len(points)} vectors to Qdrant.")

        # --- 4. Mark as completed ---
        async with get_db_session() as db:
            resume = await db.get(Resume, resume_id)
            if resume:
                resume.is_embedded = True
                resume.status = "EMBEDDING_COMPLETE"
                db.add(ScoreLog(
                    resume_id=resume_id,
                    batch_id=batch_id,
                    worker_name="embedder_worker",
                    status="COMPLETED",
                    message=f"Embedding complete. {len(points)} vectors created."
                ))
                await db.commit()

        publish_event(app, "resume.embedding.completed", {
            "resume_id": resume_id,
            "batch_id": batch_id,
            "vectors_created": len(points)
        })

        # --- 5. Send next task ---
        await asyncio.to_thread(
            app.send_task,
            settings.QUEUE_INGESTION_AGGREGATOR,
            kwargs={"resume_id": resume_id},
            queue=settings.QUEUE_INGESTION_AGGREGATOR,
        )

        print(f"[Resume {resume_id}] ✅ Completed successfully.")

    except Exception as e:
        error_message = str(e)
        print(f"[Resume {resume_id}] ❌ FAILED: {error_message}")

        publish_event(app, "resume.embedding.failed", {
            "resume_id": resume_id,
            "batch_id": batch_id,
            "error": error_message,
        })

        try:
            async with get_db_session() as db:
                resume = await db.get(Resume, resume_id)
                if resume:
                    resume.status = "EMBED_FAILED"
                    resume.is_embedded = False
                    db.add(ScoreLog(
                        resume_id=resume_id,
                        batch_id=batch_id,
                        worker_name="embedder_worker",
                        status="FAILED",
                        message=error_message
                    ))
                    await db.commit()
        except Exception as db_e:
            print(f"[Resume {resume_id}] DB Update Error: {db_e}")


# --------------------------------------------------------------------------
# Celery Task (Sync Wrapper)
# --------------------------------------------------------------------------

@app.task(name=settings.QUEUE_EMBEDDER, bind=True, acks_late=True, ignore_result=False)
def embed_resume(self, resume_id: int):
    """Celery entrypoint – run on persistent async loop thread."""
    global loop
    if not loop or not loop.is_running():
        print(f"[Resume {resume_id}] ERROR: Async loop not running.")
        return

    future = asyncio.run_coroutine_threadsafe(async_embed_resume(resume_id), loop)
    try:
        return future.result()
    except Exception as e:
        print(f"[Resume {resume_id}] Async execution failed: {e}")
        raise
