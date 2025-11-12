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
from settings import settings
from schemas.models import Job

# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------

app = Celery(
    "job_embedder_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_JOB_EMBEDDER: {
        "exchange": settings.QUEUE_JOB_EMBEDDER,
        "routing_key": settings.QUEUE_JOB_EMBEDDER,
    }
}
app.conf.task_default_queue = settings.QUEUE_JOB_EMBEDDER


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
    """Start a dedicated asyncio loop thread for this worker process."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_forever()


@worker_process_init.connect
def on_worker_init(**kwargs):
    """
    Called when a Celery worker process starts.
    We create the persistent event loop and load all async resources once.
    """
    global resources, loop_thread, loop
    resources = AsyncResources()

    print("JobEmbedderWorker: Loading embedding model...")
    resources.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("JobEmbedderWorker: Embedding model loaded.")

    # Start dedicated asyncio loop in a separate thread
    loop_thread = threading.Thread(target=start_loop_in_thread, daemon=True)
    loop_thread.start()

    # Wait for loop to be alive
    while loop is None or not loop.is_running():
        asyncio.sleep(0.1)

    # Initialize async resources inside that loop
    fut = asyncio.run_coroutine_threadsafe(setup_async_resources(resources), loop)
    fut.result()  # block until initialization completes
    print("JobEmbedderWorker: Async loop and resources initialized.")


async def setup_async_resources(res: AsyncResources):
    print("JobEmbedderWorker: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    res.qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    try:
        # 1. Check if the collection already exists
        collection_exists = await res.qdrant_client.collection_exists(
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        
        # 2. Only create if it does NOT exist
        if not collection_exists:
            print(f"Worker: Collection '{settings.QDRANT_COLLECTION_NAME}' not found. Creating...")
            
            await res.qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=res.embedding_model.get_sentence_embedding_dimension(), 
                    distance=models.Distance.COSINE
                ),
                on_disk_payload=True
            )
            print("Worker: Qdrant collection created.")
        else:
            print(f"Worker: Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' already exists. No action taken.")
            
    except Exception as e:
        print(f"Worker: ERROR checking/creating Qdrant collection: {e}")
        raise e

    print("Worker: Async resources initialized.")


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Clean shutdown of async resources and event loop thread."""
    global loop, loop_thread
    if not resources or not loop:
        return

    print("JobEmbedderWorker: Shutting down async resources...")
    fut = asyncio.run_coroutine_threadsafe(shutdown_async_resources(resources), loop)
    fut.result()

    loop.call_soon_threadsafe(loop.stop)
    if loop_thread and loop_thread.is_alive():
        loop_thread.join(timeout=5)
    print("JobEmbedderWorker: Shutdown complete.")


async def shutdown_async_resources(res: AsyncResources):
    if res.db_engine:
        await res.db_engine.dispose()
    if res.qdrant_client:
        await res.qdrant_client.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for JobEmbedderWorker.")

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
    """Helper to split text into overlapping chunks."""
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


async def async_embed_job(job_id: int):
    """
    Actual async logic for embedding a job.
    """
    print(f"[Job {job_id}]: Received embed task.")

    async with get_db_session() as db:
        try:
            publish_event(app, "job.embedding.started", {"job_id": job_id})

            # 1. Fetch job
            job = await db.get(Job, job_id)
            if not job or not job.description:
                print(f"[Job {job_id}]: ERROR - Job or description not found.")
                return

            job.status = 'EMBEDDING'
            await db.commit()

            # 2. Chunk text
            text_chunks = chunk_text(job.description)
            if not text_chunks:
                raise Exception("Text chunking resulted in no chunks.")

            print(f"[Job {job_id}]: Generating embeddings for {len(text_chunks)} chunks...")

            # 3. Generate embeddings
            vectors = await asyncio.to_thread(resources.embedding_model.encode, text_chunks)

            # 4. Store in Qdrant
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={"job_id": job.job_id, "type": "job", "text": chunk}
                )
                for chunk, vector in zip(text_chunks, vectors)
            ]

            if points:
                await resources.qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=points,
                    wait=True
                )
                print(f"[Job {job_id}]: Successfully saved {len(points)} job vectors to Qdrant.")

            job.status = 'READY'
            await db.commit()
            print(f"[Job {job_id}]: --- JOB INGESTION COMPLETE ---")

            publish_event(app, "job.embedding.completed", {"job_id": job_id, "vectors_created": len(vectors)})

        except Exception as e:
            print(f"[Job {job_id}]: --- TASK FAILED ---")
            print(f"Error: {e}")
            publish_event(app, "job.embedding.failed", {"job_id": job_id, "error": str(e)})

            try:
                await db.rollback()
                if job:
                    job.status = 'JOB_EMBED_FAILED'
                    await db.commit()
            except Exception as db_e:
                print(f"CRITICAL: Failed to write failure status to DB: {db_e}")


# --------------------------------------------------------------------------
# Celery Task Wrapper
# --------------------------------------------------------------------------

@app.task(name=settings.QUEUE_JOB_EMBEDDER, bind=True)
def embed_job(self, job_id: int):
    """
    Schedule async embedding on the persistent event loop thread.
    """
    global loop
    if not loop or not loop.is_running():
        print(f"[Job {job_id}] ERROR: Async loop not running.")
        return

    future = asyncio.run_coroutine_threadsafe(async_embed_job(job_id), loop)
    try:
        result = future.result()
        return result
    except Exception as e:
        print(f"[Job {job_id}] Async execution failed: {e}")
        raise
