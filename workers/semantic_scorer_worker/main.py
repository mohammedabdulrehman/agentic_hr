import asyncio
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import joinedload
from qdrant_client import AsyncQdrantClient, models
from settings import settings
from libs.monitoring import publish_event

# --- IMPORT MODELS ---
from schemas.models import (
    Resume, Job, Score, SemanticScore
)

# --------------------------------------------------------------------------
# Celery Configuration
# --------------------------------------------------------------------------

app = Celery(
    "semantic_embedder_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_SEMANTIC_SCORER: {
        "exchange": settings.QUEUE_SEMANTIC_SCORER,
        "routing_key": settings.QUEUE_SEMANTIC_SCORER,
    }
}
app.conf.task_default_queue = settings.QUEUE_SEMANTIC_SCORER


# --------------------------------------------------------------------------
# Async Resource Management
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    qdrant_client = None


resources: AsyncResources = None
loop = None
loop_thread = None


def start_loop_in_thread():
    """Start a persistent asyncio event loop in a separate thread."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_forever()


@worker_process_init.connect
def on_worker_init(**kwargs):
    """Initialize async resources when Celery worker starts."""
    global resources, loop_thread, loop
    resources = AsyncResources()

    print("SemanticScorer: Starting persistent async loop thread...")
    loop_thread = threading.Thread(target=start_loop_in_thread, daemon=True)
    loop_thread.start()

    # Wait until loop starts
    while loop is None or not loop.is_running():
        pass

    print("SemanticScorer: Initializing async resources inside event loop...")
    fut = asyncio.run_coroutine_threadsafe(setup_async_resources(resources), loop)
    fut.result()
    print("SemanticScorer: Async resources initialized.")


async def setup_async_resources(res: AsyncResources):
    """Initialize DB engine and Qdrant client inside the loop."""
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    res.qdrant_client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Cleanly shut down resources."""
    global loop, loop_thread
    print("SemanticScorer: Shutting down async resources...")
    if loop and loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(shutdown_async_resources(resources), loop)
        fut.result(timeout=10)

        loop.call_soon_threadsafe(loop.stop)
        if loop_thread and loop_thread.is_alive():
            loop_thread.join(timeout=5)
    print("SemanticScorer: Shutdown complete.")


async def shutdown_async_resources(res: AsyncResources):
    """Asynchronously close DB engine and Qdrant client."""
    if res.db_engine:
        await res.db_engine.dispose()
    if res.qdrant_client:
        await res.qdrant_client.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("SemanticScorer async resources not initialized.")
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# --------------------------------------------------------------------------
# Core Logic
# --------------------------------------------------------------------------

async def run_single_search(qdrant_client: AsyncQdrantClient, query_vector: list, resume_id: int) -> float:
    """Run a single Qdrant search comparing a job vector to a resume vector set."""
    results = await qdrant_client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=1,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="resume_id", match=models.MatchValue(value=resume_id))]
        ),
    )
    return round(results[0].score, 3) if results else 0.0


async def async_score_semantic(score_id: int):
    """Compute semantic similarity between job and resume embeddings."""
    print(f"[Score {score_id}]: Async semantic scoring started.")
    publish_event(app, "semantic.embedding.started", {"score_id": score_id})

    try:
        final_score = 0.0

        # --- 1. Fetch Job & Resume ---
        async with get_db_session() as db:
            score_obj = await db.get(Score, score_id, options=[
                joinedload(Score.resume, innerjoin=True),
                joinedload(Score.job, innerjoin=True)
            ])
            if not (score_obj and score_obj.resume and score_obj.job):
                raise ValueError(f"[Score {score_id}] Missing score, resume, or job.")
            job_id = score_obj.job.job_id
            resume_id = score_obj.resume.resume_id
            print(f"[Score {score_id}]: Processing Job {job_id} vs Resume {resume_id}...")

        # --- 2. Fetch job vectors from Qdrant ---
        scroll_result = await resources.qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="job_id", match=models.MatchValue(value=job_id))]
            ),
            with_vectors=True,
            limit=100
        )

        job_vectors = scroll_result[0]
        if not job_vectors:
            raise Exception(f"No vectors found for job {job_id}.")

        # --- 3. Run comparisons concurrently ---
        scores = await asyncio.gather(
            *[run_single_search(resources.qdrant_client, job_point.vector, resume_id)
              for job_point in job_vectors]
        )

        filtered = [s for s in scores if s > 0]
        if filtered:
            final_score = sum(filtered) / len(filtered)
        print(f"[Score {score_id}]: Final semantic score = {final_score}")

        # --- 4. Save score ---
        async with get_db_session() as db:
            db.add(SemanticScore(score_id=score_id, score_value=float(final_score)))
            await db.commit()
            print(f"[Score {score_id}]: Saved semantic score.")

        # --- 5. Dispatch event ---
        await asyncio.to_thread(
            app.send_task,
            settings.QUEUE_AGGREGATOR,
            kwargs={"score_id": score_id},
            queue=settings.QUEUE_AGGREGATOR
        )

        publish_event(app, "semantic.embedding.completed", {
            "score_id": score_id,
            "final_score": final_score
        })
        print(f"[Score {score_id}]: Semantic scoring completed successfully.")

    except Exception as e:
        print(f"[Score {score_id}] --- TASK FAILED --- {e}")
        publish_event(app, "semantic.embedding.failed", {
            "score_id": score_id,
            "error": str(e)
        })
        async with get_db_session() as db:
            score_fail = await db.get(Score, score_id)
            if score_fail:
                score_fail.status = 'SEMANTIC_SCORE_FAILED'
                await db.commit()


# --------------------------------------------------------------------------
# Celery Task Entrypoint
# --------------------------------------------------------------------------

@app.task(name=settings.QUEUE_SEMANTIC_SCORER, bind=True)
def score_semantic(self, score_id: int):
    """Schedule async scoring on the persistent event loop."""
    global loop
    if not loop or not loop.is_running():
        print(f"[Score {score_id}] ERROR: Async loop not running.")
        return

    future = asyncio.run_coroutine_threadsafe(async_score_semantic(score_id), loop)
    try:
        return future.result()
    except Exception as e:
        print(f"[Score {score_id}] Async execution failed: {e}")
        raise
