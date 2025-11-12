import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from libs.monitoring import publish_event

# Import our settings
from settings import settings

# Import Shared Models from libs
from schemas.models import (
    Resume, Job, Score, ScoreLog,
    LlmScore, SemanticScore, ExtractionSchema, ScoringRubric, Base
)

# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------

app = Celery(
    "scoring_engine",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_SCORING_ENGINE: {
        "exchange": settings.QUEUE_SCORING_ENGINE,
        "routing_key": settings.QUEUE_SCORING_ENGINE,
    }
}
app.conf.task_default_queue = settings.QUEUE_SCORING_ENGINE

# --------------------------------------------------------------------------
# Async Resource Management (Just the DB)
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None

resources: AsyncResources = None

@worker_process_init.connect
def on_worker_init(**kwargs):
    global resources
    resources = AsyncResources()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(setup_async_resources(resources))

async def setup_async_resources(res: AsyncResources):
    print("ScoringEngine: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    print("ScoringEngine: Async resources initialized.")

@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    if resources:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(shutdown_async_resources(resources))

async def shutdown_async_resources(res: AsyncResources):
    print("ScoringEngine: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("ScoringEngine: Async resources shut down.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    global resources

    # Lazy initialization fallback
    if not resources or not resources.db_sessionmaker:
        print("ScoringEngine: Lazy initializing async resources...")
        resources = AsyncResources()
        await setup_async_resources(resources)

    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The Scoring Engine Task
# --------------------------------------------------------------------------

async def async_start_scoring(score_id: int): # <-- Receives score_id
    """
    This is our *actual* async logic.
    It kicks off the parallel scoring pipeline.
    """
    print(f"[Score {score_id}]: Async scoring engine task started.")
    batch_id = None
    async with get_db_session() as db:
        try:
            # 1. Get the *Score* record to ensure it exists
            score = await db.get(Score, score_id)
            if not score:
                print(f"[Score {score_id}]: ERROR - Score row not found.")
                return
            batch_id = str(score.batch_id) if score.batch_id else None
            # 2. Update status on the Score object
            score.status = 'SCORING_RUBRIC'
            score.sub_scores_json = {} # Initialize empty JSON
            await db.commit()
            # --- 3. PUBLISH "STARTED" EVENT ---
            # (This was already published by match_api, but this is a good confirmation)
            publish_event(
                app,
                "score.scoring.started",
                {"score_id": score_id, "batch_id": batch_id}
            )
            # 3. Fire off the two parallel scoring tasks
            # We now pass the score_id
            await asyncio.to_thread(
                app.send_task,
                settings.QUEUE_SEMANTIC_SCORER,
                kwargs={"score_id": score.score_id},
                queue=settings.QUEUE_SEMANTIC_SCORER
            )
            await asyncio.to_thread(
                app.send_task,
                settings.QUEUE_LLM_SCORER,
                kwargs={"score_id": score.score_id},
                queue=settings.QUEUE_LLM_SCORER
            )
            # --- 5. PUBLISH EVENTS FOR THE *NEXT* WORKERS ---
            # This tells the dashboard that these specific workers have been queued
            publish_event(
                app,
                "score.semantic.started", # <-- This tells the dashboard the *next* step
                {"score_id": score_id, "batch_id": batch_id}
            )
            publish_event(
                app,
                "score.llm.started", # <-- This tells the dashboard the *next* step
                {"score_id": score_id, "batch_id": batch_id}
            )
            print(f"[Score {score_id}]: Fired tasks for Semantic and LLM scorers.")

        except Exception as e:
            print(f"[Score {score_id}]: --- TASK FAILED ---")
            print(f"Error: {e}")
            if 'db' in locals() and 'score' in locals() and score:
                try:
                    await db.rollback()
                    score.status = 'RUBRIC_FAILED'
                    await db.commit()
                except Exception as db_e:
                    print(f"CRITICAL: Failed to write failure status to DB: {db_e}")

# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_SCORING_ENGINE, bind=True)
def start_scoring(self, score_id: int): # <-- Receives score_id
    """
    This is the SYNCHRONOUS Celery task that runs our async orchestration logic.
    """
    try:
        asyncio.run(async_start_scoring(score_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        # raise self.retry(exc=e, countdown=60)