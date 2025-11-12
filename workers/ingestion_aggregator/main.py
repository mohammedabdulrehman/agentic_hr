import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.future import select

# Import our settings
from settings import settings

# Import Shared Models from libs
from schemas.models import Resume, ScoreLog
# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------

app = Celery(
    "ingestion_aggregator",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_INGESTION_AGGREGATOR: {
        "exchange": settings.QUEUE_INGESTION_AGGREGATOR,
        "routing_key": settings.QUEUE_INGESTION_AGGREGATOR,
    }
}
app.conf.task_default_queue = settings.QUEUE_INGESTION_AGGREGATOR

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
    print("IngestionAggregator: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    print("IngestionAggregator: Async resources initialized.")

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
    print("IngestionAggregator: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("IngestionAggregator: Async resources shut down.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for IngestionAggregator.")
        
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The Aggregator Task
# --------------------------------------------------------------------------

async def async_aggregate_ingestion(resume_id: int):
    """
    This is our *actual* async logic.
    It checks if both parallel ingestion tasks are complete.
    """
    print(f"[Resume {resume_id}]: Async ingestion aggregator task started.")
    
    async with get_db_session() as db:
        try:
            # 1. Get the resume record and LOCK THE ROW
            resume = await db.get(Resume, resume_id, with_for_update=True)
            if not resume:
                print(f"[Resume {resume_id}]: ERROR - Resume not found.")
                return

            # If we're already done, don't run again
            if resume.status == 'READY_TO_SCORE':
                print(f"[Resume {resume_id}]: Already aggregated. Ignoring.")
                await db.commit() # Release lock
                return

            # 2. --- THE BIG CHECK ---
            if resume.is_embedded and resume.is_extracted:
                # YES! Both tasks are done.
                print(f"[Resume {resume_id}]: Both embedding and extraction are complete.")
                
                # 3. Update the final status. This is the end of Pipeline 1.
                resume.status = 'READY_TO_SCORE'
                await db.commit() # Commit the status change and release lock
                
                print(f"[Resume {resume_id}]: --- INGESTION PIPELINE COMPLETE ---")
                
            else:
                # NO. We're still waiting for the other worker.
                print(f"[Resume {resume_id}]: Still waiting for other task. (Embedded: {resume.is_embedded}, Extracted: {resume.is_extracted})")
                # We just commit to release the lock and end the task.
                await db.commit()

        except Exception as e:
            print(f"[Resume {resume_id}]: --- TASK FAILED --- Error: {e}")
            if 'db' in locals() and 'resume' in locals() and resume:
                try:
                    await db.rollback() 
                    resume.status = 'INGESTION_AGGREGATE_FAILED'
                    await db.commit()
                except Exception as db_e:
                    print(f"CRITICAL: Failed to write failure status to DB: {db_e}")

# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_INGESTION_AGGREGATOR, bind=True)
def aggregate_ingestion(self, resume_id: int):
    """
    This is the SYNCHRONOUS Celery task that runs our async aggregation logic.
    """
    try:
        asyncio.run(async_aggregate_ingestion(resume_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        # raise self.retry(exc=e, countdown=60)