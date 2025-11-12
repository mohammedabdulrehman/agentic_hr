# /workers/parser_worker/main.py

import asyncio
import aiohttp
import aiobotocore.session
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# --- UPDATED IMPORTS ---
from libs.monitoring import publish_event
from schemas.models import Resume, ScoreLog 
# -------------------------

from settings import settings

# --------------------------------------------------------------------------
# Celery App Configuration (Unchanged)
# --------------------------------------------------------------------------
app = Celery(
    "parser_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)
app.conf.task_queues = {
    settings.QUEUE_PARSER: {
        "exchange": settings.QUEUE_PARSER,
        "routing_key": settings.QUEUE_PARSER,
    }
}
app.conf.task_default_queue = settings.QUEUE_PARSER
# --------------------------------------------------------------------------
# --- FIXED: Async Resource Management ---
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    # --- S3 Client is REMOVED from global resources ---

resources: AsyncResources = None

@worker_process_init.connect
def on_worker_init(**kwargs):
    global resources
    resources = AsyncResources()
    # --- This is now a SYNC call ---
    setup_resources(resources)

def setup_resources(res: AsyncResources):
    """Synchronously create async-capable resources (DB Only)."""
    print("ParserWorker: Initializing DB resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    print("ParserWorker: DB resources initialized.")

@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    if resources:
        # --- This now uses asyncio.run ---
        asyncio.run(shutdown_async_resources(resources))

async def shutdown_async_resources(res: AsyncResources):
    """Asynchronously dispose of resources."""
    print("ParserWorker: Shutting down DB resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("ParserWorker: DB resources shut down.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for ParserWorker.")
    
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The Parser Task (With Monitoring)
# --------------------------------------------------------------------------

async def async_parse_resume(resume_id: int):
    print(f"[Resume {resume_id}]: Received parse task.")
    
    batch_id = None
    
    # --- Create local async clients for this task ---
    s3_session = aiobotocore.session.get_session()
    async with s3_session.create_client(
        "s3",
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        aws_access_key_id=settings.MINIO_ROOT_USER,
        aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
    ) as s3_client: # <-- S3 client is created here
        
        async with aiohttp.ClientSession() as http_session:
            async with get_db_session() as db:
                try:
                    # 1. Get the resume record
                    resume = await db.get(Resume, resume_id)
                    if not resume:
                        raise Exception(f"Resume {resume_id} not found in DB.")
                    if not resume.file_key:
                        raise Exception(f"Resume {resume_id} has no file_key.")
                        
                    file_key = resume.file_key
                    batch_id = str(resume.batch_id) if resume.batch_id else None
                    
                    # 2. Update status and add log
                    resume.status = 'PARSING'
                    db.add(ScoreLog(
                        resume_id=resume_id,
                        batch_id=batch_id,
                        worker_name="parser_worker", 
                        status="STARTED",
                        message="Parsing started."
                    ))
                    await db.commit()
            
                    # --- 3. PUBLISH "STARTED" EVENT ---
                    publish_event(
                        app, 
                        "resume.parsing.started", 
                        {"resume_id": resume_id, "batch_id": batch_id, "filename": resume.filename}
                    )
                    # ------------------------------------

                    # 4. Download file from MinIO
                    print(f"[Resume {resume_id}]: Downloading {file_key} from MinIO...")
                    s3_response = await s3_client.get_object(
                        Bucket=settings.MINIO_BUCKET_NAME,
                        Key=file_key
                    )
                    async with s3_response['Body'] as stream:
                        file_content = await stream.read()
                    
                    # 5. Send file to unstructured-api
                    print(f"[Resume {resume_id}]: Sending file to unstructured-api...")
                    form_data = aiohttp.FormData()
                    form_data.add_field('files', file_content, filename=resume.filename)
                    
                    async with http_session.post(settings.UNSTRUCTURED_API_URL, data=form_data) as response:
                        response.raise_for_status()
                        parsed_data = await response.json()
                    
                    # 6. Extract text
                    parsed_text = "\n\n".join([el.get("text", "") for el in parsed_data if el.get("text")])
                    if not parsed_text:
                        raise Exception("unstructured-api returned no text.")

                    # 7. Save text and update status in Postgres
                    resume.parsed_text = parsed_text
                    resume.status = 'PARSED' # This step is done
                    db.add(ScoreLog(
                        resume_id=resume_id,
                        batch_id=batch_id,
                        worker_name="parser_worker", 
                        status="COMPLETED",
                        message=f"Parsing complete. Text length: {len(parsed_text)}"
                    ))
                    await db.commit()
                    print(f"[Resume {resume_id}]: Successfully parsed and saved text.")

                    # --- 8. PUBLISH "COMPLETED" EVENT ---
                    publish_event(
                        app, 
                        "resume.parsing.completed", 
                        {"resume_id": resume_id, "batch_id": batch_id, "text_length": len(parsed_text)}
                    )
                    # ------------------------------------
            
                    # --- 9. --- FIRE TASKS IN PARALLEL ---
                    print(f"[Resume {resume_id}]: Firing parallel tasks for embedding and extraction...")
                        
                    # We call asyncio.to_thread *inside* gather
                    await asyncio.gather(
                        asyncio.to_thread(
                            app.send_task,
                            settings.QUEUE_EMBEDDER,
                            kwargs={"resume_id": resume.resume_id},
                            queue=settings.QUEUE_EMBEDDER
                        ),
                        asyncio.to_thread(
                            app.send_task,
                            settings.QUEUE_EXTRACTOR,
                            kwargs={"resume_id": resume.resume_id},
                            queue=settings.QUEUE_EXTRACTOR
                        )
                    )
                    print(f"[Resume {resume_id}]: Parallel ingestion tasks fired.")

                except Exception as e:
                    print(f"[Resume {resume_id}]: --- TASK FAILED --- Error: {e}")
                    
                    # --- 10. PUBLISH "FAILED" EVENT ---
                    publish_event(
                        app, 
                        "resume.parsing.failed", 
                        {"resume_id": resume_id, "batch_id": batch_id, "error": str(e)}
                    )
                    # ------------------------------------
                    
                    if 'db' in locals() and 'resume' in locals() and resume:
                        try:
                            await db.rollback()
                            resume.status = 'PARSE_FAILED'
                            db.add(ScoreLog(
                                resume_id=resume_id,
                                batch_id=batch_id,
                                worker_name="parser_worker", 
                                status="FAILED",
                                message=str(e)
                            ))
                            await db.commit()
                        except Exception as db_e:
                            print(f"CRITICAL: Failed to write failure status to DB: {db_e}")
                    raise # Re-raise error to Celery

# --- Sync Wrapper (Unchanged) ---
@app.task(name=settings.QUEUE_PARSER, bind=True)
def parse_resume(self, resume_id: int):
    try:
        asyncio.run(async_parse_resume(resume_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        # We re-raised inside, so Celery will see the failure