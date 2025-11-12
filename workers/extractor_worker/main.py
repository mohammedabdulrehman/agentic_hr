import asyncio
import instructor
from openai import AsyncOpenAI
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, Field, create_model

# --- UPDATED IMPORTS ---
from libs.monitoring import publish_event
from schemas.models import Resume, ExtractionSchema, ScoreLog
# -------------------------

from settings import settings

# --------------------------------------------------------------------------
# Dynamic Pydantic Model Generator
# --------------------------------------------------------------------------
TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "List[str]": List[str],
    "list[str]": List[str],
}

def create_dynamic_model(schema_name: str, schema_definition: dict) -> BaseModel:
    """Generates a Pydantic BaseModel class in memory from our DB JSON definition."""
    fields_to_create = {}
    for field_name, definition in schema_definition.items():
        field_type_str = definition.get("type", "str")
        field_type = TYPE_MAP.get(field_type_str, str)
        if field_type is str and field_type_str != "str":
              print(f"Warning: Unknown type '{field_type_str}'. Defaulting to 'str'.")
        
        field_description = definition.get("description", None)
        field_info = Field(description=field_description)
        fields_to_create[field_name] = (field_type, field_info)

    DynamicModel = create_model(schema_name, __base__=BaseModel, **fields_to_create)
    return DynamicModel

# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------

app = Celery(
    "extractor_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_EXTRACTOR: {
        "exchange": settings.QUEUE_EXTRACTOR,
        "routing_key": settings.QUEUE_EXTRACTOR,
    }
}
app.conf.task_default_queue = settings.QUEUE_EXTRACTOR

# --------------------------------------------------------------------------
# --- FIXED: Async Resource Management ---
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    openai_client = None

resources: AsyncResources = None

def setup_resources(res: AsyncResources):
    """Synchronously create async-capable resources."""
    print("ExtractorWorker: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    # Creating the client object is sync
    res.openai_client = instructor.patch(
        AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        mode=instructor.Mode.TOOLS
    )
    print("ExtractorWorker: Async resources initialized.")

@worker_process_init.connect
def on_worker_init(**kwargs):
    global resources
    resources = AsyncResources()
    setup_resources(resources) # <-- SYNC call

@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    global resources
    if resources:
        asyncio.run(shutdown_async_resources(resources)) # <-- ASYNC call

async def shutdown_async_resources(res: AsyncResources):
    print("ExtractorWorker: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    # No async close needed for openai client
    print("ExtractorWorker: Async resources shut down.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for ExtractorWorker.")
    
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# --- REFACTORED: The Extractor Task ---
# --------------------------------------------------------------------------

async def async_extract_profile(resume_id: int):
    print(f"[Resume {resume_id}]: Received extract task.")
    
    batch_id = None
    parsed_text = None
    schema_def = None
    schema_name = None
    
    try:
        # --- SESSION 1: Get data, update status (Short Transaction) ---
        async with get_db_session() as db:
            resume = await db.get(Resume, resume_id)
            if not resume or not resume.parsed_text:
                raise Exception(f"Resume {resume_id} or parsed_text not found.")
            
            schema_id = resume.extraction_schema_id
            if not schema_id:
                raise Exception(f"Resume {resume_id} has no extraction_schema_id set.")

            schema = await db.get(ExtractionSchema, schema_id)
            if not schema:
                raise Exception(f"Extraction schema (ID: {schema_id}) not found.")

            # Copy data out of the session
            batch_id = str(resume.batch_id) if resume.batch_id else None
            parsed_text = resume.parsed_text
            schema_def = schema.schema_definition
            schema_name = schema.schema_name
            
            # Update status and log
            resume.status = 'EXTRACTING'
            db.add(ScoreLog(
                resume_id=resume_id,
                batch_id=batch_id,
                worker_name="extractor_worker", 
                status="STARTED",
                message="Profile extraction started."
            ))
            await db.commit() # Commit status change, release lock

        # --- 2. PUBLISH "STARTED" EVENT ---
        publish_event(
            app, 
            "resume.extraction.started", 
            {"resume_id": resume_id, "batch_id": batch_id, "schema_name": schema_name}
        )
        
        # --- 3. CPU/API Logic (Outside DB session, NO LOCK HELD) ---
        print(f"[Resume {resume_id}]: Using model '{schema_name}'...")
        DynamicExtractionModel = create_dynamic_model(schema_name, schema_def)
        
        profile: BaseModel = await resources.openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            response_model=DynamicExtractionModel,
            messages=[
                {"role": "system", "content": f"You are an expert HR resume parser. Extract the candidate's profile using the '{schema_name}' structure."},
                {"role": "user", "content": parsed_text}
            ]
        )

        # --- SESSION 2: Save results (Short Transaction) ---
        async with get_db_session() as db:
            resume = await db.get(Resume, resume_id) # Re-fetch
            if resume:
                resume.profile_json = profile.model_dump()
                resume.is_extracted = True
                resume.status = 'EXTRACTION_COMPLETE'
                db.add(ScoreLog(
                    resume_id=resume_id,
                    batch_id=batch_id,
                    worker_name="extractor_worker", 
                    status="COMPLETED",
                    message="Profile extraction successful."
                ))
                await db.commit()
            print(f"[Resume {resume_id}]: Successfully extracted profile. Set is_extracted=True.")

        # --- 4. PUBLISH "COMPLETED" EVENT ---
        publish_event(
            app, 
            "resume.extraction.completed", 
            {"resume_id": resume_id, "batch_id": batch_id}
        )

        # --- 5. Send task to INGESTION AGGREGATOR ---
        print(f"[Resume {resume_id}]: Sending task to queue: {settings.QUEUE_INGESTION_AGGREGATOR}")
        await asyncio.to_thread(
            app.send_task,
            settings.QUEUE_INGESTION_AGGREGATOR,
            kwargs={"resume_id": resume_id},
            queue=settings.QUEUE_INGESTION_AGGREGATOR
        )

    except Exception as e:
        error_message = str(e)
        print(f"[Resume {resume_id}]: --- TASK FAILED --- Error: {error_message}")
        
        # --- 6. PUBLISH "FAILED" EVENT ---
        publish_event(
            app, 
            "resume.extraction.failed", 
            {"resume_id": resume_id, "batch_id": batch_id, "error": error_message}
        )
        
        # --- SESSION 3: Update status to FAILED ---
        try:
            async with get_db_session() as error_db:
                resume = await error_db.get(Resume, resume_id)
                if resume:
                    resume.status = 'EXTRACT_FAILED'
                    resume.is_extracted = False
                    error_db.add(ScoreLog(
                        resume_id=resume_id,
                        batch_id=batch_id,
                        worker_name="extractor_worker", 
                        status="FAILED",
                        message=error_message
                    ))
                    await error_db.commit()
        except Exception as db_e:
            print(f"CRITICAL: Failed to write failure status to DB: {db_e}")
            
        raise e # Re-raise to Celery

# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_EXTRACTOR, bind=True)
def extract_profile(self, resume_id: int):
    """
    This is the SYNCHRONOUS Celery task that runs our async logic.
    """
    try:
        if not resources:
            raise Exception("ExtractorWorker resources not initialized.")
        asyncio.run(async_extract_profile(resume_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        # Re-raise the exception so Celery knows the task failed
        raise