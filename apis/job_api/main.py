# /apis/job_api/main.py

import asyncio
import redis.asyncio as async_redis  # Use a specific alias
import redis                      # For exceptions
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from celery import Celery

# Import all models so Base.metadata can build them all
from schemas.models import (
    Base, Job, ExtractionSchema, ScoringRubric, 
    Resume, Score, ScoreLog, 
    SemanticScore, LlmScore
)

from settings import settings
from libs.monitoring import publish_event

# --------------------------------------------------------------------------
# Database Setup
# --------------------------------------------------------------------------

async_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)

async def create_db_tables():
    """Creates all tables defined in Base.metadata."""
    print("JobAPI: Checking/Creating database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("JobAPI: Database tables are ready.")

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Provides an async database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# API Lifecycle (Connects to DB, Celery, and Redis)
# --------------------------------------------------------------------------

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("JobAPI starting up...")
    await create_db_tables()
    app_state["celery_app"] = Celery("job_api", broker=settings.BROKER_URL)
    
    print(f"JobAPI: Creating Redis pool for {settings.REDIS_HOST}...")
    app_state["redis_pool"] = async_redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    
    print("JobAPI started successfully.")
    yield
    print("JobAPI shutting down...")
    
    if "redis_pool" in app_state:
        pool = app_state["redis_pool"]
        await pool.disconnect()
        print("JobAPI: Redis pool disconnected.")
    
    await async_engine.dispose()
    print("JobAPI: DB engine disposed.")

app = FastAPI(title="Job & Schema API", lifespan=lifespan)

# --- Dependency injectors ---
def get_celery():
    return app_state["celery_app"]

async def get_redis() -> AsyncGenerator[async_redis.Redis, None]:
    """Dependency injector for a single Redis connection from the pool."""
    if "redis_pool" not in app_state:
        raise HTTPException(status_code=500, detail="Redis connection pool not initialized.")
    
    r = async_redis.Redis(connection_pool=app_state["redis_pool"])
    try:
        yield r
    except redis.exceptions.ConnectionError as e:
        print(f"Redis connection error: {e}")
        raise HTTPException(status_code=503, detail="Could not connect to Redis service.")
    finally:
        await r.close()

# --------------------------------------------------------------------------
# --- THIS IS THE FIX ---
# Pydantic Request Models are now fully defined
# --------------------------------------------------------------------------

class SchemaIn(BaseModel):
    schema_name: str
    schema_definition: Dict[str, Any]

class RubricIn(BaseModel):
    rubric_name: str
    rubric_definition: Dict[str, Any]

class JobIn(BaseModel):
    # This field is required by the DB, so it must be required here.
    title: str 
    
    # These fields are nullable in the DB, so they can be optional.
    description: Optional[str] = None
    extraction_schema_id: Optional[int] = None
    scoring_rubric_id: Optional[int] = None

class JobStatusResponse(BaseModel):
    job_id: int
    title: str
    status: str

class RichJobStatusResponse(BaseModel):
    job_id: int
    source: str
    status_details: Optional[Dict[str, Any]] = None

# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------

@app.post("/api/v1/schemas", status_code=201)
async def create_extraction_schema(schema: SchemaIn, db: AsyncSession = Depends(get_async_db)):
    new_schema = ExtractionSchema(**schema.model_dump())
    db.add(new_schema)
    await db.commit()
    await db.refresh(new_schema)
    return new_schema

@app.post("/api/v1/rubrics", status_code=201)
async def create_scoring_rubric(rubric: RubricIn, db: AsyncSession = Depends(get_async_db)):
    new_rubric = ScoringRubric(**rubric.model_dump())
    db.add(new_rubric)
    await db.commit()
    await db.refresh(new_rubric)
    return new_rubric

@app.post("/api/v1/jobs", status_code=201)
async def create_job(
    job: JobIn, # <-- This will now validate that 'title' is present
    db: AsyncSession = Depends(get_async_db),
    celery_app = Depends(get_celery)
):
    # Because of the fix to JobIn, job.model_dump() will
    # no longer have 'title: None' unless you explicitly send it.
    new_job = Job(**job.model_dump())
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)
    
    publish_event(
        celery_app_instance=celery_app,
        routing_key="job.created",
        data={ "job_id": new_job.job_id, "title": new_job.title }
    )
    
    await asyncio.to_thread(
        celery_app.send_task,
        settings.QUEUE_JOB_EMBEDDER,
        kwargs={"job_id": new_job.job_id},
        queue=settings.QUEUE_JOB_EMBEDDER
    )
    
    return new_job

@app.get("/api/v1/jobs/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: int, db: AsyncSession = Depends(get_async_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.job_id, title=job.title, status=job.status)

@app.get("/api/v1/jobs/dashboard-status/{job_id}", response_model=RichJobStatusResponse)
async def get_dashboard_job_status(
    job_id: int, 
    r: async_redis.Redis = Depends(get_redis)
):
    key = f"job:{job_id}"
    status_details = await r.hgetall(key)
    
    if not status_details:
        raise HTTPException(
            status_code=404, 
            detail="Job status not found in dashboard cache. The job may not have started or data has expired."
        )
        
    return RichJobStatusResponse(
        job_id=job_id,
        source="redis_dashboard_cache",
        status_details=status_details
    )