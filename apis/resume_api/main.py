import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator,List,Optional,Any,Dict
import redis.asyncio as async_redis  # <-- IMPORT REDIS
import redis                      # <-- IMPORT REDIS (for exceptions)
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks
from pydantic import BaseModel
from celery import Celery
from aiohttp import ClientError
import aiobotocore.session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from libs.monitoring import publish_event
from schemas.models import Resume, ScoreLog # <-- Import ScoreLog
from settings import settings
from sqlalchemy.future import select    # <-- ADD IMPORT
from sqlalchemy.orm import joinedload # <-- ADD IMPORT
# --------------------------------------------------------------------------
# Database Setup (Read-Only, no table creation)
# --------------------------------------------------------------------------
async_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, expire_on_commit=False)

async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ResumeAPI starting up...")
    
    # Connect to MinIO/S3
    session = aiobotocore.session.get_session()
    s3_client_context = session.create_client(
        "s3",
        endpoint_url=settings.MINIO_ENDPOINT_URL,
        aws_access_key_id=settings.MINIO_ROOT_USER,
        aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
    )
    app_state["s3_client_context"] = s3_client_context # Store context for shutdown
    app_state["s3_client"] = await s3_client_context.__aenter__()
    
    # Create Celery App (for sending)
    app_state["celery_app"] = Celery("resume_api", broker=settings.BROKER_URL)
    
    # --- ADD REDIS POOL ---
    print(f"ResumeAPI: Creating Redis pool for {settings.REDIS_HOST}...")
    app_state["redis_pool"] = async_redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    # ----------------------
    
    print("ResumeAPI started successfully.")
    yield
    print("ResumeAPI shutting down...")
    
    # --- ADD REDIS SHUTDOWN ---
    if "redis_pool" in app_state:
        pool = app_state["redis_pool"]
        await pool.disconnect()
        print("ResumeAPI: Redis pool disconnected.")
    # --------------------------

    if "s3_client_context" in app_state:
        await app_state["s3_client_context"].__aexit__(None, None, None)
        print("ResumeAPI: S3 client shut down.")
        
    await async_engine.dispose()
    print("ResumeAPI: DB engine disposed.")

app = FastAPI(title="Resume Ingestion API", lifespan=lifespan)

# Dependencies
async def get_s3_client():
    return app_state["s3_client"]

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
# API Endpoints
# --------------------------------------------------------------------------
class ResumeUploadResponse(BaseModel):
    resume_id: int
    tracking_id: str
    status: str
    batch_id: str

class BatchUploadResponse(BaseModel):
    message: str
    batch_id: str
    file_count: int

# --- ADD NEW RESPONSE MODELS ---
class RichResumeStatusResponse(BaseModel):
    resume_id: int
    batch_id: Optional[uuid.UUID]
    filename: Optional[str]
    current_status: str
    is_embedded: bool
    is_extracted: bool
    history: List[Dict[str, Any]]

class RichBatchStatusResponse(BaseModel):
    batch_id: str
    source: str
    status_details: Optional[Dict[str, Any]] = None
# --- NEW RESPONSE MODEL FOR BATCH LIST ---
class BatchResumeInfo(BaseModel):
    resume_id: int
    status: str
    filename: Optional[str]
# --------------------------------------------------------------------------
# --- NEW: Background Task Logic ---
# --------------------------------------------------------------------------
async def process_resume_batch(
    files: List[UploadFile], 
    generic_schema_id: int, 
    batch_id: uuid.UUID
):
    """
    This function runs in the background.
    It creates its own clients and loops through all files.
    """
    print(f"[Batch {batch_id}]: Background processing started for {len(files)} files.")
    
    # We must create our own clients because we are in a background task
    # 1. Create DB session
    async with AsyncSessionLocal() as db:
    # 2. Create S3 client
        s3_session = aiobotocore.session.get_session()
        s3_client_context = s3_session.create_client(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT_URL,
            aws_access_key_id=settings.MINIO_ROOT_USER,
            aws_secret_access_key=settings.MINIO_ROOT_PASSWORD,
        )
    # 3. Create Celery client
        async with s3_client_context as s3_client:
            celery_app = Celery("resume_api_background", broker=settings.BROKER_URL)
    
            success_count = 0
            fail_count = 0
            # --- PUBLISH "BATCH STARTED" EVENT ---
            # We publish this *inside* the background task
            publish_event(
                celery_app,
                routing_key="resume.batch.started",
                data={
                    "batch_id": str(batch_id),
                    "tasks_to_send": len(files)
                }
            )
            new_resume_id=None
            for file in files:
                tracking_id = uuid.uuid4()
                file_key = f"raw_resumes/{tracking_id}-{file.filename}"
                
                try:
                    # 1. Save to MinIO
                    await s3_client.put_object(
                        Bucket=settings.MINIO_BUCKET_NAME, 
                        Key=file_key, 
                        Body=await file.read()
                    )
                    
                    # 2. Create Resume row
                    new_resume = Resume(
                        tracking_id=tracking_id,
                        batch_id=batch_id, # <-- Link to the batch
                        status='PENDING',
                        filename=file.filename,
                        file_key=file_key,
                        extraction_schema_id=generic_schema_id
                    )
                    db.add(new_resume)
                    await db.commit()
                    await db.refresh(new_resume)
                    new_resume_id = new_resume.resume_id # <-- Get the ID
                    # 3. Send to Ingestion Pipeline
                    await asyncio.to_thread(
                        celery_app.send_task,
                        settings.QUEUE_PARSER, # Kicks off Pipeline 1
                        kwargs={"resume_id": new_resume.resume_id},
                        queue=settings.QUEUE_PARSER
                        
                    )
                    # --- PUBLISH "ITEM" EVENT ---
                    publish_event(
                        celery_app,
                        "resume.parsing.started",
                        {
                            "resume_id": new_resume_id,
                            "batch_id": str(batch_id),
                            "filename": file.filename
                        }
                    )
                    # ----------------------------
                    success_count += 1
                
                except Exception as e:
                    print(f"[Batch {batch_id}]: Failed to process file {file.filename}. Error: {e}")
                    
                    publish_event(
                        celery_app,
                        "resume.upload.failed",
                        {
                            "resume_id": new_resume_id, # Will be None if DB insert failed
                            "batch_id": str(batch_id),
                            "filename": file.filename,
                            "error": str(e)
                        }
                    )
                    fail_count += 1
                finally:
                    await file.close()
            # --- PUBLISH "BATCH FINISHED" EVENT ---
            # This signals the *API's* job is done (sending tasks)
            publish_event(
                celery_app,
                "resume.batch.sent",
                {
                    "batch_id": str(batch_id),
                    "success_count": success_count,
                    "fail_count": fail_count
                }
            )

        print(f"[Batch {batch_id}]: Background processing finished. Success: {success_count}, Failed: {fail_count}")

# --- NEW ENDPOINT ---
@app.post("/api/v1/resumes/upload-multiple", status_code=202)
async def upload_multiple_resumes(
    background_tasks: BackgroundTasks, # <-- FastAPI injects this
    celery_app = Depends(get_celery), # <-- Get the app-level celery
    files: List[UploadFile] = File(...),
    generic_schema_id: int = Form(1)
) -> BatchUploadResponse:
    
    batch_id = uuid.uuid4()
    # --- PUBLISH "BATCH CREATED" EVENT ---
    # Publish this *immediately* so the dashboard shows it
    publish_event(
        celery_app,
        "resume.batch.created",
        {
            "batch_id": str(batch_id),
            "total_resumes": len(files)
        }
    )
    # --- FIRE AND FORGET ---
    # We add our slow, looping function as a background task
    # FastAPI will run this *after* returning the response
    background_tasks.add_task(
        process_resume_batch,
        files=files,
        generic_schema_id=generic_schema_id,
        batch_id=batch_id
    )
    
    # --- RETURN INSTANTLY ---
    return BatchUploadResponse(
        message="Batch upload accepted. Processing in background.",
        batch_id=str(batch_id),
        file_count=len(files)
    )

# --- UPDATED SINGLE UPLOAD ENDPOINT ---
@app.post("/api/v1/resumes/upload", status_code=202)
async def upload_resume(
    file: UploadFile = File(...),
    generic_schema_id: int = Form(1), 
    db: AsyncSession = Depends(get_async_db),
    s3_client = Depends(get_s3_client),
    celery_app = Depends(get_celery)
) -> ResumeUploadResponse:
    
    tracking_id = uuid.uuid4()
    batch_id = uuid.uuid4() # A single file is just a batch of 1
    file_key = f"raw_resumes/{tracking_id}-{file.filename}"
    new_resume = None
    # --- PUBLISH "BATCH CREATED" EVENT ---
    publish_event(
        celery_app,
        "resume.batch.created",
        {
            "batch_id": str(batch_id),
            "total_resumes": 1
        }
    )
    # 1. Save to MinIO
    try:
        await s3_client.put_object(
            Bucket=settings.MINIO_BUCKET_NAME, Key=file_key, Body=await file.read()
        )
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # 2. Create 'PENDING' Resume
    try:
        new_resume = Resume(
            tracking_id=tracking_id,
            batch_id=batch_id, # <-- Set the batch_id
            status='PENDING',
            filename=file.filename,
            file_key=file_key,
            extraction_schema_id=generic_schema_id
        )
        db.add(new_resume)
        await db.commit()
        await db.refresh(new_resume)
    except Exception as e:
        # --- PUBLISH "ITEM FAILED" EVENT ---
        publish_event(
            celery_app,
            "resume.upload.failed",
            {
                "resume_id": None,
                "batch_id": str(batch_id),
                "filename": file.filename,
                "error": str(e)
            }
        )
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    # 3. Send to *Ingestion Pipeline*
    await asyncio.to_thread(
        celery_app.send_task,
        settings.QUEUE_PARSER,
        kwargs={"resume_id": new_resume.resume_id},
        queue=settings.QUEUE_PARSER
    )
    # --- PUBLISH "ITEM" EVENT ---
    publish_event(
        celery_app,
        "resume.parsing.started",
        {
            "resume_id": new_resume.resume_id,
            "batch_id": str(batch_id),
            "filename": new_resume.filename
        }
    )

    return ResumeUploadResponse(
        resume_id=new_resume.resume_id,
        tracking_id=str(new_resume.tracking_id),
        status=new_resume.status,
        batch_id=str(batch_id)
    )

@app.get("/api/v1/resumes/status/{resume_id}")
async def get_resume_ingestion_status(resume_id: int, db: AsyncSession = Depends(get_async_db)):
    resume = await db.get(Resume, resume_id)
    if not resume:
        raise HTTPException(404, "Resume not found")
    return {"resume_id": resume.resume_id, "status": resume.status}
@app.get("/api/v1/resumes/dashboard-status/batch/{batch_id}", response_model=RichBatchStatusResponse)
async def get_dashboard_batch_status(
    batch_id: str, 
    r: async_redis.Redis = Depends(get_redis)
):
    """
    Gets the real-time, aggregated counters for a *batch* from Redis.
    This is for the main dashboard view.
    """
    key = f"batch:{batch_id}"
    status_details = await r.hgetall(key)
    
    if not status_details:
        raise HTTPException(
            status_code=404, 
            detail="Batch status not found in dashboard cache. The batch may not exist or data has expired."
        )
        
    return RichBatchStatusResponse(
        batch_id=batch_id,
        source="redis_dashboard_cache",
        status_details=status_details
    )

@app.get("/api/v1/resumes/dashboard-status/resume/{resume_id}", response_model=RichResumeStatusResponse)
async def get_dashboard_resume_status(
    resume_id: int, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    Gets the rich, detailed status and history for a *single resume* from Postgres.
    This is for the "drill-down" dashboard view.
    """
    # 1. Get the resume and its logs in one query
    query = (
        select(Resume)
        .options(joinedload(Resume.logs)) # <-- This loads all related logs
        .filter(Resume.resume_id == resume_id)
    )
    result = await db.execute(query)
    resume = result.unique().scalar_one_or_none()

    if not resume:
        raise HTTPException(404, "Resume not found")
        
    # 2. Format the logs for the dashboard
    logs = [
        {
            "timestamp": log.timestamp.isoformat(),
            "worker_name": log.worker_name,
            "status": log.status,
            "message": log.message
        } 
        for log in sorted(resume.logs, key=lambda log: log.timestamp) # Sort by time
    ]

    # 3. Return the rich status
    return RichResumeStatusResponse(
        resume_id=resume.resume_id,
        batch_id=resume.batch_id,
        filename=resume.filename,
        current_status=resume.status,
        is_embedded=resume.is_embedded,
        is_extracted=resume.is_extracted,
        history=logs
    )
# --- THIS IS THE NEW ENDPOINT YOU NEED ---
@app.get("/api/v1/resumes/batch/{batch_id}", response_model=List[BatchResumeInfo])
async def get_batch_resume_list(
    batch_id: str, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    Gets the list of all resume_ids and their final status for a given batch.
    This is how you get the resume_ids to use in other endpoints.
    """
    try:
        batch_uuid = uuid.UUID(batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid batch_id format. Must be a UUID.")

    query = (
        select(Resume)
        .filter(Resume.batch_id == batch_uuid)
        .order_by(Resume.resume_id)
    )
    result = await db.execute(query)
    resumes = result.scalars().all()

    if not resumes:
        raise HTTPException(
            status_code=404,
            detail="Batch ID not found or no resumes in this batch."
        )

    return [
        BatchResumeInfo(
            resume_id=resume.resume_id,
            status=resume.status,
            filename=resume.filename
        ) for resume in resumes
    ]