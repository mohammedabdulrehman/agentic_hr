import asyncio
import uuid
import redis.asyncio as async_redis
import redis
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends,Query
from pydantic import BaseModel
from celery import Celery
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload

# --- UPDATED IMPORTS ---
from libs.monitoring import publish_event
# Import all models to ensure relationships are built
from schemas.models import (
    Resume, Job, Score, ScoreLog,
    LlmScore, SemanticScore, ExtractionSchema, ScoringRubric, Base
)
# -------------------------
from settings import settings

# --------------------------------------------------------------------------
# Database Setup
# --------------------------------------------------------------------------
# This API does NOT create tables, it just connects.
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

# --------------------------------------------------------------------------
# --- UPDATED: App State & Lifecycle (Connects to Celery & Redis) ---
# --------------------------------------------------------------------------
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("MatchAPI starting up...")
    
    # 1. Create Celery App (for sending)
    app_state["celery_app"] = Celery("match_api", broker=settings.BROKER_URL)
    
    # 2. --- ADD REDIS POOL ---
    print(f"MatchAPI: Creating Redis pool for {settings.REDIS_HOST}...")
    app_state["redis_pool"] = async_redis.ConnectionPool(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True
    )
    # ----------------------
    
    print("MatchAPI started successfully.")
    yield
    print("MatchAPI shutting down...")
    
    # --- ADD REDIS SHUTDOWN ---
    if "redis_pool" in app_state:
        pool = app_state["redis_pool"]
        await pool.disconnect()
        print("MatchAPI: Redis pool disconnected.")
    # --------------------------
    
    await async_engine.dispose()
    print("MatchAPI: DB engine disposed.")

app = FastAPI(title="Matching API", lifespan=lifespan)

# --- UPDATED: Dependencies ---
def get_celery():
    return app_state["celery_app"]

async def get_redis() -> AsyncGenerator[async_redis.Redis, None]:
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
# ------------------------------

# --------------------------------------------------------------------------
# API Models
# --------------------------------------------------------------------------
class MatchRequest(BaseModel):
    resume_id: int
    job_id: int

class MatchResponse(BaseModel):
    score_id: int
    resume_id: int
    job_id: int
    status: str
    batch_id: str # <-- Send batch_id back

class BatchMatchRequest(BaseModel):
    job_id: int
    resume_ids: List[int]

class BatchMatchResponse(BaseModel):
    batch_id: str
    job_id: int
    tasks_created: int
    failed_resumes: List[Dict[str, Any]]

# --- ADD NEW RESPONSE MODELS ---
class RichScoreStatusResponse(BaseModel):
    score_id: int
    batch_id: Optional[uuid.UUID]
    resume_id: int
    job_id: int
    current_status: str
    overall_score: Optional[int]
    history: List[Dict[str, Any]]

class RichBatchStatusResponse(BaseModel):
    batch_id: str
    source: str
    status_details: Optional[Dict[str, Any]] = None
class FinalReportResponse(BaseModel):
    score_id: int
    resume_id: int
    job_id: int
    status: str
    name: Optional[str] = None
    contact: Optional[str] = None
    overallScore: Optional[float] = None
    subscores: Dict[str, float]
    explanations: List[Dict[str, Any]]
    modelVersion: str
    timestamp: Optional[str] = None
class BatchMatchByBatchIDRequest(BaseModel):
    job_id: int
    resume_batch_id: str # The batch_id from the resume upload
# --------------------------------------------------------------------------
# Background Task Logic (Moved from resume_api)
# --------------------------------------------------------------------------

# ------------------------
# --------------------------------

# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------

@app.post("/api/v1/match", status_code=202)
async def create_match(
    request: MatchRequest,
    db: AsyncSession = Depends(get_async_db),
    celery_app = Depends(get_celery)
) -> MatchResponse:
    
    # 1. Check if the resume is ready
    resume = await db.get(Resume, request.resume_id)
    if not resume:
        raise HTTPException(404, f"Resume ID {request.resume_id} not found.")
    if resume.status != 'READY_TO_SCORE':
        raise HTTPException(400, f"Resume not ready. Status: {resume.status}")
    
    # 2. Check if the job exists
    job = await db.get(Job, request.job_id)
    if not job:
        raise HTTPException(404, f"Job ID {request.job_id} not found.")

    # 3. Create a new 'Score' row to track this job
    batch_id = uuid.uuid4() # Create a new, unique batch_id for this single match
    new_score = Score(
        resume_id=request.resume_id,
        job_id=request.job_id,
        status='PENDING',
        batch_id=batch_id # <-- Link to the new batch_id
    )
    db.add(new_score)
    await db.commit()
    await db.refresh(new_score)

    # 4. --- PUBLISH "BATCH CREATED" EVENT ---
    publish_event(
        celery_app,
        "match.batch.created",
        {
            "batch_id": str(batch_id),
            "job_id": new_score.job_id,
            "total_resumes": 1
        }
    )

    # 5. Send to the *Scoring Pipeline*
    await asyncio.to_thread(
        celery_app.send_task,
        settings.QUEUE_SCORING_ENGINE,
        kwargs={"score_id": new_score.score_id},
        queue=settings.QUEUE_SCORING_ENGINE
    )
    
    # 6. --- PUBLISH "SCORING STARTED" EVENT ---
    publish_event(
        celery_app,
        "score.scoring.started",
        {
            "score_id": new_score.score_id,
            "batch_id": str(batch_id)
        }
    )
    
    return MatchResponse(
        score_id=new_score.score_id,
        resume_id=new_score.resume_id,
        job_id=new_score.job_id,
        status=new_score.status,
        batch_id=str(batch_id)
    )

@app.post("/api/v1/match/batch", status_code=202)
async def create_batch_match(
    request: BatchMatchRequest,
    db: AsyncSession = Depends(get_async_db),
    celery_app = Depends(get_celery)
) -> BatchMatchResponse:
    
    # 1. Check if the job exists
    job = await db.get(Job, request.job_id)
    if not job:
        raise HTTPException(404, f"Job ID {request.job_id} not found.")

    # 2. Efficiently fetch all requested resumes
    query = select(Resume).where(Resume.resume_id.in_(request.resume_ids))
    result = await db.execute(query)
    resumes_map = {r.resume_id: r for r in result.scalars()}

    new_scores_to_create = []
    failed_resumes_response = []
    
    # --- 3. CREATE A NEW BATCH ID FOR THIS OPERATION ---
    batch_id = uuid.uuid4()

    # 4. Loop through requested IDs and validate
    for resume_id in request.resume_ids:
        resume = resumes_map.get(resume_id)
        
        if not resume:
            failed_resumes_response.append({"resume_id": resume_id, "reason": "Resume ID not found."})
        elif resume.status != 'READY_TO_SCORE':
            failed_resumes_response.append({"resume_id": resume_id, "reason": f"Resume not ready. Status: {resume.status}"})
        else:
            new_scores_to_create.append(Score(
                resume_id=resume_id,
                job_id=request.job_id,
                status='PENDING',
                batch_id=batch_id # <-- Link all scores to this NEW batch_id
            ))

    if not new_scores_to_create:
        raise HTTPException(400, "No valid resumes to match.")

    # 5. --- PUBLISH "BATCH CREATED" EVENT ---
    publish_event(
        celery_app,
        "match.batch.created",
        {
            "batch_id": str(batch_id),
            "job_id": request.job_id,
            "total_resumes": len(new_scores_to_create)
        }
    )

    # 6. Create all new Score rows in one transaction
    db.add_all(new_scores_to_create)
    await db.commit()
    print(f"Created {len(new_scores_to_create)} new score rows for batch {batch_id}.")

    # 7. Fire off all Celery tasks in parallel
    celery_tasks = []
    for score in new_scores_to_create:
        await db.refresh(score) # Get the new score_id
        
        # --- PUBLISH "SCORING STARTED" EVENT (for each score) ---
        publish_event(
            celery_app,
            "score.scoring.started",
            {
                "score_id": score.score_id,
                "batch_id": str(batch_id)
            }
        )
        
        celery_tasks.append(
            asyncio.to_thread(
                celery_app.send_task,
                settings.QUEUE_SCORING_ENGINE,
                kwargs={"score_id": score.score_id},
                queue=settings.QUEUE_SCORING_ENGINE
            )
        )
    
    await asyncio.gather(*celery_tasks)
    print(f"Sent {len(celery_tasks)} tasks to the scoring engine.")

    # 8. Return the response
    return BatchMatchResponse(
        batch_id=str(batch_id),
        job_id=request.job_id,
        tasks_created=len(celery_tasks),
        failed_resumes=failed_resumes_response
    )
# --- THIS IS THE NEW ENDPOINT YOU ASKED FOR ---
@app.post("/api/v1/match/by-batch", status_code=202)
async def create_batch_match_by_batch_id(
    request: BatchMatchByBatchIDRequest,
    db: AsyncSession = Depends(get_async_db),
    celery_app = Depends(get_celery)
) -> BatchMatchResponse:
    """
    Creates a new match job by finding all 'READY_TO_SCORE' resumes
    from a *previous* resume upload batch.
    """
    
    # 1. Validate the Job ID
    job = await db.get(Job, request.job_id)
    if not job:
        raise HTTPException(404, f"Job ID {request.job_id} not found.")

    # 2. Validate the Resume Batch ID
    try:
        resume_batch_uuid = uuid.UUID(request.resume_batch_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resume_batch_id format. Must be a UUID.")
        
    # 3. Find all resumes in that batch that are READY
    query = (
        select(Resume)
        .filter(Resume.batch_id == resume_batch_uuid)
        .filter(Resume.status == 'READY_TO_SCORE')
    )
    result = await db.execute(query)
    ready_resumes = result.scalars().all()
    
    if not ready_resumes:
        raise HTTPException(
            status_code=404, 
            detail="No resumes in this batch are 'READY_TO_SCORE'. Please wait for ingestion to complete."
        )

    # 4. Create a *new* batch_id for this *matching* operation
    new_match_batch_id = uuid.uuid4()
    new_scores_to_create = []
    
    for resume in ready_resumes:
        new_scores_to_create.append(Score(
            resume_id=resume.resume_id,
            job_id=request.job_id,
            status='PENDING',
            batch_id=new_match_batch_id # Link to the new match batch
        ))

    # 5. Publish the "match.batch.created" event
    publish_event(
        celery_app,
        "match.batch.created",
        { "batch_id": str(new_match_batch_id), "job_id": request.job_id, "total_resumes": len(new_scores_to_create) }
    )

    # 6. Save all new Score objects
    db.add_all(new_scores_to_create)
    await db.commit()
    print(f"Created {len(new_scores_to_create)} new score rows for match batch {new_match_batch_id}.")

    # 7. Fire off all Celery tasks in parallel
    celery_tasks = []
    for score in new_scores_to_create:
        await db.refresh(score)
        
        publish_event(
            celery_app, "score.scoring.started",
            { "score_id": score.score_id, "batch_id": str(new_match_batch_id) }
        )
        
        celery_tasks.append(
            asyncio.to_thread(
                celery_app.send_task,
                settings.QUEUE_SCORING_ENGINE,
                kwargs={"score_id": score.score_id},
                queue=settings.QUEUE_SCORING_ENGINE
            )
        )
    await asyncio.gather(*celery_tasks)
    
    # 8. Return the new batch info
    return BatchMatchResponse(
        batch_id=str(new_match_batch_id),
        job_id=request.job_id,
        tasks_created=len(celery_tasks),
        failed_resumes=[] # We only processed ready resumes
    )
# --------------------------------------------------------------------------
# --- UPDATED and NEW DASHBOARD STATUS ENDPOINTS ---
# --------------------------------------------------------------------------

@app.get("/api/v1/match/status/{score_id}",response_model=FinalReportResponse)
async def get_match_status(score_id: int, db: AsyncSession = Depends(get_async_db)):
    """
    (Original Endpoint, now FIXED)
    Gets the rich, fully-computed score data from Postgres.
    This is the "final report" for a single match.
    """
    
    # --- THIS IS THE FIX ---
    # We must load the score and its related sub-score tables
    query = (
        select(Score)
        .options(
            joinedload(Score.llm_score),
            joinedload(Score.semantic_score)
        )
        .filter(Score.score_id == score_id)
    )
    result = await db.execute(query)
    score = result.unique().scalar_one_or_none() # <-- Must use .unique()
    # --- END FIX ---
    
    if not score:
        raise HTTPException(404, "Score ID not found")
    
    # --- 2. Build the response format (Your logic was correct) ---
    subscores = {}
    explanations = []
    
    if score.llm_score and score.llm_score.scores_json:
        for criterion_name, data in score.llm_score.scores_json.items():
            if isinstance(data, dict):
                subscores[criterion_name] = data.get('score')
                explanations.append({
                    "criterion": criterion_name,
                    "evidence": [data.get('evidence', 'No evidence found')]
                })

    if score.semantic_score:
        subscores["semantic_similarity"] = score.semantic_score.score_value
    
    return FinalReportResponse(
        score_id=score.score_id,
        resume_id=score.resume_id,
        job_id=score.job_id,
        status=score.status,
        # Convert 0-100 int score to 0.0-1.0 float
        overallScore=(score.overall_score / 100.0) if score.overall_score is not None else None,
        subscores=subscores,
        explanations=explanations,
        # This is the main rationale from the Score table.
        # It will be 'null' until you add a rationale_worker.
        modelVersion="v1.0.0", 
        timestamp=score.updated_at.isoformat() if score.updated_at else None
    )
    
    return FinalReportResponse

@app.get("/api/v1/match/dashboard-status/batch/{batch_id}", response_model=RichBatchStatusResponse)
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

@app.get("/api/v1/match/dashboard-status/score/{score_id}", response_model=RichScoreStatusResponse)
async def get_dashboard_score_status(
    score_id: int, 
    db: AsyncSession = Depends(get_async_db)
):
    """
    Gets the rich, detailed status and history for a *single score* from Postgres.
    This is for the "drill-down" dashboard view.
    """
    # 1. Get the score and its logs in one query
    query = (
        select(Score)
        .options(joinedload(Score.logs)) # <-- This loads all related logs
        .filter(Score.score_id == score_id)
    )
    result = await db.execute(query)
    score = result.unique().scalar_one_or_none() # <-- Must use .unique()

    if not score:
        raise HTTPException(404, "Score not found")
        
    # 2. Format the logs for the dashboard
    logs = [
        {
            "timestamp": log.timestamp.isoformat(),
            "worker_name": log.worker_name,
            "status": log.status,
            "message": log.message
        } 
        for log in sorted(score.logs, key=lambda log: log.timestamp) # Sort by time
    ]

    # 3. Return the rich status
    return RichScoreStatusResponse(
        score_id=score.score_id,
        batch_id=score.batch_id,
        resume_id=score.resume_id,
        job_id=score.job_id,
        current_status=score.status,
        overall_score=score.overall_score,
        history=logs
    )
# --- THIS IS THE NEW ENDPOINT YOU ASKED FOR ---
@app.get("/api/v1/match/reports/batch/{batch_id}", response_model=List[FinalReportResponse])
async def get_batch_final_reports(
    batch_id: str, 
    db: AsyncSession = Depends(get_async_db),
    # Add pagination:
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100) # Default 20, max 100
):
    """
    Gets a paginated list of all *completed* final reports for a batch.
    This is the heavy-lifting endpoint for your dashboard to show results.
    """
    
    # 1. Query for all scores in the batch that are COMPLETE
    query = (
        select(Score)
        .options(
            joinedload(Score.llm_score),
            joinedload(Score.semantic_score),
            joinedload(Score.resume) # <-- CRITICAL: Load the resume to access profile_json
        )
        .filter(Score.batch_id == uuid.UUID(batch_id)) # Filter by batch
        .filter(Score.status == 'COMPLETE')         # Only get finished reports
        .order_by(Score.overall_score.desc())       # Order by best score
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(query)
    scores = result.unique().scalars().all() # .unique() is critical

    # 2. Transform the data into the FinalReportResponse
    response_list = []
    for score in scores:
        subscores = {}
        explanations = []
        
        if score.llm_score and score.llm_score.scores_json:
            for criterion_name, data in score.llm_score.scores_json.items():
                if isinstance(data, dict):
                    subscores[criterion_name] = data.get('score')
                    explanations.append({
                        "criterion": criterion_name,
                        "evidence": [data.get('evidence', 'No evidence found')],
                        "rationale":data.get('rationale',"No rationale found")
                    })

        if score.semantic_score:
            subscores["semantic_similarity"] = round(score.semantic_score.score_value, 2)
        # --- NEW: Extract Personal Info from Resume Profile ---
        # We handle cases where profile_json might be None or keys might differ
        profile = score.resume.profile_json or {}
        candidate_name = profile.get("name") or profile.get("full_name") or "Unknown"
        candidate_contact = profile.get("contact")
        
        report = FinalReportResponse(
            score_id=score.score_id,
            resume_id=score.resume_id,
            job_id=score.job_id,
            status=score.status,
            name=candidate_name,
            contact=candidate_contact,
            overallScore=(score.overall_score / 100.0) if score.overall_score is not None else None,
            subscores=subscores,
            explanations=explanations,
            modelVersion="v1.0.0", 
            timestamp=score.updated_at.isoformat() if score.updated_at else None
        )
        response_list.append(report)
        
    return response_list