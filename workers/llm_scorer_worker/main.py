import asyncio
import instructor
from openai import AsyncOpenAI # <-- Back to Async
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.future import select
from pydantic import BaseModel, Field, create_model

from settings import settings
# --- IMPORT NEW MODEL ---
from schemas.models import (
    Resume, Job, Score, ScoreLog,
    LlmScore, SemanticScore, ExtractionSchema, ScoringRubric, Base
)


# --------------------------------------------------------------------------
# Dynamic Pydantic Model Generator (No changes)
# --------------------------------------------------------------------------
class ScoreWithEvidence(BaseModel):
    score: float = Field(..., description="The numeric score from 0.0 to 1.0 for this specific criterion.")
    rationale: str = Field(..., description="A 1-2 sentence rationale explaining *why* you gave this score.")
    evidence: str = Field(..., description="A single, short, direct quote from the *FULL RESUME TEXT* (not the profile) that best supports this score.")
def create_dynamic_rubric_model(schema_name: str, rubric_criteria: dict) -> BaseModel:
    fields_to_create = {}
    for field_name, description in rubric_criteria.items():
        fields_to_create[field_name] = (ScoreWithEvidence, Field(
            description=f"{description}. Score must be a float between 0.0 and 1.0."
        ))
    DynamicModel = create_model(schema_name, __base__=BaseModel, **fields_to_create)
    return DynamicModel

# --------------------------------------------------------------------------
# Celery App Configuration (No changes)
# --------------------------------------------------------------------------

app = Celery(
    "llm_scorer_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)
app.conf.task_queues = {
    settings.QUEUE_LLM_SCORER: {
        "exchange": settings.QUEUE_LLM_SCORER,
        "routing_key": settings.QUEUE_LLM_SCORER,
    }
}
app.conf.task_default_queue = settings.QUEUE_LLM_SCORER

# --------------------------------------------------------------------------
# Synchronous Resource Management
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    openai_client = None

resources: AsyncResources = None
def setup_resources(res: AsyncResources):
    """Synchronously create async-capable resources."""
    print("LLMScorer: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    res.openai_client = instructor.patch(
        AsyncOpenAI(api_key=settings.OPENAI_API_KEY), # <-- AsyncOpenAI
        mode=instructor.Mode.TOOLS
    )
    print("LLMScorer: Async resources initialized.")
async def shutdown_async_resources(res: AsyncResources):
    """Asynchronously dispose of resources."""
    print("LLMScorer: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("LLMScorer: Async resources shut down.")
@worker_process_init.connect
def on_worker_init(**kwargs):
    global resources
    resources = AsyncResources()
    setup_resources(resources)
@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    global resources
    if resources:
        asyncio.run(shutdown_async_resources(resources))
    resources = None


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for LLMScorerWorker.")
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The (Synchronous) LLM Scorer Task
# --------------------------------------------------------------------------

# --- CHANGED ---
# This is no longer async. It's the main logic function.
async def async_score_llm(score_id: int):
    print(f"[Score {score_id}]: Async LLM score task started.")
    
    try:
        job_description = None
        profile_json = None
        DynamicScoringModel = None
        
        # --- SESSION 1: Get data (No lock) ---
        async with get_db_session() as db:
            # 1. Get Score, Resume, Job. NO LOCK NEEDED.
            score_obj = await db.get(Score, score_id, options=[
                joinedload(Score.resume, innerjoin=True),
                joinedload(Score.job, innerjoin=True).joinedload(Job.scoring_rubric)
            ])

            if not (score_obj and score_obj.resume and score_obj.resume.profile_json and
                    score_obj.job and score_obj.job.scoring_rubric):
                   print(f"[Score {score_id}]: ERROR - Missing related data. Skipping.")
                   return
            
            job = score_obj.job
            profile = score_obj.resume.profile_json
            criteria = job.scoring_rubric.rubric_definition.get("criteria")
            if not criteria:
                raise Exception(f"Rubric {job.scoring_rubric.rubric_name} has no 'criteria' defined.")

            # 2. Store data needed for API call
            job_description = job.description
            profile_json = profile
            DynamicScoringModel = create_dynamic_rubric_model(job.scoring_rubric.rubric_name, criteria)

        # --- OPENAI LOGIC (Outside session) ---
        # 3. Call OpenAI (asynchronously)
        llm_scores: BaseModel = await resources.openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL_NAME,
            response_model=DynamicScoringModel,
            messages=[
                {"role": "system", "content": "You are an expert HR analyst..."},
                {"role": "user", "content": f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE'S EXTRACTED PROFILE:\n{profile_json}"}
            ]
        )
        
        # --- SESSION 2: Write result to its own table ---
        async with get_db_session() as db:
            # 4. Create the new LlmScore object
            new_score = LlmScore(
                score_id=score_id,
                scores_json=llm_scores.model_dump()
            )
            db.add(new_score)
            await db.commit()
            print(f"[Score {score_id}]: Successfully saved LLM scores.")

        # 5. Publish task for the *aggregator*
        print(f"[Score {score_id}]: Sending task to aggregator...")
        await asyncio.to_thread(
            app.send_task,
            settings.QUEUE_AGGREGATOR,
            kwargs={"score_id": score_id},
            queue=settings.QUEUE_AGGREGATOR
        )

    except Exception as e:
        print(f"[Score {score_id}]: --- TASK FAILED --- Error: {e}")
        try:
            async with get_db_session() as error_db:
                score_fail = await error_db.get(Score, score_id)
                if score_fail:
                    score_fail.status = 'LLM_SCORE_FAILED'
                    await error_db.commit()
        except Exception as db_e:
            print(f"CRITICAL: Failed to write failure status to DB: {db_e}")
        raise e


# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_LLM_SCORER, bind=True)
def score_llm(self, score_id: int):
    try:
        if not resources:
            raise Exception("LLMScorer resources not initialized.")
        asyncio.run(async_score_llm(score_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        raise