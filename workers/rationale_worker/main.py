import asyncio
import instructor
from openai import AsyncOpenAI
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Dict, Any

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.future import select
from pydantic import BaseModel, Field

# Import our settings
from settings import settings

# Import Shared Models from libs
from schemas.models import Resume, Job, Score, ScoringRubric

# --------------------------------------------------------------------------
# Pydantic Model for Instructor (The *output* of this worker)
# --------------------------------------------------------------------------

class FinalReport(BaseModel):
    rationale: str = Field(..., description="A 2-3 sentence, professional rationale explaining the final score, based on the sub-scores and weights.")
    evidence_spans: List[str] = Field(..., description="1-3 direct, short quotes from the *full* resume text that strongly support the sub-scores.")

# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------

app = Celery(
    "rationale_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)

app.conf.task_queues = {
    settings.QUEUE_RATIONALE: {
        "exchange": settings.QUEUE_RATIONALE,
        "routing_key": settings.QUEUE_RATIONALE,
    }
}
app.conf.task_default_queue = settings.QUEUE_RATIONALE

# --------------------------------------------------------------------------
# Async Resource Management
# --------------------------------------------------------------------------

class AsyncResources:
    db_engine = None
    db_sessionmaker = None
    openai_client = None

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
    print("RationaleWorker: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    res.openai_client = instructor.patch(
        AsyncOpenAI(api_key=settings.OPENAI_API_KEY),
        mode=instructor.Mode.TOOLS
    )
    print("RationaleWorker: Async resources initialized.")

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
    print("RationaleWorker: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("RationaleWorker: Async resources shut down.")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    if not resources or not resources.db_sessionmaker:
        raise Exception("Async resources not initialized for RationaleWorker.")
        
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The Rationale Task
# --------------------------------------------------------------------------

async def async_generate_rationale(score_id: int):
    """
    This is our *actual* async logic.
    """
    print(f"[Score {score_id}]: Async rationale task started.")
    
    async with get_db_session() as db:
        try:
            # 1. Get ALL data: Score, Resume (for text), Job (for rubric)
            query = (
                select(Score)
                .options(
                    joinedload(Score.resume), # Need resume for parsed_text
                    joinedload(Score.job).joinedload(Job.scoring_rubric) # Need job for rubric
                )
                .filter(Score.score_id == score_id)
            )
            result = await db.execute(query)
            score_obj = result.scalar_one_or_none()

            # Check for all required data points
            if not (score_obj and score_obj.job and score_obj.job.scoring_rubric and
                    score_obj.resume and score_obj.resume.parsed_text and
                    score_obj.sub_scores_json and score_obj.overall_score is not None):
                 print(f"[Score {score_id}]: ERROR - Data missing for rationale generation.")
                 score_obj.status = 'RATIONALE_FAILED'
                 await db.commit()
                 return
            
            job = score_obj.job
            resume = score_obj.resume
            
            # 2. Update status
            score_obj.status = 'GENERATING_RATIONALE'
            await db.commit()

            # 3. Build the master prompt
            prompt = f"""
            You are an expert HR Manager. A candidate has been scored.
            Your job is to provide a final, human-readable report.
            DO NOT recalculate anything. Just explain the results.

            --- CONTEXT ---
            Job Description: {job.description}
            Final Score (0-100): {score_obj.overall_score}
            Calculated Sub-Scores: {score_obj.sub_scores_json}
            Rubric Weights Used: {job.scoring_rubric.rubric_definition.get('weights', {})}
            Candidate's Extracted Profile: {resume.profile_json}

            --- FULL RESUME TEXT (for evidence) ---
            {resume.parsed_text}
            ---
            
            Based on all of
            the data above, generate the final report.
            The rationale must explain *why* the score is high or low.
            The evidence_spans MUST be exact quotes from the "FULL RESUME TEXT".
            """
            
            # 4. Call OpenAI with Instructor
            report: FinalReport = await resources.openai_client.chat.completions.create(
                model=settings.OPENAI_MODEL_NAME,
                response_model=FinalReport,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Generate the final rationale and find the evidence spans."}
                ]
            )
            
            # 5. Save the final report to the Score row
            score_obj.rationale = report.rationale
            score_obj.evidence_json = report.evidence_spans
            score_obj.status = 'COMPLETE' # <-- The end of the line
            
            await db.commit()
            print(f"[Score {score_id}]: --- SCORING PIPELINE COMPLETE ---")

        except Exception as e:
            print(f"[Score {score_id}]: --- TASK FAILED --- Error: {e}")
            if 'db' in locals() and 'score_obj' in locals() and score_obj:
                try:
                    await db.rollback()
                    score_obj.status = 'RATIONALE_FAILED'
                    await db.commit()
                except Exception as db_e:
                    print(f"CRITICAL: Failed to write failure status to DB: {db_e}")

# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_RATIONALE, bind=True)
def generate_rationale(self, score_id: int):
    """
    This is the SYNCHRONOUS Celery task that runs our async logic.
    """
    try:
        asyncio.run(async_generate_rationale(score_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        # raise self.retry(exc=e, countdown=60)