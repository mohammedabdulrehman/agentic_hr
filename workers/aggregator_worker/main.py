import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import joinedload
from sqlalchemy.future import select

from settings import settings

# --- UPDATED IMPORTS ---
from libs.monitoring import publish_event
from schemas.models import Resume, Job, Score, ScoringRubric, SemanticScore, LlmScore, ScoreLog
# -------------------------

# --------------------------------------------------------------------------
# Celery App Configuration
# --------------------------------------------------------------------------
app = Celery(
    "aggregator_worker",
    broker=settings.BROKER_URL,
    backend="rpc://"
)
app.conf.task_queues = {
    settings.QUEUE_AGGREGATOR: {
        "exchange": settings.QUEUE_AGGREGATOR,
        "routing_key": settings.QUEUE_AGGREGATOR,
    }
}
app.conf.task_default_queue = settings.QUEUE_AGGREGATOR

# --------------------------------------------------------------------------
# Async Resource Management (CORRECTED PATTERN)
# --------------------------------------------------------------------------
class AsyncResources:
    db_engine = None
    db_sessionmaker = None

resources: AsyncResources = None

def setup_resources(res: AsyncResources):
    """Synchronously create async-capable resources."""
    print("AggregatorWorker: Initializing async resources...")
    res.db_engine = create_async_engine(settings.DATABASE_URL_PSYCOPG, pool_pre_ping=True)
    res.db_sessionmaker = async_sessionmaker(bind=res.db_engine, expire_on_commit=False)
    print("AggregatorWorker: Async resources initialized.")

async def shutdown_async_resources(res: AsyncResources):
    """Asynchronously dispose of resources."""
    print("AggregatorWorker: Shutting down async resources...")
    if res.db_engine:
        await res.db_engine.dispose()
    print("AggregatorWorker: Async resources shut down.")

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
        raise Exception("Async resources not initialized for AggregatorWorker.")
    async with resources.db_sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# --------------------------------------------------------------------------
# The Aggregator Task (UPDATED WITH MONITORING)
# --------------------------------------------------------------------------
async def async_aggregate_scores(score_id: int):
    print(f"[Score {score_id}]: Async aggregator task started.")
    
    batch_id = None # For monitoring
    
    async with get_db_session() as db:
        try:
            # 1. --- THE CHECK ---
            sem_score = await db.get(SemanticScore, score_id)
            llm_score_obj = await db.get(LlmScore, score_id)

            if not (sem_score and llm_score_obj):
                # We are still waiting for the other worker.
                print(f"[Score {score_id}]: Still waiting for other scores. Doing nothing.")
                
                # --- PUBLISH "WAITING" EVENT ---
                # Get batch_id for the event
                score_obj_check = await db.get(Score, score_id)
                if score_obj_check:
                    batch_id = str(score_obj_check.batch_id) if score_obj_check.batch_id else None
                
                publish_event(
                    app,
                    "score.aggregator.waiting",
                    {"score_id": score_id, "batch_id": batch_id, "has_semantic": bool(sem_score), "has_llm": bool(llm_score_obj)}
                )
                return # Exit successfully.

            # 2. --- BOTH SCORES ARE PRESENT! ---
            print(f"[Score {score_id}]: All scores present! Calculating weighted score.")
            
            # 3. Get the main Score object and Rubric
            score_obj = await db.get(Score, score_id, options=[
                joinedload(Score.job, innerjoin=True).joinedload(Job.scoring_rubric)
            ])
            if not (score_obj and score_obj.job and score_obj.job.scoring_rubric):
                raise Exception("Critical: Score or Job/Rubric missing after aggregation check.")

            batch_id = str(score_obj.batch_id) if score_obj.batch_id else None

            # Idempotency check
            if score_obj.status == 'COMPLETE':
                print(f"[Score {score_id}]: Already aggregated. Ignoring.")
                return

            # --- ADD "STARTED" LOG ---
            db.add(ScoreLog(
                score_id=score_id,
                batch_id=batch_id,
                worker_name="aggregator_worker",
                status="STARTED",
                message="Aggregating final score."
            ))
            await db.flush() # Flush to log start before processing

            # --- PUBLISH "STARTED" EVENT ---
            publish_event(
                app,
                "score.aggregator.started",
                {"score_id": score_id, "batch_id": batch_id}
            )

            # 4. --- CALCULATE THE WEIGHTED SCORE ---
            rubric_def = score_obj.job.scoring_rubric.rubric_definition
            required_weights = rubric_def.get("weights")
            if not required_weights:
                raise Exception(f"Rubric {score_obj.job.scoring_rubric.rubric_name} has no 'weights' defined.")
            
            llm_scores_json = llm_score_obj.scores_json
            
            extracted_numeric_scores = {
                key: value.get('score', 0.0)
                for key, value in llm_scores_json.items()
                if isinstance(value, dict)
            }

            all_scores = {
                "semantic_similarity": sem_score.score_value,
                **extracted_numeric_scores
            }

            overall_score = 0.0
            for key, weight in required_weights.items():
                if key not in all_scores:
                    print(f"[Score {score_id}]: WARNING - Required key '{key}' not in scores. Skipping.")
                    continue
                score = float(all_scores.get(key, 0.0))
                overall_score += score * weight
            
            final_weighted_score = round(overall_score * 100)
            print(f"[Score {score_id}]: Final weighted score: {final_weighted_score}")
            
            # 5. Update status and save the final score
            score_obj.overall_score = final_weighted_score
            score_obj.status = 'COMPLETE' 
            
            # --- ADD "COMPLETED" LOG ---
            db.add(ScoreLog(
                score_id=score_id,
                batch_id=batch_id,
                worker_name="aggregator_worker",
                status="COMPLETED",
                message=f"Aggregation complete. Final score: {final_weighted_score}"
            ))
            
            await db.commit() # Commit the final score and log
            
            # 6. --- PUBLISH "COMPLETED" EVENT ---
            publish_event(
                app,
                "score.aggregator.completed",
                {"score_id": score_id, "batch_id": batch_id, "final_score": final_weighted_score}
            )
            
            print(f"[Score {score_id}]: Pipeline complete.")

        except Exception as e:
            error_message = str(e)
            print(f"[Score {score_id}]: --- TASK FAILED --- Error: {error_message}")
            
            try:
                # Use a new session to write the failure status
                async with get_db_session() as error_db:
                    score_fail = await error_db.get(Score, score_id)
                    if score_fail and score_fail.status != 'COMPLETE':
                        score_fail.status = 'AGGREGATE_FAILED'
                        batch_id = str(score_fail.batch_id) if score_fail.batch_id else None
                        
                        # --- ADD "FAILED" LOG ---
                        error_db.add(ScoreLog(
                            score_id=score_id,
                            batch_id=batch_id,
                            worker_name="aggregator_worker",
                            status="FAILED",
                            message=error_message
                        ))
                        await error_db.commit()

                        # --- PUBLISH "FAILED" EVENT ---
                        publish_event(
                            app,
                            "score.aggregator.failed",
                            {"score_id": score_id, "batch_id": batch_id, "error": error_message}
                        )
            except Exception as db_e:
                print(f"CRITICAL: Failed to write failure status to DB: {db_e}")
            raise e

# --- Sync Wrapper ---
@app.task(name=settings.QUEUE_AGGREGATOR, bind=True)
def aggregate_scores(self, score_id: int):
    try:
        if not resources:
            raise Exception("Aggregator resources not initialized.")
        asyncio.run(async_aggregate_scores(score_id))
    except Exception as e:
        print(f"Failed to run async task: {e}")
        raise