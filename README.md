# 🤖 Agentic HR: Intelligent Resume Matching Platform

## 1\. Executive Summary

**Agentic HR** is an event-driven, microservices-based platform designed to automate the hiring pipeline. It ingests resumes, parses them using OCR and LLMs, generates semantic embeddings, and scores candidates against job descriptions using a hybrid approach (Vector Similarity + Generative AI reasoning).

The system is built for **high concurrency** and **observability**, using an asynchronous message bus to decouple heavy processing tasks from user-facing APIs.

-----

## 2\. Architecture Overview
## 🧭 System Architecture
![System Architecture](docs/images/Event-Driven%20Matching%20Pipeline%20Architecture.png)
The system follows a **Producer-Consumer** pattern using RabbitMQ as the event bus.

### Core Technology Stack

  * **APIs:** Python 3.11, FastAPI (Async)
  * **Task Orchestration:** Celery
  * **Message Broker:** RabbitMQ (Topic Exchange)
  * **Databases:**
      * **PostgreSQL:** Relational data (Jobs, Resumes, Scores, Logs).
      * **Qdrant:** Vector database for semantic search.
      * **Redis:** High-speed cache for real-time dashboard counters.
  * **Object Storage:** MinIO (S3 Compatible) for raw resume files.
  * **AI Engines:** OpenAI (GPT-4/3.5) for reasoning, `sentence-transformers` for local embeddings.

### High-Level Data Flow

-----

## 3\. Service Dictionary

### A. The APIs (Gateways)

1.  **`Job API` (Port 8000):**
      * Manages Job Descriptions, Extraction Schemas, and Scoring Rubrics.
      * Triggers the `job_embedder_worker`.
2.  **`Resume API` (Port 8081):**
      * Handles file uploads (Single & Batch).
      * Saves files to MinIO.
      * Triggers the ingestion pipeline (`parser_worker`).
      * Provides real-time status of resume processing.
3.  **`Match API` (Port 8002):**
      * The central orchestrator.
      * Triggers the matching pipeline between a Job and a Resume Batch.
      * Serves as the **Dashboard Backend**, reading aggregated status from Redis and detailed reports from Postgres.

### B. The Workers (Consumers)

| Worker Name | Queue Name | Responsibility |
| :--- | :--- | :--- |
| **Job Embedder** | `q_embed_job` | Chunks job descriptions and saves vectors to Qdrant. |
| **Parser** | `q_parse_resume` | Downloads file from S3, extracts raw text (OCR/PDF parsing). |
| **Extractor** | `q_extract_profile` | Uses OpenAI to extract structured JSON (skills, experience) based on a dynamic schema. |
| **Embedder** | `q_embed_resume` | Generates vectors for resume chunks and saves to Qdrant. |
| **Ingestion Aggregator** | `q_ingest_agg` | "Gatekeeper." Checks if *both* extraction and embedding are done. Marks resume `READY_TO_SCORE`. |
| **Scoring Engine** | `q_scoring_engine` | Orchestrator. Fans out tasks to Semantic and LLM scorers. |
| **Semantic Scorer** | `q_score_semantic` | Performs Cosine Similarity search in Qdrant. |
| **LLM Scorer** | `q_score_llm` | Uses OpenAI to grade resumes against a Rubric, generating sub-scores, rationales, and evidence. |
| **Aggregator** | `q_score_aggregate` | "Gatekeeper." Waits for both scores, calculates weighted average, and marks the match `COMPLETE`. |
| **Dashboard Aggregator** | `(Topic Listener)` | Listens to *all* RabbitMQ events and updates real-time counters in Redis. |

-----

## 4\. Data Models (Database Schema)

The system uses a relational schema in PostgreSQL with strict foreign key constraints.

  * **`MatchBatch`**: Represents a group of resumes uploaded together or a matching session. Used for tracking progress.
  * **`ScoreLog`**: An append-only log table. Every worker writes a row here (`STARTED`, `COMPLETED`, `FAILED`) to provide a detailed history for the dashboard.
  * **`Resume`**: Stores metadata, S3 file keys, parsed text, and the extracted JSON profile.
  * **`Score`**: The central join table between a `Job` and a `Resume`. Stores the final weighted score.
  * **`LlmScore` / `SemanticScore`**: Specialized tables storing the raw results from specific workers to prevent race conditions.

-----

## 5\. API Reference & Workflows

### Workflow 1: Resume Ingestion

**Goal:** Upload 50 resumes and get them ready for matching.

1.  **Upload:** `POST /api/v1/resumes/upload-batch`
      * **Input:** Multipart form data (files).
      * **Output:** `batch_id` (UUID).
2.  **Monitor:** `GET /api/v1/resumes/dashboard-status/batch/{batch_id}`
      * **Output:** Real-time counters from Redis (e.g., `{"parsing_completed": 45, "embedding_completed": 50}`).
3.  **Verify:** The Ingestion Aggregator worker marks resumes as `READY_TO_SCORE` once processing is done.

### Workflow 2: Job Creation

**Goal:** Add a new job opening.

1.  **Create:** `POST /api/v1/jobs`
      * **Input:** Title, Description, Schema ID, Rubric ID.
      * **Output:** `job_id`.
2.  **Process:** System automatically chunks and embeds this job description into Qdrant.

### Workflow 3: Matching & Scoring

**Goal:** Rank the 50 resumes against the new Job.

1.  **Get Resumes:** `GET /api/v1/resumes/batch/{resume_batch_id}`
      * **Output:** List of `resume_id`s that are `READY_TO_SCORE`.
2.  **Start Match:** `POST /api/v1/match/by-batch`
      * **Input:** `job_id`, `resume_batch_id`.
      * **Output:** `match_batch_id`.
3.  **Monitor:** `GET /api/v1/match/dashboard-status/batch/{match_batch_id}`
      * **Output:** Redis counters showing scoring progress.
4.  **Get Results:** `GET /api/v1/match/reports/batch/{match_batch_id}`
      * **Output:** A list of `FinalReportResponse` objects, sorted by highest score, including the candidate's name, email, score breakdown, and rationale.

-----

## 6\. Environment Configuration (`.env`)

```ini
# --- Databases ---
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin
POSTGRES_DB=hr_db
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
DATABASE_URL_PSYCOPG=postgresql+psycopg://admin:admin@postgres:5432/hr_db

# --- Brokers & Cache ---
RABBITMQ_USER=admin
RABBITMQ_PASS=admin
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
BROKER_URL=amqp://admin:admin@rabbitmq:5672//

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# --- Monitoring ---
MONITORING_EXCHANGE_NAME=hr_monitoring_exchange

# --- Vector DB & AI ---
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=resumes
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-3.5-turbo

# --- Storage ---
MINIO_ENDPOINT_URL=http://minio:9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET_NAME=resumes

# --- Queues ---
QUEUE_PARSER=q_parse_resume
QUEUE_EXTRACTOR=q_extract_profile
QUEUE_RESUME_EMBEDDER=q_embed_resume
QUEUE_INGESTION_AGGREGATOR=q_ingest_agg
QUEUE_JOB_EMBEDDER=q_embed_job
QUEUE_SCORING_ENGINE=q_scoring_engine
QUEUE_SEMANTIC_SCORER=q_score_semantic
QUEUE_LLM_SCORER=q_score_llm
QUEUE_AGGREGATOR=q_score_aggregate
```

-----

## 7\. Deployment Instructions

### Running Locally with Docker Compose

1.  **Build and Start:**

    ```bash
    docker-compose up -d --build
    ```

    *This spins up all containers, creates database tables, and initializes the message bus.*

2.  **Accessing Services:**

      * **Job API:** `http://localhost:8000/docs`
      * **Resume API:** `http://localhost:8001/docs`
      * **Match API:** `http://localhost:8002/docs`
      * **MinIO Console:** `http://localhost:9001`
      * **RabbitMQ UI:** `http://localhost:15672`

3.  **Stopping:**

    ```bash
    docker-compose down
    ```

### Maintenance Commands

  * **Clean Wipe (Danger\!):** If you need to reset the database, vectors, and file storage:
    ```bash
    docker-compose down
    docker system prune -a --volumes
    ```
  * **Viewing Logs for a Specific Worker:**
    ```bash
    docker-compose logs -f hr_llm_scorer
    ```

-----

## 8\. Troubleshooting Guide

| Issue | Likely Cause | Fix |
| :--- | :--- | :--- |
| **MinIO "Storage Full"** | Docker volume is on a full partition (C: drive). | Run `docker system prune` or move Docker storage location in settings. |
| **Qdrant "OutputTooSmall"** | Race condition corrupted the vector collection. | Stop containers and wipe volumes to recreate the collection cleanly. |
| **Event Loop Closed** | `asyncio` resources initialized in the wrong scope. | Ensure resources like `s3_client` are created *inside* the async task, not globally. |
| **404 on Dashboard** | Redis is empty because the Aggregator isn't running. | Ensure `hr_dashboard_aggregator` container is up and connected to RabbitMQ. |