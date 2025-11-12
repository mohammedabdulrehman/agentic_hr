# /libs/schemas/models.py
import uuid
from sqlalchemy import (
    Column, Integer, String, Text, TIMESTAMP, ForeignKey, func, JSON, Boolean, Float
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

# --- Base Class ---
Base = declarative_base()

# --- 1. Define Schemas, Rubrics, and Jobs first ---
class ExtractionSchema(Base):
    __tablename__ = "extraction_schemas"
    schema_id = Column(Integer, primary_key=True, autoincrement=True)
    schema_name = Column(String(100), unique=True, nullable=False, index=True)
    schema_definition = Column(JSON, nullable=False)
    jobs = relationship("Job", back_populates="extraction_schema")

class ScoringRubric(Base):
    __tablename__ = "scoring_rubrics"
    rubric_id = Column(Integer, primary_key=True, autoincrement=True)
    rubric_name = Column(String(100), unique=True, nullable=False, index=True)
    rubric_definition = Column(JSON, nullable=False)
    jobs = relationship("Job", back_populates="scoring_rubric")
    
class Job(Base):
    __tablename__ = "jobs"
    job_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    extraction_schema_id = Column(Integer, ForeignKey("extraction_schemas.schema_id"), nullable=True)
    extraction_schema = relationship("ExtractionSchema", back_populates="jobs")
    scoring_rubric_id = Column(Integer, ForeignKey("scoring_rubrics.rubric_id"), nullable=True)
    scoring_rubric = relationship("ScoringRubric", back_populates="jobs")
    scores = relationship("Score", back_populates="job")

# --- 2. Define Resume and Score next ---
# These are the "parent" tables for the logs and sub-scores
class Resume(Base):
    __tablename__ = "resumes"
    resume_id = Column(Integer, primary_key=True, autoincrement=True)
    tracking_id = Column(PG_UUID(as_uuid=True), default=uuid.uuid4, unique=True, index=True)
    batch_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    filename = Column(String(255))
    file_key = Column(String(1024))
    extraction_schema_id = Column(Integer, ForeignKey("extraction_schemas.schema_id"), nullable=True)
    is_embedded: Column[bool] = Column(Boolean, default=False, nullable=False)
    is_extracted: Column[bool] = Column(Boolean, default=False, nullable=False)
    parsed_text = Column(Text, nullable=True)
    profile_json = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    scores = relationship("Score", back_populates="resume")
    # This relationship will be created *after* ScoreLog is defined
    logs = relationship("ScoreLog", back_populates="resume")

class Score(Base):
    __tablename__ = "scores"
    score_id = Column(Integer, primary_key=True, autoincrement=True)
    resume_id = Column(Integer, ForeignKey("resumes.resume_id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("jobs.job_id"), nullable=False, index=True)
    batch_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    status = Column(String(50), nullable=False, default='PENDING', index=True)
    overall_score = Column(Integer, nullable=True)
    rationale = Column(Text, nullable=True)
    evidence_json = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    resume = relationship("Resume", back_populates="scores")
    job = relationship("Job", back_populates="scores")
    # These relationships will be created *after* the sub-tables are defined
    semantic_score = relationship("SemanticScore", back_populates="score", uselist=False, cascade="all, delete-orphan")
    llm_score = relationship("LlmScore", back_populates="score", uselist=False, cascade="all, delete-orphan")
    logs = relationship("ScoreLog", back_populates="score")

# --- 3. Define the "child" tables last ---
# These tables have ForeignKeys pointing to the models above

class SemanticScore(Base):
    __tablename__ = "semantic_scores"
    score_id = Column(Integer, ForeignKey("scores.score_id"), primary_key=True)
    score_value = Column(Float, nullable=False)
    score = relationship("Score", back_populates="semantic_score")

class LlmScore(Base):
    __tablename__ = "llm_scores"
    score_id = Column(Integer, ForeignKey("scores.score_id"), primary_key=True)
    scores_json = Column(JSON, nullable=False)
    score = relationship("Score", back_populates="llm_score")

class ScoreLog(Base):
    __tablename__ = "score_logs"
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    score_id = Column(Integer, ForeignKey("scores.score_id"), nullable=True, index=True)
    resume_id = Column(Integer, ForeignKey("resumes.resume_id"), nullable=True, index=True)
    batch_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    worker_name = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    message = Column(Text, nullable=True)
    
    # These back-populate the 'logs' properties on Score and Resume
    score = relationship("Score", back_populates="logs")
    resume = relationship("Resume", back_populates="logs")