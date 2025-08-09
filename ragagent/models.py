from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MARKDOWN = "markdown"
    CSV = "csv"


class Chunk(BaseModel):
    id: str
    text: str
    token_count: int
    document_type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_id: Optional[str] = None

    @validator("token_count")
    def validate_token_count(cls, v):
        if v < 0:
            raise ValueError("Token count cannot be negative")
        return v


class RoutingResult(BaseModel):
    selected_ids: List[str]
    scratchpad: str
    reasoning: str


class AnswerWithCitations(BaseModel):
    answer: str
    citations: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)

    @validator("citations")
    def validate_citations(cls, v, values):
        # Allow empty citations for now to avoid breaking the system
        return v or []


class VerificationConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class VerificationResult(BaseModel):
    is_accurate: bool
    explanation: str
    confidence: VerificationConfidence
    issues_found: List[str] = Field(default_factory=list)

    @validator("issues_found")
    def validate_issues_found(cls, v):
        if not isinstance(v, list):
            raise ValueError("Issues found must be a list")
        return v


class AgentConfig(BaseModel):
    model_name: str = "gpt-3.5-turbo"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "dummy-key"  # Für Testzwecke
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 30
    base_url: str = "http://localhost:11434"
    format: Optional[str] = None  # For qwen2.5 model


class DocumentProcessingConfig(BaseModel):
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    overlap_tokens: int = 100
    max_initial_chunks: int = 20
    max_navigation_depth: int = 3


class PipelineConfig(BaseModel):
    document_processing: DocumentProcessingConfig = Field(
        default_factory=DocumentProcessingConfig
    )
    agent: AgentConfig = Field(default_factory=AgentConfig)
    enable_verification: bool = True
    max_parallel_requests: int = 3


class ProcessingRequest(BaseModel):
    question: str
    document_path: str
    document_type: Optional[DocumentType] = None
    config: Optional[PipelineConfig] = None


class ProcessingResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    confidence_score: float
    verification: Optional[VerificationResult] = None
    source_paragraphs: int
    total_processing_steps: int
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorDetail(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIResponse(BaseModel):
    success: bool
    data: Optional[ProcessingResponse] = None
    error: Optional[ErrorDetail] = None
