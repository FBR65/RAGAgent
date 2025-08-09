from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
import tempfile
import os
from datetime import datetime
from pathlib import Path

from .pipeline import AgenticRAGPipeline, create_pipeline
from .unified_client import UnifiedClient, ClientFactory, ProviderType
from .enhanced_timeout_retry import ResilienceManager, get_resilience_manager
from .config_versioning import ConfigVersionManager, get_version_manager
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    PipelineConfig,
    AgentConfig,
    DocumentProcessingConfig,
    APIResponse,
    ErrorDetail,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for processing documents with AI agents",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
pipeline = create_pipeline()
unified_client = ClientFactory.create_client()
resilience_manager = get_resilience_manager()
version_manager = get_version_manager()


class ProcessRequest(BaseModel):
    question: str
    document_path: Optional[str] = None
    document_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class ProcessResponse(BaseModel):
    success: bool
    data: Optional[ProcessingResponse] = None
    error: Optional[ErrorDetail] = None


class BatchRequest(BaseModel):
    requests: List[ProcessRequest]


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agentic RAG API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test connection
        info = pipeline.test_connection()
        return {"status": "healthy", "connection": info}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """
    Process a document with a question

    Args:
        request: ProcessRequest with question and document path

    Returns:
        ProcessResponse with processing result
    """
    try:
        # Convert to ProcessingRequest
        processing_request = ProcessingRequest(
            question=request.question,
            document_path=request.document_path,
            document_type=request.document_type,
            config=PipelineConfig(**request.config) if request.config else None,
        )

        # Process request
        result = pipeline.process_request(processing_request)

        return ProcessResponse(
            success=result.success, data=result.data, error=result.error
        )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/upload")
async def process_uploaded_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    document_type: Optional[str] = Form(None),
    config: Optional[str] = Form(None),
):
    """
    Process an uploaded file with a question

    Args:
        file: Uploaded file
        question: User question
        document_type: Optional document type
        config: Optional JSON configuration

    Returns:
        ProcessResponse with processing result
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file.filename.split('.')[-1]}"
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Parse config if provided
            pipeline_config = None
            if config:
                config_data = json.loads(config)
                pipeline_config = PipelineConfig(**config_data)

            # Create request
            processing_request = ProcessingRequest(
                question=question,
                document_path=tmp_file_path,
                document_type=document_type,
                config=pipeline_config,
            )

            # Process request
            result = pipeline.process_request(processing_request)

            return ProcessResponse(
                success=result.success, data=result.data, error=result.error
            )

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except Exception as e:
        logger.error(f"Processing uploaded file failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-process")
async def batch_process(request: BatchRequest):
    """
    Process multiple requests in batch

    Args:
        request: BatchRequest with list of ProcessRequest objects

    Returns:
        List of ProcessResponse objects
    """
    try:
        # Convert to ProcessingRequest objects
        processing_requests = []
        for req in request.requests:
            processing_request = ProcessingRequest(
                question=req.question,
                document_path=req.document_path,
                document_type=req.document_type,
                config=PipelineConfig(**req.config) if req.config else None,
            )
            processing_requests.append(processing_request)

        # Process requests
        results = []
        for processing_request in processing_requests:
            result = pipeline.process_request(processing_request)
            results.append(
                ProcessResponse(
                    success=result.success, data=result.data, error=result.error
                )
            )

        return results

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/info")
async def get_system_info():
    """Get system information"""
    try:
        info = pipeline.get_system_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/test-connection")
async def test_connection(model: Optional[str] = None, base_url: Optional[str] = None):
    """
    Test connection to the model service

    Args:
        model: Optional model name
        base_url: Optional base URL

    Returns:
        Connection test results
    """
    try:
        # Update config if provided
        if model:
            pipeline.config.agent.model_name = model
        if base_url:
            pipeline.config.agent.base_url = base_url

        # Test unified client connection
        client_info = unified_client.test_connection()

        # Test resilience manager
        resilience_stats = resilience_manager.get_stats()

        # Test version manager
        version_history = version_manager.get_version_history()

        return {
            "pipeline": pipeline.test_connection(),
            "unified_client": client_info,
            "resilience_manager": resilience_stats,
            "version_manager": {
                "total_versions": len(version_history),
                "current_version": next(
                    (v["version"] for v in version_history if v["is_current"]), None
                ),
                "active_versions": len([v for v in version_history if v["is_active"]]),
            },
        }
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/chunks")
async def get_document_chunks(document_path: str, document_type: Optional[str] = None):
    """
    Extract chunks from a document

    Args:
        document_path: Path to the document
        document_type: Optional document type

    Returns:
        List of document chunks
    """
    try:
        chunks = pipeline._process_document(document_path, document_type)

        # Convert to serializable format
        chunks_data = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "document_type": chunk.document_type.value,
                "metadata": chunk.metadata,
                "parent_id": chunk.parent_id,
            }
            for chunk in chunks
        ]

        return {
            "document_path": document_path,
            "document_type": document_type,
            "chunks": chunks_data,
            "total_chunks": len(chunks),
        }

    except Exception as e:
        logger.error(f"Failed to extract chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
async def update_config(config: Dict[str, Any]):
    """
    Update pipeline configuration

    Args:
        config: New configuration

    Returns:
        Updated configuration
    """
    try:
        # Update pipeline config
        pipeline.config = PipelineConfig(**config)

        # Update unified client if agent config changed
        if "agent" in config:
            agent_config = config["agent"]
            unified_client.update_config(agent_config)

        # Save version if versioning is available
        try:
            from .config_versioning import ConfigMetadata

            metadata = ConfigMetadata(
                version=f"v{len(version_manager.get_version_history()) + 1}.0.0",
                created_at=datetime.now().isoformat(),
                created_by="api",
                description="Configuration update via API",
                is_active=True,
                is_current=True,
            )
            version_manager.save_version(pipeline.config, metadata)
        except Exception as e:
            logger.warning(f"Failed to save config version: {e}")

        return {
            "message": "Configuration updated",
            "config": pipeline.config.dict(),
            "unified_client": unified_client.get_config(),
            "resilience_stats": resilience_manager.get_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return {
            "pipeline": pipeline.config.dict(),
            "unified_client": unified_client.get_config(),
            "resilience_stats": resilience_manager.get_stats(),
            "version_history": version_manager.get_version_history(),
        }
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/versions")
async def get_config_versions():
    """Get configuration version history"""
    try:
        return version_manager.get_version_history()
    except Exception as e:
        logger.error(f"Failed to get config versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/versions/{version}/activate")
async def activate_config_version(version: str):
    """Activate specific configuration version"""
    try:
        success = version_manager.activate_version(version)
        if success:
            return {"message": f"Version {version} activated"}
        else:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
    except Exception as e:
        logger.error(f"Failed to activate config version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/versions/{version}/rollback")
async def rollback_config_version(version: str):
    """Rollback to specific configuration version"""
    try:
        success = version_manager.rollback_to_version(version)
        if success:
            # Update pipeline with rolled back config
            rolled_back_version = version_manager.get_version(version)
            if rolled_back_version:
                pipeline.config = PipelineConfig(**rolled_back_version.config_data)
                return {"message": f"Rolled back to version {version}"}
        else:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
    except Exception as e:
        logger.error(f"Failed to rollback config version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/resilience/stats")
async def get_resilience_stats():
    """Get resilience manager statistics"""
    try:
        return resilience_manager.get_stats()
    except Exception as e:
        logger.error(f"Failed to get resilience stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resilience/reset")
async def reset_resilience_stats():
    """Reset resilience manager statistics"""
    try:
        global resilience_manager
        resilience_manager = get_resilience_manager()
        return {"message": "Resilience manager reset"}
    except Exception as e:
        logger.error(f"Failed to reset resilience stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
