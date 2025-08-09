import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .models import (
    ProcessingRequest,
    ProcessingResponse,
    PipelineConfig,
    AgentConfig,
    DocumentProcessingConfig,
    Chunk,
    AnswerWithCitations,
    VerificationResult,
    ErrorDetail,
    APIResponse,
)
from .parsers import ParserRegistry
from .agents import (
    DeepDiverAgent,
    AnswerSynthesizerAgent,
    JudgeAgent,
)
from .openai_client import OpenAIClientFactory
from .utils import (
    CacheManager,
    RequestPoolManager,
    RequestPriority,
    get_error_handler,
    with_retry,
    get_error_stats,
)

logger = logging.getLogger(__name__)


class AgenticRAGPipeline:
    """Main pipeline for the Agentic RAG system"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.parser_registry = ParserRegistry()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize caching and pooling
        self.cache_manager = CacheManager(
            memory_cache_size=self.config.get("cache", {}).get("memory_size", 1000),
            disk_cache_dir=self.config.get("cache", {}).get("disk_dir", None),
            disk_cache_size_mb=self.config.get("cache", {}).get("disk_size_mb", 1000),
            default_ttl=self.config.get("cache", {}).get("default_ttl", 3600),
        )

        self.pool_manager = RequestPoolManager(
            max_connections=self.config.get("pool", {}).get("max_connections", 10),
            max_concurrent_requests=self.config.get("pool", {}).get(
                "max_concurrent_requests", 5
            ),
            enable_priority=self.config.get("pool", {}).get("enable_priority", True),
        )

        # Setup logging first
        self._setup_logging()

        # Setup parsers after logging is configured
        self._setup_parsers()

    def _setup_parsers(self):
        """Setup all document parsers"""
        from .parsers import PDFParser, TXTParser, DOCXParser, MarkdownParser, CSVParser

        parsers = [
            PDFParser(),
            TXTParser(),
            DOCXParser(),
            MarkdownParser(),
            CSVParser(),
        ]

        for parser in parsers:
            self.parser_registry.register(parser)

        self.logger.info(f"Registered {len(parsers)} parsers")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("ragagent.log")],
        )

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def process_request(self, request: ProcessingRequest) -> APIResponse:
        """
        Process a single RAG request with caching and error handling

        Args:
            request: ProcessingRequest with question and document path

        Returns:
            APIResponse with processing result or error
        """
        start_time = time.time()
        error_handler = get_error_handler()

        try:
            self.logger.info(f"Processing request: {request.question}")
            self.logger.info(f"Document path: {request.document_path}")

            # Check cache first
            cached_response = self.cache_manager.get(request)
            if cached_response:
                self.logger.info("Returning cached response")
                # Update processing time
                cached_response.processing_time = time.time() - start_time
                return APIResponse(success=True, data=cached_response)

            # Override config if provided in request
            if request.config:
                self.config = request.config

            # Process document with retry
            document_chunks = self._process_document_with_retry(
                request.document_path, request.document_type
            )

            # Navigate to relevant chunks with retry
            relevant_chunks = self._navigate_to_relevant_chunks_with_retry(
                request.question, document_chunks
            )

            # Generate answer with retry
            answer = self._generate_answer_with_retry(request.question, relevant_chunks)

            # Verify answer with retry
            verification = None
            if self.config.enable_verification:
                verification = self._verify_answer_with_retry(
                    request.question, answer, relevant_chunks
                )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Build response
            response = ProcessingResponse(
                question=request.question,
                answer=answer.answer,
                citations=answer.citations,
                confidence_score=answer.confidence_score,
                verification=verification,
                source_paragraphs=len(relevant_chunks),
                total_processing_steps=4,
                processing_time=processing_time,
                metadata={
                    "document_path": str(request.document_path),
                    "document_type": request.document_type.value
                    if request.document_type
                    else "unknown",
                    "config": self.config.dict(),
                },
            )

            # Cache the response
            self.cache_manager.set(request, response)

            self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
            self.logger.info(f"Generated answer with {len(answer.citations)} citations")

            return APIResponse(success=True, data=response)

        except Exception as e:
            processing_time = time.time() - start_time

            # Create error context
            from .utils import ErrorContext, ErrorType

            error_context = ErrorContext(
                error_type=error_handler.classify_error(e),
                exception=e,
                attempt=1,
                total_attempts=3,
                request=request,
            )

            # Handle error with recovery strategies
            error_response = error_handler.handle_error(error_context)
            if error_response:
                return error_response

            # Fallback error response
            error_detail = ErrorDetail(
                error="ProcessingError",
                message=str(e),
                details={
                    "processing_time": processing_time,
                    "error_type": error_context.error_type.value,
                    "cache_stats": self.cache_manager.get_stats(),
                },
            )

            self.logger.error(f"Processing failed: {e}")
            return APIResponse(success=False, error=error_detail)

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def _process_document_with_retry(
        self, document_path: str, document_type=None
    ) -> List[Chunk]:
        """Process document with retry logic"""
        return self._process_document(document_path, document_type)

    def _process_document(self, document_path: str, document_type=None) -> List[Chunk]:
        """Process document and return chunks"""
        path = Path(document_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")

        # Determine document type
        if document_type is None:
            document_type = self.parser_registry._get_document_type(path)

        self.logger.info(f"Processing document: {document_path}")
        self.logger.info(f"Document type: {document_type}")

        # Parse document
        chunks = self.parser_registry.parse_file(path)

        self.logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def _navigate_to_relevant_chunks_with_retry(
        self, question: str, document_chunks: List[Chunk]
    ) -> List[Chunk]:
        """Navigate to relevant chunks with retry logic"""
        return self._navigate_to_relevant_chunks(question, document_chunks)

    def _navigate_to_relevant_chunks(
        self, question: str, document_chunks: List[Chunk]
    ) -> List[Chunk]:
        """Navigate to relevant chunks using the deep diver agent"""

        self.logger.info("Starting chunk navigation")

        agent_config = self.config.agent
        processing_config = self.config.document_processing

        diver = DeepDiverAgent(agent_config, processing_config)
        try:
            relevant_chunks = diver.navigate_to_paragraphs(question, document_chunks)
        finally:
            diver.close()

        self.logger.info(
            f"Navigation completed, found {len(relevant_chunks)} relevant chunks"
        )
        return relevant_chunks

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def _generate_answer_with_retry(
        self, question: str, relevant_chunks: List[Chunk]
    ) -> AnswerWithCitations:
        """Generate answer with retry logic"""
        return self._generate_answer(question, relevant_chunks)

    def _generate_answer(
        self, question: str, relevant_chunks: List[Chunk]
    ) -> AnswerWithCitations:
        """Generate answer from relevant chunks"""

        self.logger.info("Generating answer")

        agent_config = self.config.agent

        synthesizer = AnswerSynthesizerAgent(agent_config)
        try:
            answer = synthesizer.generate_answer(question, relevant_chunks)
        finally:
            synthesizer.close()

        self.logger.info(f"Answer generated with confidence: {answer.confidence_score}")
        return answer

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=30.0)
    def _verify_answer_with_retry(
        self, question: str, answer: AnswerWithCitations, relevant_chunks: List[Chunk]
    ) -> VerificationResult:
        """Verify answer with retry logic"""
        return self._verify_answer(question, answer, relevant_chunks)

    def _verify_answer(
        self, question: str, answer: AnswerWithCitations, relevant_chunks: List[Chunk]
    ) -> VerificationResult:
        """Verify the generated answer"""

        self.logger.info("Verifying answer")

        agent_config = self.config.agent

        judge = JudgeAgent(agent_config)
        try:
            verification = judge.verify_answer(question, answer, relevant_chunks)
        finally:
            judge.close()

        self.logger.info(f"Answer verification completed: {verification.is_accurate}")
        return verification

    def batch_process(self, requests: List[ProcessingRequest]) -> List[APIResponse]:
        """
        Process multiple requests in batch

        Args:
            requests: List of ProcessingRequest objects

        Returns:
            List of APIResponse objects
        """

        self.logger.info(f"Starting batch processing of {len(requests)} requests")

        results = []

        for i, request in enumerate(requests):
            self.logger.info(f"Processing request {i + 1}/{len(requests)}")

            try:
                result = self.process_request(request)
                results.append(result)

                # Small delay between requests to avoid overwhelming the system
                if i < len(requests) - 1:
                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error processing request {i + 1}: {e}")
                error_detail = ErrorDetail(
                    error="BatchProcessingError",
                    message=str(e),
                    details={"request_index": i},
                )
                results.append(APIResponse(success=False, error=error_detail))

        successful_requests = sum(1 for r in results if r.success)
        self.logger.info(
            f"Batch processing completed: {successful_requests}/{len(requests)} successful"
        )

        return results

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the system"""

        return {
            "version": "0.1.0",
            "parsers": len(self.parser_registry.get_all_parsers()),
            "config": self.config.dict(),
            "supported_formats": ["pdf", "txt", "docx", "md", "markdown", "csv"],
            "cache_stats": self.cache_manager.get_stats(),
            "pool_stats": self.pool_manager.get_stats(),
            "error_stats": get_error_stats(),
        }

    def test_connection(self) -> Dict[str, Any]:
        """Test connection to configured AI service and check model availability"""

        try:
            # Use OpenAI-compatible client for both OpenAI and Ollama
            client_type = "openai"
            client = OpenAIClientFactory.create_client(self.config.agent)

            try:
                is_available = client.check_model_availability()
                model_info = client.get_model_info()
            finally:
                client.close()

            return {
                f"{client_type}_connection": "success",
                "model_available": is_available,
                "model_name": self.config.agent.model_name,
                "model_info": model_info,
                "base_url": self.config.agent.base_url,
            }

        except Exception as e:
            client_type = "openai"
            return {
                f"{client_type}_connection": "failed",
                "error": str(e),
                "model_available": False,
                "model_name": self.config.agent.model_name,
                "base_url": self.config.agent.base_url,
            }


# Factory function for creating pipeline instances
def create_pipeline(config: Optional[PipelineConfig] = None) -> AgenticRAGPipeline:
    """Create a new AgenticRAGPipeline instance"""
    return AgenticRAGPipeline(config)


# Convenience function for simple processing
def process_document(
    question: str, document_path: str, config: Optional[PipelineConfig] = None
) -> APIResponse:
    """
    Simple function for processing a single document

    Args:
        question: User question
        document_path: Path to the document
        config: Optional pipeline configuration

    Returns:
        APIResponse with processing result
    """

    request = ProcessingRequest(
        question=question, document_path=document_path, config=config
    )

    pipeline = create_pipeline(config)
    return pipeline.process_request(request)
