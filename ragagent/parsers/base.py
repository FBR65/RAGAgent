from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from ..models import Chunk, DocumentType

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """Base class for all document parsers"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file"""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """Parse the file and return extracted text"""
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        pass

    def create_chunks(
        self,
        text: str,
        document_type: DocumentType,
        file_path: Path,
        max_chunk_size: int = 2000,
    ) -> List[Chunk]:
        """Split text into chunks with proper token counting"""
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self.logger.warning("tiktoken not available, using simple chunking")
            return self._simple_chunking(text, document_type, file_path)

        chunks = []
        sentences = self._split_into_sentences(text)
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(tokenizer.encode(sentence))

            if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
                chunks.append(
                    self._create_chunk_object(
                        current_chunk.strip(), document_type, file_path, len(chunks)
                    )
                )
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(
                self._create_chunk_object(
                    current_chunk.strip(), document_type, file_path, len(chunks)
                )
            )

        self.logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks

    def _simple_chunking(
        self,
        text: str,
        document_type: DocumentType,
        file_path: Path,
        max_chunk_size: int = 2000,
    ) -> List[Chunk]:
        """Fallback chunking method when tiktoken is not available"""
        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(
                    self._create_chunk_object(
                        current_chunk.strip(), document_type, file_path, len(chunks)
                    )
                )
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(
                self._create_chunk_object(
                    current_chunk.strip(), document_type, file_path, len(chunks)
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        try:
            import nltk

            nltk.data.find("tokenizers/punkt")
            sentences = nltk.sent_tokenize(text)
        except (ImportError, LookupError):
            self.logger.warning("NLTK not available, using simple sentence splitting")
            sentences = self._simple_sentence_split(text)

        return [s.strip() for s in sentences if s.strip()]

    def _simple_sentence_split(self, text: str) -> List[str]:
        """Simple sentence splitting as fallback"""
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk_object(
        self, text: str, document_type: DocumentType, file_path: Path, chunk_id: int
    ) -> Chunk:
        """Create a Chunk object with metadata"""
        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
            token_count = len(tokenizer.encode(text))
        except ImportError:
            token_count = len(text.split())

        return Chunk(
            id=f"{file_path.stem}_{chunk_id}",
            text=text,
            token_count=token_count,
            document_type=document_type,
            metadata={
                "file_path": str(file_path),
                "file_name": file_path.name,
                "chunk_id": chunk_id,
            },
        )


class ParserRegistry:
    """Registry for document parsers"""

    def __init__(self):
        self._parsers: List[BaseParser] = []

    def register(self, parser: BaseParser):
        """Register a new parser"""
        self._parsers.append(parser)
        logger.info(f"Registered parser: {parser.__class__.__name__}")

    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Get appropriate parser for file"""
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def get_all_parsers(self) -> List[BaseParser]:
        """Get all registered parsers"""
        return self._parsers.copy()

    def parse_file(self, file_path: Path) -> List[Chunk]:
        """Parse file using appropriate parser"""
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser found for file: {file_path}")

        try:
            text = parser.parse(file_path)
            document_type = self._get_document_type(file_path)
            return parser.create_chunks(text, document_type, file_path)
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise

    def _get_document_type(self, file_path: Path) -> DocumentType:
        """Determine document type from file extension"""
        extension = file_path.suffix.lower().lstrip(".")

        type_mapping = {
            "pdf": DocumentType.PDF,
            "txt": DocumentType.TXT,
            "docx": DocumentType.DOCX,
            "md": DocumentType.MARKDOWN,
            "markdown": DocumentType.MARKDOWN,
            "csv": DocumentType.CSV,
        }

        if extension not in type_mapping:
            raise ValueError(f"Unsupported file extension: {extension}")

        return type_mapping[extension]
