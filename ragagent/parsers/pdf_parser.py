from pathlib import Path
from typing import List
import logging
from .base import BaseParser
from ..models import DocumentType

logger = logging.getLogger(__name__)


class PDFParser(BaseParser):
    """Parser for PDF documents"""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a PDF"""
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> str:
        """Extract text from PDF file"""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2"
            )

        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"

            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return ""

            logger.info(f"Extracted {len(text)} characters from PDF: {file_path}")
            return text.strip()

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    def get_supported_extensions(self) -> List[str]:
        """Return supported PDF extensions"""
        return [".pdf"]
