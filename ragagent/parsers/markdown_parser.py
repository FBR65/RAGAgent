from pathlib import Path
from typing import List
import logging
from .base import BaseParser
from ..models import DocumentType

logger = logging.getLogger(__name__)


class MarkdownParser(BaseParser):
    """Parser for Markdown files"""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a Markdown file"""
        return file_path.suffix.lower() in [".md", ".markdown"]

    def parse(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                logger.warning(f"File is empty: {file_path}")
                return ""

            logger.info(f"Read {len(text)} characters from Markdown file: {file_path}")
            return self._clean_markdown_text(text)

        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()
                logger.info(
                    f"Read {len(text)} characters from Markdown file (latin-1): {file_path}"
                )
                return self._clean_markdown_text(text)
            except Exception as e:
                logger.error(f"Error reading Markdown file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading Markdown file {file_path}: {e}")
            raise

    def _clean_markdown_text(self, text: str) -> str:
        """Clean and normalize markdown text"""
        import re

        # Remove markdown links but keep the text
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # Remove markdown images
        text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def get_supported_extensions(self) -> List[str]:
        """Return supported Markdown extensions"""
        return [".md", ".markdown"]
