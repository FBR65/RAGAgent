from pathlib import Path
from typing import List
import logging
from .base import BaseParser
from ..models import DocumentType

logger = logging.getLogger(__name__)


class TXTParser(BaseParser):
    """Parser for plain text files"""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a TXT file"""
        return file_path.suffix.lower() == ".txt"

    def parse(self, file_path: Path) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            if not text.strip():
                logger.warning(f"File is empty: {file_path}")
                return ""

            logger.info(f"Read {len(text)} characters from TXT file: {file_path}")
            return text.strip()

        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    text = file.read()
                logger.info(
                    f"Read {len(text)} characters from TXT file (latin-1): {file_path}"
                )
                return text.strip()
            except Exception as e:
                logger.error(f"Error reading TXT file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading TXT file {file_path}: {e}")
            raise

    def get_supported_extensions(self) -> List[str]:
        """Return supported TXT extensions"""
        return [".txt"]
