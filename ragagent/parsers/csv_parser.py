from pathlib import Path
from typing import List
import logging
import csv
from .base import BaseParser
from ..models import DocumentType

logger = logging.getLogger(__name__)


class CSVParser(BaseParser):
    """Parser for CSV files"""

    def can_parse(self, file_path: Path) -> bool:
        """Check if file is a CSV file"""
        return file_path.suffix.lower() == ".csv"

    def parse(self, file_path: Path) -> str:
        """Extract text from CSV file"""
        try:
            text = ""
            with open(file_path, "r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file)
                for i, row in enumerate(reader):
                    if i == 0:
                        # Header row
                        text += (
                            "HEADERS: " + " | ".join(str(cell) for cell in row) + "\n"
                        )
                    else:
                        # Data rows
                        text += (
                            f"ROW {i}: " + " | ".join(str(cell) for cell in row) + "\n"
                        )

                    if i >= 100:  # Limit to first 100 rows to avoid huge outputs
                        text += f"... and {len(list(csv.reader(open(file_path)))) - 101} more rows\n"
                        break

            if not text.strip():
                logger.warning(f"No data extracted from CSV: {file_path}")
                return ""

            logger.info(f"Extracted {len(text)} characters from CSV: {file_path}")
            return text.strip()

        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1", newline="") as file:
                    reader = csv.reader(file)
                    text = ""
                    for i, row in enumerate(reader):
                        if i == 0:
                            text += (
                                "HEADERS: "
                                + " | ".join(str(cell) for cell in row)
                                + "\n"
                            )
                        else:
                            text += (
                                f"ROW {i}: "
                                + " | ".join(str(cell) for cell in row)
                                + "\n"
                            )
                        if i >= 100:
                            break
                logger.info(
                    f"Read {len(text)} characters from CSV file (latin-1): {file_path}"
                )
                return text.strip()
            except Exception as e:
                logger.error(f"Error reading CSV file {file_path}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            raise

    def get_supported_extensions(self) -> List[str]:
        """Return supported CSV extensions"""
        return [".csv"]
