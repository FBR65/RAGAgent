"""
OCR support for scanned documents in the RAG system
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import io
import time

import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

from .models import (
    ProcessingRequest,
    ProcessingResponse,
    Chunk,
    DocumentType,
    ErrorDetail,
    APIResponse,
)
from .parsers.base import BaseParser
from .utils import get_file_hash

logger = logging.getLogger(__name__)


class OCRMode(Enum):
    """OCR processing modes"""

    FULL_PAGE = "full_page"
    BLOCK_BASED = "block_based"
    PARAGRAPH_BASED = "paragraph_based"
    WORD_BASED = "word_based"
    LINE_BASED = "line_based"


class OCRQuality(Enum):
    """OCR quality levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BEST = "best"


@dataclass
class OCRConfig:
    """OCR configuration"""

    mode: OCRMode = OCRMode.PARAGRAPH_BASED
    quality: OCRQuality = OCRQuality.HIGH
    language: str = "eng"
    dpi: int = 300
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    enable_layout_analysis: bool = True
    max_pages: int = 100
    timeout: int = 300
    retry_count: int = 3
    cache_enabled: bool = True


@dataclass
class OCRResult:
    """OCR processing result"""

    text: str
    confidence: float
    processing_time: float
    pages_processed: int
    metadata: Dict[str, Any]
    layout_info: Optional[Dict[str, Any]] = None


class OCRPreprocessor:
    """Preprocess images for better OCR results"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for OCR"""
        try:
            # Convert to grayscale
            if image.mode != "L":
                image = image.convert("L")

            # Apply preprocessing if enabled
            if self.config.enable_preprocessing:
                image = self._apply_denoising(image)
                image = self._apply_binarization(image)
                image = self._apply_contrast_enhancement(image)
                image = self._apply_deskew(image)

            return image

        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return image

    def _apply_denoising(self, image: Image.Image) -> Image.Image:
        """Apply denoising to image"""
        try:
            # Simple denoising using Gaussian blur
            from PIL import ImageFilter

            return image.filter(ImageFilter.GaussianBlur(radius=0.5))
        except Exception as e:
            self.logger.error(f"Error applying denoising: {e}")
            return image

    def _apply_binarization(self, image: Image.Image) -> Image.Image:
        """Apply binarization to image"""
        try:
            # Simple thresholding
            import numpy as np

            # Convert to numpy array
            img_array = np.array(image)

            # Apply threshold
            threshold = 128
            binary_img = img_array > threshold
            binary_img = binary_img * 255

            # Convert back to PIL Image
            return Image.fromarray(binary_img.astype(np.uint8))

        except Exception as e:
            self.logger.error(f"Error applying binarization: {e}")
            return image

    def _apply_contrast_enhancement(self, image: Image.Image) -> Image.Image:
        """Apply contrast enhancement to image"""
        try:
            from PIL import ImageEnhance

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)

        except Exception as e:
            self.logger.error(f"Error applying contrast enhancement: {e}")
            return image

    def _apply_deskew(self, image: Image.Image) -> Image.Image:
        """Apply deskewing to image"""
        try:
            # Simple deskewing using OpenCV
            import cv2
            import numpy as np

            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Find contours
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Get the minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[-1]

                # Correct the skew
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle

                # Rotate the image
                (h, w) = image.size
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    cv_image,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

                return Image.fromarray(rotated)

            return image

        except Exception as e:
            self.logger.error(f"Error applying deskewing: {e}")
            return image


class OCRPostprocessor:
    """Postprocess OCR results"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def postprocess_text(self, text: str) -> str:
        """Postprocess OCR text"""
        try:
            if not self.config.enable_postprocessing:
                return text

            # Apply various text cleaning operations
            text = self._remove_extra_whitespace(text)
            text = self._fix_common_ocr_errors(text)
            text = self._normalize_text(text)
            text = self._preserve_document_structure(text)

            return text

        except Exception as e:
            self.logger.error(f"Error postprocessing text: {e}")
            return text

    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        import re

        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors"""
        import re

        # Common OCR error corrections
        corrections = {
            r"0": "O",  # Zero to letter O
            r"1": "I",  # One to letter I
            r"5": "S",  # Five to letter S
            r"6": "G",  # Six to letter G
            r"8": "B",  # Eight to letter B
            r"9": "g",  # Nine to letter g
            r"\bll\b": "II",  # Double L to Roman numeral II
            r"\bbl\b": "BI",  # BL to BI
            r"\bcl\b": "CI",  # CL to CI
            r"\btl\b": "TI",  # TL to TI
        }

        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text)

        return text

    def _normalize_text(self, text: str) -> str:
        """Normalize text"""
        import re

        # Normalize punctuation
        text = re.sub(
            r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\~\`\|\\]",
            " ",
            text,
        )

        # Normalize quotes
        text = re.sub(r'[""' "`]", '"', text)
        text = re.sub(r"''", "'", text)

        return text

    def _preserve_document_structure(self, text: str) -> str:
        """Preserve document structure"""
        import re

        # Ensure proper spacing after punctuation
        text = re.sub(r"([\.!\?,;])([a-zA-Z])", r"\1 \2", text)

        # Ensure proper paragraph spacing
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text


class LayoutAnalyzer:
    """Analyze document layout"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout"""
        try:
            if not self.config.enable_layout_analysis:
                return {}

            # Convert image to numpy array
            import numpy as np

            img_array = np.array(image)

            # Find text blocks
            blocks = self._find_text_blocks(img_array)

            # Analyze document structure
            structure = self._analyze_document_structure(blocks)

            return structure

        except Exception as e:
            self.logger.error(f"Error analyzing layout: {e}")
            return {}

    def _find_text_blocks(self, img_array: np.ndarray) -> List[Dict[str, Any]]:
        """Find text blocks in image"""
        try:
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # Apply threshold
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            blocks = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Filter out small blocks
                if w > 20 and h > 10:
                    blocks.append(
                        {"x": x, "y": y, "width": w, "height": h, "area": w * h}
                    )

            return blocks

        except Exception as e:
            self.logger.error(f"Error finding text blocks: {e}")
            return []

    def _analyze_document_structure(
        self, blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze document structure"""
        try:
            if not blocks:
                return {}

            # Sort blocks by y-coordinate (top to bottom)
            blocks.sort(key=lambda b: b["y"])

            # Group blocks into lines
            lines = self._group_blocks_into_lines(blocks)

            # Group lines into paragraphs
            paragraphs = self._group_lines_into_paragraphs(lines)

            # Identify document elements
            elements = self._identify_document_elements(paragraphs)

            return {
                "blocks": blocks,
                "lines": lines,
                "paragraphs": paragraphs,
                "elements": elements,
                "total_blocks": len(blocks),
                "total_lines": len(lines),
                "total_paragraphs": len(paragraphs),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing document structure: {e}")
            return {}

    def _group_blocks_into_lines(
        self, blocks: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group blocks into lines"""
        lines = []
        current_line = []
        current_y = 0
        tolerance = 10

        for block in blocks:
            if current_line and abs(block["y"] - current_y) > tolerance:
                # Start new line
                lines.append(current_line)
                current_line = [block]
                current_y = block["y"]
            else:
                # Add to current line
                current_line.append(block)
                current_y = block["y"]

        if current_line:
            lines.append(current_line)

        return lines

    def _group_lines_into_paragraphs(
        self, lines: List[List[Dict[str, Any]]]
    ) -> List[List[List[Dict[str, Any]]]]:
        """Group lines into paragraphs"""
        paragraphs = []
        current_paragraph = []
        current_x = 0
        tolerance = 50

        for line in lines:
            if current_paragraph and abs(line[0]["x"] - current_x) > tolerance:
                # Start new paragraph
                paragraphs.append(current_paragraph)
                current_paragraph = [line]
                current_x = line[0]["x"]
            else:
                # Add to current paragraph
                current_paragraph.append(line)
                current_x = line[0]["x"]

        if current_paragraph:
            paragraphs.append(current_paragraph)

        return paragraphs

    def _identify_document_elements(
        self, paragraphs: List[List[List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Identify document elements"""
        elements = []

        for i, paragraph in enumerate(paragraphs):
            # Check if it's a heading (centered and short)
            if self._is_heading(paragraph):
                elements.append(
                    {
                        "type": "heading",
                        "level": self._determine_heading_level(paragraph),
                        "content": paragraph,
                        "index": i,
                    }
                )
            # Check if it's a list (left-aligned with consistent indentation)
            elif self._is_list(paragraph):
                elements.append({"type": "list", "content": paragraph, "index": i})
            # Check if it's a table (grid-like structure)
            elif self._is_table(paragraph):
                elements.append({"type": "table", "content": paragraph, "index": i})
            # Default to paragraph
            else:
                elements.append({"type": "paragraph", "content": paragraph, "index": i})

        return elements

    def _is_heading(self, paragraph: List[List[Dict[str, Any]]]) -> bool:
        """Check if paragraph is a heading"""
        # Simple heuristic: centered and short
        if len(paragraph) > 3:
            return False

        # Check if centered
        first_line = paragraph[0]
        last_line = paragraph[-1]

        if abs(first_line[0]["x"] - last_line[0]["x"]) < 20:
            return True

        return False

    def _determine_heading_level(self, paragraph: List[List[Dict[str, Any]]]) -> int:
        """Determine heading level"""
        # Simple heuristic based on font size (approximated by block height)
        avg_height = sum(line[0]["height"] for line in paragraph) / len(paragraph)

        if avg_height > 30:
            return 1
        elif avg_height > 20:
            return 2
        elif avg_height > 15:
            return 3
        else:
            return 4

    def _is_list(self, paragraph: List[List[Dict[str, Any]]]) -> bool:
        """Check if paragraph is a list"""
        # Simple heuristic: consistent indentation
        if len(paragraph) < 2:
            return False

        first_line = paragraph[0]
        second_line = paragraph[1]

        # Check if first line has bullet or number
        if first_line[0]["x"] < second_line[0]["x"]:
            return True

        return False

    def _is_table(self, paragraph: List[List[Dict[str, Any]]]) -> bool:
        """Check if paragraph is a table"""
        # Simple heuristic: grid-like structure
        if len(paragraph) < 2:
            return False

        # Check if lines are aligned in columns
        line_lengths = [len(line) for line in paragraph]

        if len(set(line_lengths)) == 1:
            return True

        return False


class OCRProcessor:
    """Main OCR processor"""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.preprocessor = OCRPreprocessor(config)
        self.postprocessor = OCRPostprocessor(config)
        self.layout_analyzer = LayoutAnalyzer(config)

        # Tesseract configuration
        self._setup_tesseract()

    def _setup_tesseract(self):
        """Setup Tesseract OCR"""
        try:
            # Set Tesseract path if available
            tesseract_path = os.getenv("TESSERACT_PATH")
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

            # Test Tesseract installation
            pytesseract.get_tesseract_version()

            self.logger.info("Tesseract OCR initialized successfully")

        except Exception as e:
            self.logger.error(f"Error setting up Tesseract OCR: {e}")
            raise

    def process_document(self, document_path: str) -> List[OCRResult]:
        """Process document with OCR"""
        start_time = time.time()

        try:
            # Determine document type
            doc_type = self._determine_document_type(document_path)

            if doc_type == DocumentType.PDF:
                return self._process_pdf(document_path)
            elif doc_type == DocumentType.IMAGE:
                return self._process_image(document_path)
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            raise

    def _determine_document_type(self, document_path: str) -> DocumentType:
        """Determine document type"""
        try:
            path = Path(document_path)

            if path.suffix.lower() == ".pdf":
                return DocumentType.PDF
            elif path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
                return DocumentType.IMAGE
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        except Exception as e:
            self.logger.error(f"Error determining document type: {e}")
            raise

    def _process_pdf(self, pdf_path: str) -> List[OCRResult]:
        """Process PDF document"""
        results = []

        try:
            # Convert PDF to images
            images = self._convert_pdf_to_images(pdf_path)

            # Process each page
            for i, image in enumerate(images):
                if i >= self.config.max_pages:
                    break

                result = self._process_image_page(image, page_number=i + 1)
                results.append(result)

            self.logger.info(f"Processed {len(results)} pages from PDF")

        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise

        return results

    def _process_image(self, image_path: str) -> List[OCRResult]:
        """Process image document"""
        try:
            # Load image
            image = Image.open(image_path)

            # Process image
            result = self._process_image_page(image, page_number=1)

            return [result]

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise

    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to images"""
        try:
            # Use pdf2image to convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.config.dpi,
                first_page=1,
                last_page=self.config.max_pages,
                thread_count=4,
            )

            return images

        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {e}")
            raise

    def _process_image_page(self, image: Image.Image, page_number: int) -> OCRResult:
        """Process single image page"""
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(image)

            # Analyze layout
            layout_info = self.layout_analyzer.analyze_layout(processed_image)

            # Perform OCR based on mode
            if self.config.mode == OCRMode.FULL_PAGE:
                text, confidence = self._ocr_full_page(processed_image)
            elif self.config.mode == OCRMode.BLOCK_BASED:
                text, confidence = self._ocr_block_based(processed_image, layout_info)
            elif self.config.mode == OCRMode.PARAGRAPH_BASED:
                text, confidence = self._ocr_paragraph_based(
                    processed_image, layout_info
                )
            elif self.config.mode == OCRMode.WORD_BASED:
                text, confidence = self._ocr_word_based(processed_image, layout_info)
            elif self.config.mode == OCRMode.LINE_BASED:
                text, confidence = self._ocr_line_based(processed_image, layout_info)
            else:
                text, confidence = self._ocr_full_page(processed_image)

            # Postprocess text
            text = self.postprocessor.postprocess_text(text)

            # Calculate processing time
            processing_time = time.time() - start_time

            return OCRResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                pages_processed=1,
                metadata={
                    "page_number": page_number,
                    "image_size": image.size,
                    "dpi": self.config.dpi,
                    "ocr_mode": self.config.mode.value,
                    "ocr_quality": self.config.quality.value,
                },
                layout_info=layout_info,
            )

        except Exception as e:
            self.logger.error(f"Error processing image page: {e}")
            raise

    def _ocr_full_page(self, image: Image.Image) -> Tuple[str, float]:
        """Perform OCR on full page"""
        try:
            # Configure Tesseract
            config = f"--oem 3 --psm 6 -l {self.config.language}"

            # Perform OCR
            text = pytesseract.image_to_string(
                image, config=config, timeout=self.config.timeout
            )

            # Get confidence
            data = pytesseract.image_to_data(
                image, config=config, timeout=self.config.timeout
            )

            # Calculate average confidence
            confidences = []
            for line in data.split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) > 10 and parts[10].isdigit():
                        confidences.append(float(parts[10]))

            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Error performing full page OCR: {e}")
            return "", 0.0

    def _ocr_block_based(
        self, image: Image.Image, layout_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Perform OCR based on text blocks"""
        try:
            if not layout_info.get("blocks"):
                return self._ocr_full_page(image)

            text_parts = []
            confidences = []

            # Process each block
            for block in layout_info["blocks"]:
                # Extract block region
                block_image = image.crop(
                    (
                        block["x"],
                        block["y"],
                        block["x"] + block["width"],
                        block["y"] + block["height"],
                    )
                )

                # Perform OCR on block
                block_text, block_confidence = self._ocr_full_page(block_image)

                if block_text.strip():
                    text_parts.append(block_text)
                    confidences.append(block_confidence)

            # Combine results
            text = "\n".join(text_parts)
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Error performing block-based OCR: {e}")
            return self._ocr_full_page(image)

    def _ocr_paragraph_based(
        self, image: Image.Image, layout_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Perform OCR based on paragraphs"""
        try:
            if not layout_info.get("paragraphs"):
                return self._ocr_block_based(image, layout_info)

            text_parts = []
            confidences = []

            # Process each paragraph
            for paragraph in layout_info["paragraphs"]:
                paragraph_text = []
                paragraph_confidences = []

                for line in paragraph:
                    line_text = []
                    line_confidences = []

                    for block in line:
                        # Extract block region
                        block_image = image.crop(
                            (
                                block["x"],
                                block["y"],
                                block["x"] + block["width"],
                                block["y"] + block["height"],
                            )
                        )

                        # Perform OCR on block
                        block_text, block_confidence = self._ocr_full_page(block_image)

                        if block_text.strip():
                            line_text.append(block_text)
                            line_confidences.append(block_confidence)

                    if line_text:
                        line_result = " ".join(line_text)
                        paragraph_text.append(line_result)
                        paragraph_confidences.append(
                            sum(line_confidences) / len(line_confidences)
                        )

                if paragraph_text:
                    paragraph_result = "\n".join(paragraph_text)
                    text_parts.append(paragraph_result)
                    confidences.append(
                        sum(paragraph_confidences) / len(paragraph_confidences)
                    )

            # Combine results
            text = "\n\n".join(text_parts)
            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Error performing paragraph-based OCR: {e}")
            return self._ocr_block_based(image, layout_info)

    def _ocr_word_based(
        self, image: Image.Image, layout_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Perform OCR based on words"""
        try:
            # Configure Tesseract for word-level OCR
            config = f"--oem 3 --psm 8 -l {self.config.language}"

            # Perform OCR
            text = pytesseract.image_to_string(
                image, config=config, timeout=self.config.timeout
            )

            # Get confidence
            data = pytesseract.image_to_data(
                image, config=config, timeout=self.config.timeout
            )

            # Calculate average confidence
            confidences = []
            for line in data.split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) > 10 and parts[10].isdigit():
                        confidences.append(float(parts[10]))

            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Error performing word-based OCR: {e}")
            return self._ocr_full_page(image)

    def _ocr_line_based(
        self, image: Image.Image, layout_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Perform OCR based on lines"""
        try:
            # Configure Tesseract for line-level OCR
            config = f"--oem 3 --psm 7 -l {self.config.language}"

            # Perform OCR
            text = pytesseract.image_to_string(
                image, config=config, timeout=self.config.timeout
            )

            # Get confidence
            data = pytesseract.image_to_data(
                image, config=config, timeout=self.config.timeout
            )

            # Calculate average confidence
            confidences = []
            for line in data.split("\n"):
                if line.strip():
                    parts = line.split("\t")
                    if len(parts) > 10 and parts[10].isdigit():
                        confidences.append(float(parts[10]))

            confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return text, confidence

        except Exception as e:
            self.logger.error(f"Error performing line-based OCR: {e}")
            return self._ocr_full_page(image)


class OCRSupport:
    """OCR support for the RAG system"""

    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize OCR processor
        self.ocr_processor = OCRProcessor(self.config)

        # Cache for processed documents
        self.cache = {}

        self.logger.info("OCR support initialized")

    def process_document(self, document_path: str) -> List[OCRResult]:
        """Process document with OCR"""
        try:
            # Check cache
            if self.config.cache_enabled:
                doc_hash = get_file_hash(document_path)
                if doc_hash in self.cache:
                    self.logger.info("Using cached OCR result")
                    return self.cache[doc_hash]

            # Process document
            results = self.ocr_processor.process_document(document_path)

            # Cache result
            if self.config.cache_enabled:
                self.cache[doc_hash] = results

            return results

        except Exception as e:
            self.logger.error(f"Error processing document with OCR: {e}")
            raise

    def get_text_chunks(self, document_path: str) -> List[Chunk]:
        """Get text chunks from OCR-processed document"""
        try:
            # Process document with OCR
            ocr_results = self.process_document(document_path)

            # Combine text from all pages
            full_text = ""
            for result in ocr_results:
                full_text += result.text + "\n\n"

            # Create chunks
            chunks = self._create_chunks_from_text(full_text)

            return chunks

        except Exception as e:
            self.logger.error(f"Error getting text chunks from OCR: {e}")
            raise

    def _create_chunks_from_text(self, text: str) -> List[Chunk]:
        """Create chunks from OCR text"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Tokenize sentences
            sentences = sent_tokenize(text)

            # Create chunks
            chunks = []
            current_chunk = ""
            current_tokens = 0

            for sentence in sentences:
                # Estimate tokens (rough approximation)
                sentence_tokens = len(sentence.split())

                if current_tokens + sentence_tokens > self.config.max_chunk_size:
                    # Add current chunk
                    if current_chunk.strip():
                        chunks.append(
                            Chunk(
                                id=len(chunks),
                                text=current_chunk.strip(),
                                metadata={"source": "ocr"},
                            )
                        )

                    # Start new chunk
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk += " " + sentence
                    current_tokens += sentence_tokens

            # Add final chunk
            if current_chunk.strip():
                chunks.append(
                    Chunk(
                        id=len(chunks),
                        text=current_chunk.strip(),
                        metadata={"source": "ocr"},
                    )
                )

            return chunks

        except Exception as e:
            self.logger.error(f"Error creating chunks from OCR text: {e}")
            raise

    def get_ocr_statistics(self, document_path: str) -> Dict[str, Any]:
        """Get OCR processing statistics"""
        try:
            # Process document with OCR
            ocr_results = self.process_document(document_path)

            # Calculate statistics
            total_pages = len(ocr_results)
            total_text = ""
            total_confidence = 0.0
            total_processing_time = 0.0

            for result in ocr_results:
                total_text += result.text
                total_confidence += result.confidence
                total_processing_time += result.processing_time

            avg_confidence = total_confidence / total_pages if total_pages > 0 else 0.0
            avg_processing_time = (
                total_processing_time / total_pages if total_pages > 0 else 0.0
            )

            return {
                "total_pages": total_pages,
                "total_characters": len(total_text),
                "total_words": len(total_text.split()),
                "average_confidence": avg_confidence,
                "average_processing_time": avg_processing_time,
                "total_processing_time": total_processing_time,
                "ocr_mode": self.config.mode.value,
                "ocr_quality": self.config.quality.value,
            }

        except Exception as e:
            self.logger.error(f"Error getting OCR statistics: {e}")
            raise

    def clear_cache(self):
        """Clear OCR cache"""
        self.cache.clear()
        self.logger.info("OCR cache cleared")

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status"""
        return {
            "cache_size": len(self.cache),
            "cache_enabled": self.config.cache_enabled,
            "cached_documents": list(self.cache.keys()),
        }


class OCRParser(BaseParser):
    """OCR parser for scanned documents"""

    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize OCR support
        self.ocr_support = OCRSupport(self.config)

        self.logger.info("OCR parser initialized")

    def parse_file(self, file_path: Path) -> List[Chunk]:
        """Parse file using OCR"""
        try:
            # Check if file is supported
            if not self._is_supported_file(file_path):
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            # Parse file
            chunks = self.ocr_support.get_text_chunks(str(file_path))

            self.logger.info(f"Created {len(chunks)} chunks from OCR parsing")

            return chunks

        except Exception as e:
            self.logger.error(f"Error parsing file with OCR: {e}")
            raise

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported for OCR"""
        supported_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        return file_path.suffix.lower() in supported_extensions

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information"""
        try:
            # Get OCR statistics
            stats = self.ocr_support.get_ocr_statistics(str(file_path))

            # Get file info
            file_stat = file_path.stat()

            return {
                "file_path": str(file_path),
                "file_size": file_stat.st_size,
                "file_extension": file_path.suffix.lower(),
                "ocr_statistics": stats,
                "supported": self._is_supported_file(file_path),
            }

        except Exception as e:
            self.logger.error(f"Error getting file info: {e}")
            return {"file_path": str(file_path), "error": str(e)}

    def is_scanned_document(self, file_path: Path) -> bool:
        """Check if document is scanned"""
        try:
            # For PDF files, check if they contain text
            if file_path.suffix.lower() == ".pdf":
                return self._is_scanned_pdf(file_path)

            # For image files, they are always scanned
            elif file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking if document is scanned: {e}")
            return False

    def _is_scanned_pdf(self, file_path: Path) -> bool:
        """Check if PDF is scanned"""
        try:
            # Try to extract text with PyPDF2
            with open(file_path, "rb") as file:
                reader = PdfReader(file)

                # Check first few pages for text content
                text_content = ""
                for i, page in enumerate(reader.pages[:5]):  # Check first 5 pages
                    if page.extract_text():
                        text_content += page.extract_text()

                # If very little text, it's likely scanned
                if len(text_content.strip()) < 100:
                    return True

                return False

        except Exception as e:
            self.logger.error(f"Error checking if PDF is scanned: {e}")
            return True  # Assume scanned if we can't determine


# Global OCR parser instance
ocr_parser = OCRParser()
