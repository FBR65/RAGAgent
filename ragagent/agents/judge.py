import json
import logging
from typing import List, Dict, Any, Optional
from ..models import (
    Chunk,
    AnswerWithCitations,
    VerificationResult,
    VerificationConfidence,
    AgentConfig,
)

from ..unified_client import ClientFactory, UnifiedClient

logger = logging.getLogger(__name__)


class JudgeAgent:
    """Agent responsible for verifying answer accuracy and quality"""

    def __init__(self, config: AgentConfig, client: Optional[UnifiedClient] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use provided client or create new one
        if client is not None:
            self.client = client
            self.owns_client = False  # Don't close shared client
        else:
            # Use UnifiedClient for both OpenAI and Ollama
            if (
                "localhost:11434" in config.base_url
                or "ollama" in config.base_url.lower()
            ):
                self.client = ClientFactory.create_ollama_client(
                    base_url=config.base_url,
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
            else:
                self.client = ClientFactory.create_openai_client(
                    base_url=config.base_url,
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=getattr(config, "api_key", None),
                )
            self.owns_client = True  # Close owned client

    def verify_answer(
        self, question: str, answer: AnswerWithCitations, cited_paragraphs: List[Chunk]
    ) -> VerificationResult:
        """
        Verify if the answer is grounded in the cited paragraphs

        Args:
            question: Original user question
            answer: Answer to verify
            cited_paragraphs: Paragraphs that were cited in the answer

        Returns:
            VerificationResult with accuracy assessment
        """
        try:
            self.logger.info(f"Verifying answer for question: {question}")
            self.logger.info(f"Verifying {len(answer.citations)} citations")

            # Prepare context string from cited paragraphs only
            context = self._prepare_verification_context(
                cited_paragraphs, answer.citations
            )

            # Build messages for the LLM
            messages = self._build_verification_messages(question, answer, context)

            # Generate verification result
            verification = self._generate_verification_result(messages)

            self.logger.info(
                f"Verification completed: {verification.is_accurate} "
                f"(Confidence: {verification.confidence})"
            )

            return verification

        except Exception as e:
            self.logger.error(f"Error in answer verification: {e}")
            # Return conservative verification result
            return VerificationResult(
                is_accurate=False,
                explanation=f"Verification failed: {str(e)}",
                confidence=VerificationConfidence.LOW,
                issues_found=["Verification process encountered an error"],
            )

    def _prepare_verification_context(
        self, paragraphs: List[Chunk], cited_ids: List[str]
    ) -> str:
        """Prepare context string from only the cited paragraphs"""
        context = ""

        for para in paragraphs:
            if str(para.id) in cited_ids:
                context += f"[ID: {para.id}]\n"
                context += f"{para.text}\n"
                context += "-" * 50 + "\n\n"

        self.logger.debug(
            f"Prepared verification context with {len(cited_ids)} cited paragraphs"
        )
        return context

    def _build_verification_messages(
        self, question: str, answer: AnswerWithCitations, context: str
    ) -> List[Dict[str, str]]:
        """Build messages for the verification LLM"""

        system_prompt = self._get_verification_system_prompt()

        user_prompt = f"""Frage: {question}

ZU VERIFIZIERENDE ANTWORT: {answer.answer}

VERWENDETE ZITATE: {", ".join(answer.citations)}

ORIGINAL VERTRAUENLICHKEIT: {answer.confidence_score}

QUELLPARAGRAPHEN:
{context}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    def _get_verification_system_prompt(self) -> str:
        """Get the system prompt for verification"""
        return """Du bist ein Faktenprüfer für juristische Informationen.
Deine Aufgabe ist es zu überprüfen, ob die bereitgestellte Antwort:
    
1. Tatsächlich korrekt gemäß den Quellparagraphen ist
2. Zitate korrekt verwendet (jede Aussage sollte unterstützt werden)
3. keine Informationen erfindet, die nicht in den Quellen stehen
4. Angemessene Vertrauenswürdigkeitsstufen beibehält

Bewerte die Vertrauenswürdigkeit als:
- HOCH: Antwort wird durch klare, explizite Aussagen direkt unterstützt
- MITTEL: Antwort erfordert vernünftige Schlussfolgerungen aus den Quellen
- NIEDRIG: Antwort wird schlecht unterstützt oder enthält nicht unterstützte Behauptungen

WICHTIG: Antworte NUR mit einem validen JSON-Objekt in diesem exakten Format:
{
  "is_accurate": true/false,
  "explanation": "Deine detaillierte Erklärung",
  "confidence": "high/medium/low",
  "issues_found": ["Liste", "von", "Problemen"]
}

Schreibe keinen anderen Text - nur das JSON-Objekt."""

    def _generate_verification_result(
        self, messages: List[Dict[str, str]]
    ) -> VerificationResult:
        """Generate verification result using the LLM"""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "verification_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_accurate": {
                            "type": "boolean",
                            "description": "Ob die Antwort faktisch korrekt ist",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Erklärung der Bewertung",
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Vertrauenswürdigkeitsstufe der Bewertung",
                        },
                        "issues_found": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Liste der gefundenen Probleme",
                        },
                    },
                    "required": [
                        "is_accurate",
                        "explanation",
                        "confidence",
                        "issues_found",
                    ],
                },
            },
        }

        try:
            response = self.client.chat_completion_sync(
                messages=messages,
                response_format=response_format,
                temperature=0.0,  # Maximum consistency for verification
            )

            # Extract content from OpenAI-compatible response format
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Fallback for other response formats
            if not content:
                content = response.get("message", {}).get("content", "{}")

            # Debug logging
            self.logger.debug(f"Raw verification response: {repr(content)}")

            # Check if content is empty or None
            if not content or content.strip() == "":
                self.logger.warning("Received empty response from verification LLM")
                content = '{"is_accurate": false, "explanation": "Empty response from LLM", "confidence": "low", "issues_found": ["Empty LLM response"]}'

            # Clean up qwen3 think tags
            if content and "<think>" in content:
                import re

                # Remove <think>...</think> tags
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = content.strip()

            # Final check after cleaning
            if not content or content.strip() == "":
                self.logger.warning("Content is empty after cleaning think tags")
                content = '{"is_accurate": false, "explanation": "Empty response after cleaning", "confidence": "low", "issues_found": ["Empty response after processing"]}'

            try:
                result = json.loads(content)
            except json.JSONDecodeError as json_error:
                self.logger.error(f"JSON decode error: {json_error}")
                self.logger.error(f"Problematic content: {repr(content)}")

                # Try to extract JSON from text using smart extraction
                result = self._extract_json_from_text(content)
                if result:
                    self.logger.info(
                        "Successfully extracted JSON from natural language response"
                    )
                else:
                    # Generate fallback result from natural language analysis
                    result = self._analyze_natural_language_response(content)
                    self.logger.info(
                        "Generated fallback verification result from natural language analysis"
                    )

            return VerificationResult(
                is_accurate=result.get("is_accurate", False),
                explanation=result.get("explanation", ""),
                confidence=VerificationConfidence(result.get("confidence", "low")),
                issues_found=result.get("issues_found", []),
            )

        except Exception as e:
            self.logger.error(f"Error generating verification result: {e}")
            # Return conservative verification result
            return VerificationResult(
                is_accurate=False,
                explanation=f"Verification failed: {str(e)}",
                confidence=VerificationConfidence.LOW,
                issues_found=["Verification process encountered an error"],
            )

    def comprehensive_verification(
        self, question: str, answer: AnswerWithCitations, all_paragraphs: List[Chunk]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive verification including citation validation

        Args:
            question: Original user question
            answer: Answer to verify
            all_paragraphs: All paragraphs that were available for synthesis

        Returns:
            Dictionary with comprehensive verification results
        """

        # Basic verification
        verification = self.verify_answer(question, answer, all_paragraphs)

        # Citation validation
        citation_issues = self._validate_citations(answer, all_paragraphs)

        # Completeness check
        completeness_issues = self._check_completeness(question, answer, all_paragraphs)

        # Consistency check
        consistency_issues = self._check_consistency(answer)

        # Combine all issues
        all_issues = (
            verification.issues_found
            + citation_issues
            + completeness_issues
            + consistency_issues
        )

        # Update verification with additional issues
        verification.issues_found = all_issues

        # Recalculate confidence based on all issues
        verification.confidence = self._recalculate_confidence(verification, all_issues)

        return {
            "verification": verification,
            "citation_issues": citation_issues,
            "completeness_issues": completeness_issues,
            "consistency_issues": consistency_issues,
            "total_issues": len(all_issues),
        }

    def _validate_citations(
        self, answer: AnswerWithCitations, all_paragraphs: List[Chunk]
    ) -> List[str]:
        """Validate that all citations are properly supported"""

        issues = []
        cited_paragraphs = {}

        # Build map of cited paragraphs
        for para in all_paragraphs:
            if str(para.id) in answer.citations:
                cited_paragraphs[str(para.id)] = para

        # Check each citation
        for citation in answer.citations:
            if citation not in cited_paragraphs:
                issues.append(f"Zitat {citation} existiert nicht in den Quellen")
            else:
                para = cited_paragraphs[citation]
                # Check if the paragraph actually supports the answer
                if not self._paragraph_supports_answer(para, answer.answer):
                    issues.append(
                        f"Zitat {citation} unterstützt die Antwort nicht ausreichend"
                    )

        return issues

    def _check_completeness(
        self, question: str, answer: AnswerWithCitations, all_paragraphs: List[Chunk]
    ) -> List[str]:
        """Check if the answer addresses all aspects of the question"""

        issues = []

        # Simple completeness check - could be enhanced with more sophisticated analysis
        if len(answer.answer) < 50:
            issues.append("Antwort ist möglicherweise unvollständig")

        # Check if key terms from question are addressed
        question_terms = set(question.lower().split())
        answer_terms = set(answer.answer.lower().split())

        if not question_terms.intersection(answer_terms):
            issues.append("Antwort scheint nicht direkt auf die Frage zu antworten")

        return issues

    def _check_consistency(self, answer: AnswerWithCitations) -> List[str]:
        """Check for internal consistency in the answer"""

        issues = []

        # Check for contradictory statements
        sentences = answer.answer.split(".")
        for i, sentence in enumerate(sentences):
            for j, other_sentence in enumerate(sentences[i + 1 :], i + 1):
                if self._are_sentences_contradictory(
                    sentence.strip(), other_sentence.strip()
                ):
                    issues.append("Widersprüchliche Aussagen in der Antwort")
                    break

        return issues

    def _paragraph_supports_answer(self, paragraph: Chunk, answer: str) -> bool:
        """Check if a paragraph supports the given answer"""

        # Simple keyword overlap check
        paragraph_words = set(paragraph.text.lower().split())
        answer_words = set(answer.lower().split())

        # Calculate overlap
        overlap = len(paragraph_words.intersection(answer_words))
        total_words = len(answer_words)

        # Consider it supporting if at least 30% of answer words are in paragraph
        return overlap >= max(1, total_words * 0.3)

    def _are_sentences_contradictory(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences are contradictory"""

        # Simple contradiction detection - could be enhanced
        contradiction_words = ["nicht", "kein", "nie", "verneinend", "verneint"]

        has_contradiction1 = any(
            word in sentence1.lower() for word in contradiction_words
        )
        has_contradiction2 = any(
            word in sentence2.lower() for word in contradiction_words
        )

        return has_contradiction1 and has_contradiction2

    def _recalculate_confidence(
        self, verification: VerificationResult, additional_issues: List[str]
    ) -> VerificationConfidence:
        """Recalculate confidence based on additional issues"""

        # Base confidence from verification
        base_confidence = verification.confidence

        # Adjust based on number of additional issues
        issue_count = len(additional_issues)

        if issue_count >= 3:
            return VerificationConfidence.LOW
        elif issue_count >= 1:
            if base_confidence == VerificationConfidence.HIGH:
                return VerificationConfidence.MEDIUM
            else:
                return VerificationConfidence.LOW
        else:
            return base_confidence

    def should_accept_answer(
        self, verification: VerificationResult, max_issues: int = 2
    ) -> bool:
        """
        Determine if an answer should be accepted based on verification

        Args:
            verification: Verification result
            max_issues: Maximum number of acceptable issues

        Returns:
            True if answer should be accepted
        """

        # Check if answer is accurate
        if not verification.is_accurate:
            return False

        # Check confidence level
        if verification.confidence == VerificationConfidence.LOW:
            return False

        # Check number of issues
        if len(verification.issues_found) > max_issues:
            return False

        return True

    def _extract_json_from_text(self, content: str) -> Dict[str, Any]:
        """Try to extract JSON from natural language text"""
        import re

        # Look for JSON-like structures in the text
        json_patterns = [
            r'\{[^{}]*"is_accurate"[^{}]*\}',  # Simple single-line JSON
            r'\{.*?"is_accurate".*?\}',  # Multi-line JSON
            r"```json\s*(\{.*?\})\s*```",  # JSON in code blocks
            r"```\s*(\{.*?\})\s*```",  # JSON in any code block
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    # If the pattern captured a group, use that
                    json_text = match if isinstance(match, str) else match[0]
                    result = json.loads(json_text)
                    if "is_accurate" in result:
                        return result
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    def _analyze_natural_language_response(self, content: str) -> Dict[str, Any]:
        """Analyze natural language response to extract verification information"""
        content_lower = content.lower()

        # Determine accuracy
        positive_indicators = [
            "korrekt",
            "richtig",
            "accurate",
            "correct",
            "stimmt",
            "zutreffend",
            "vertrauenswürdig",
            "reliable",
            "präzise",
            "accurate",
        ]
        negative_indicators = [
            "falsch",
            "incorrect",
            "wrong",
            "ungenau",
            "inaccurate",
            "fehler",
            "error",
            "problematisch",
            "unreliable",
            "unzuverlässig",
        ]

        # Count positive and negative indicators
        positive_count = sum(
            1 for indicator in positive_indicators if indicator in content_lower
        )
        negative_count = sum(
            1 for indicator in negative_indicators if indicator in content_lower
        )

        # Determine accuracy based on indicators
        if positive_count > negative_count:
            is_accurate = True
        elif negative_count > positive_count:
            is_accurate = False
        else:
            # Default to conservative approach
            is_accurate = False

        # Determine confidence
        high_conf_indicators = [
            "definitiv",
            "sicher",
            "eindeutig",
            "clear",
            "obviously",
            "certainly",
        ]
        medium_conf_indicators = [
            "wahrscheinlich",
            "likely",
            "vermutlich",
            "scheint",
            "appears",
        ]
        low_conf_indicators = [
            "unsicher",
            "unclear",
            "unklar",
            "möglicherweise",
            "potentially",
        ]

        if any(indicator in content_lower for indicator in high_conf_indicators):
            confidence = "high"
        elif any(indicator in content_lower for indicator in medium_conf_indicators):
            confidence = "medium"
        elif any(indicator in content_lower for indicator in low_conf_indicators):
            confidence = "low"
        else:
            # Default based on content length and detail
            confidence = "medium" if len(content) > 200 else "low"

        # Extract issues
        issues_found = []
        if "fehler" in content_lower or "error" in content_lower:
            issues_found.append("Potential errors mentioned in analysis")
        if "unvollständig" in content_lower or "incomplete" in content_lower:
            issues_found.append("Incomplete information noted")
        if "patent" in content_lower and "nicht" in content_lower:
            issues_found.append("Patent-related limitations mentioned")

        # Add fallback indicator
        issues_found.append("Derived from natural language analysis")

        # Create explanation
        explanation = f"Analysis derived from natural language response. Accuracy: {is_accurate}, Confidence: {confidence}"
        if len(content) > 100:
            explanation += f". Key content: {content[:100]}..."

        return {
            "is_accurate": is_accurate,
            "explanation": explanation,
            "confidence": confidence,
            "issues_found": issues_found,
        }

    def close(self):
        """Close the client if we own it"""
        if self.owns_client:
            self.client.close_sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
