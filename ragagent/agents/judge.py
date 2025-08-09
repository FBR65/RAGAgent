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

from ..openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class JudgeAgent:
    """Agent responsible for verifying answer accuracy and quality"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use OpenAI-compatible client for both OpenAI and Ollama
        self.client = OpenAIClient(config)

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

Sei gründlich, aber fair in deiner Bewertung."""

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
            response = self.client.chat_completion(
                messages=messages,
                response_format=response_format,
                temperature=0.0,  # Maximum consistency for verification
            )

            content = response.get("message", {}).get("content", "{}")
            result = json.loads(content)

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

    def close(self):
        """Close the client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
