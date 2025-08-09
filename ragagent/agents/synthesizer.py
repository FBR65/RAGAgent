import json
import logging
from typing import List, Dict, Any, Optional
from ..models import Chunk, AnswerWithCitations, AgentConfig
from ..openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class AnswerSynthesizerAgent:
    """Agent responsible for generating structured answers with citations"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use OpenAI-compatible client for both OpenAI and Ollama
        self.client = OpenAIClient(config)

    def generate_answer(
        self, question: str, paragraphs: List[Chunk], scratchpad: str = ""
    ) -> AnswerWithCitations:
        """
        Generate a structured answer from retrieved paragraphs

        Args:
            question: User question
            paragraphs: List of relevant paragraphs/chunks
            scratchpad: Previous reasoning context

        Returns:
            AnswerWithCitations containing answer, citations, and confidence score
        """
        try:
            self.logger.info(f"Generating answer for question: {question}")
            self.logger.info(f"Using {len(paragraphs)} paragraphs for synthesis")

            # Prepare context string with paragraph IDs
            context = self._prepare_context(paragraphs)

            # Build messages for the LLM
            messages = self._build_synthesis_messages(question, context, scratchpad)

            # Generate structured answer
            answer = self._generate_structured_answer(messages)

            self.logger.info(f"Generated answer with {len(answer.citations)} citations")
            self.logger.info(f"Confidence score: {answer.confidence_score}")

            return answer

        except Exception as e:
            self.logger.error(f"Error in answer synthesis: {e}")
            raise

    def _prepare_context(self, paragraphs: List[Chunk]) -> str:
        """Prepare context string from paragraphs"""
        context = ""
        valid_ids = []

        for para in paragraphs:
            context += f"[ID: {para.id}]\n"
            context += f"{para.text}\n"
            context += f"Type: {para.document_type.value}\n"
            context += f"Tokens: {para.token_count}\n"
            context += "-" * 50 + "\n\n"

            valid_ids.append(str(para.id))

        self.logger.debug(f"Prepared context with {len(valid_ids)} valid IDs")
        return context

    def _build_synthesis_messages(
        self, question: str, context: str, scratchpad: str
    ) -> List[Dict[str, str]]:
        """Build messages for the synthesis LLM"""

        system_prompt = self._get_synthesis_system_prompt()

        user_prompt = f"""Frage: {question}

Verfügbare Kontextinformationen:
{context}

Antworte NUR mit gültigem JSON im Format:
{{
  "answer": "Deine Antwort hier",
  "citations": ["[ID: X]"],
  "confidence_score": 0.8
}}

WICHTIG:
- Verwende exakt die gleichen Begriffe aus dem Kontext
- Füge mindestens ein Zitat ein
- Gib eine Vertrauenswürdigkeit zwischen 0.0 und 1.0 an"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    def _get_synthesis_system_prompt(self) -> str:
        """Get the system prompt for answer synthesis"""
        return """Du bist ein KI-Forschungsexperte. Deine Aufgabe ist es, Fragen basierend auf den bereitgestellten Kontextinformationen zu beantworten.

WICHTIG:
1. Beantworte die Frage NUR mit Informationen aus dem Kontext
2. Füge IMMER mindestens ein Zitat im Format [ID: X] ein
3. Gib eine Vertrauenswürdigkeit von 0.0-1.0 an
4. Verwende exakt die gleichen Begriffe aus dem Kontext

Antworte immer in diesem exakten JSON Format:
{
  "answer": "Deine Antwort hier",
  "citations": ["[ID: 0]"],
  "confidence_score": 0.8
}

Beispiel:
Frage: Was sind Hauptarten des maschinellen Lernens?
Kontext: [ID: 0] Es gibt drei Hauptarten: überwachtes Lernen, unüberwartetes Lernen und Reinforcement Learning.
Antwort: {"answer": "Die Hauptarten des maschinellen Lernens sind überwachtes Lernen, unüberwartetes Lernen und Reinforcement Learning.", "citations": ["[ID: 0]"], "confidence_score": 1.0}"""

    def _generate_structured_answer(
        self, messages: List[Dict[str, str]]
    ) -> AnswerWithCitations:
        """Generate structured answer using the LLM"""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_with_citations",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Die strukturierte Antwort auf die Frage",
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Liste der zitierten Chunk-IDs im Format [ID: X]",
                        },
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Vertrauenswürdigkeitsbewertung der Antwort",
                        },
                    },
                    "required": ["answer", "citations", "confidence_score"],
                },
            },
        }

        try:
            # Debug: Zeige die Nachrichten an
            print(f"DEBUG: Sending messages to LLM:")
            for msg in messages:
                print(f"  {msg['role']}: {msg['content'][:200]}...")

            # Remove response_format to avoid issues with qwen3:latest
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
            )

            content = response.get("message", {}).get("content", "")

            # Log the raw response content
            print(f"DEBUG: Raw LLM response: '{content}'")
            print(f"DEBUG: Response type: {type(content)}")
            print(f"DEBUG: Response length: {len(content)}")

            # If response is empty, try with different approach
            if not content or content.strip() == "":
                print("DEBUG: Empty response, trying with simple text completion")
                # Try simple text completion instead
                simple_prompt = f"Beantworte diese Frage basierend auf dem Kontext: {messages[1]['content']}"
                response = self.client.chat_completion(
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.1,
                )
                content = response.get("message", {}).get("content", "")
                print(f"DEBUG: Simple response: '{content}'")

            # Handle different response formats
            if isinstance(content, str):
                if not content.strip():
                    # Handle empty response
                    print("DEBUG: Empty response, using fallback")
                    result = {
                        "answer": "Keine Antwort generiert.",
                        "citations": ["[ID: 0]"],
                        "confidence_score": 0.0,
                    }
                else:
                    try:
                        # Try to parse as JSON first
                        result = json.loads(content)
                        print(f"DEBUG: Parsed JSON result: {result}")
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decode error: {e}")
                        # Fallback: extract from text
                        import re

                        print(f"DEBUG: Failed to parse JSON, trying text extraction")

                        # Try to extract JSON from text
                        json_match = re.search(r"\{.*\}", content, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(0))
                                print(f"DEBUG: Extracted JSON result: {result}")
                            except json.JSONDecodeError:
                                print("DEBUG: Failed to extract JSON from text")
                                # Simple fallback - create a basic response
                                result = {
                                    "answer": content,
                                    "citations": ["[ID: 0]"],
                                    "confidence_score": 0.0,
                                }
                        else:
                            # Fallback: extract from text
                            print(
                                "DEBUG: No JSON found in response, using simple fallback"
                            )
                            # Create a basic response from the content
                            result = {
                                "answer": content,
                                "citations": ["[ID: 0]"],
                                "confidence_score": 0.0,
                            }
            else:
                result = content
                print(f"DEBUG: Using content as result: {result}")

            return AnswerWithCitations(
                answer=result.get("answer", ""),
                citations=result.get(
                    "citations", ["[ID: 0]"]
                ),  # Ensure at least one citation
                confidence_score=result.get("confidence_score", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Error generating structured answer: {e}")
            # Return fallback answer with citation
            return AnswerWithCitations(
                answer="Konnte keine Antwort generieren.",
                citations=["[ID: 0]"],  # Add a dummy citation
                confidence_score=0.0,
            )

    def generate_multiple_answers(
        self,
        question: str,
        paragraphs: List[Chunk],
        scratchpad: str = "",
        num_answers: int = 3,
    ) -> List[AnswerWithCitations]:
        """
        Generate multiple answers for the same question

        Args:
            question: User question
            paragraphs: List of relevant paragraphs
            scratchpad: Previous reasoning context
            num_answers: Number of answers to generate

        Returns:
            List of AnswerWithCitations objects
        """

        answers = []

        for i in range(num_answers):
            try:
                self.logger.info(f"Generating answer {i + 1}/{num_answers}")

                # Vary temperature slightly for different answers
                temperature = 0.1 + (i * 0.05)

                answer = self.generate_answer(question, paragraphs, scratchpad)
                answers.append(answer)

            except Exception as e:
                self.logger.error(f"Error generating answer {i + 1}: {e}")
                continue

        self.logger.info(f"Generated {len(answers)} answers")
        return answers

    def select_best_answer(
        self, answers: List[AnswerWithCitations]
    ) -> AnswerWithCitations:
        """
        Select the best answer from multiple generated answers

        Args:
            answers: List of AnswerWithCitations objects

        Returns:
            Best AnswerWithCitations object
        """

        if not answers:
            raise ValueError("No answers provided")

        if len(answers) == 1:
            return answers[0]

        # Score each answer based on multiple criteria
        scored_answers = []

        for answer in answers:
            score = self._score_answer(answer)
            scored_answers.append((answer, score))

        # Sort by score (descending)
        scored_answers.sort(key=lambda x: x[1], reverse=True)

        best_answer = scored_answers[0][0]
        self.logger.info(f"Selected best answer with score: {scored_answers[0][1]}")

        return best_answer

    def _score_answer(self, answer: AnswerWithCitations) -> float:
        """Score an answer based on multiple criteria"""

        score = 0.0

        # Confidence score (40%)
        score += answer.confidence_score * 0.4

        # Number of citations (30%)
        citation_score = min(len(answer.citations) / 5.0, 1.0)  # Normalize to 0-1
        score += citation_score * 0.3

        # Answer length (20%)
        length_score = min(len(answer.answer) / 200.0, 1.0)  # Normalize to 0-1
        score += length_score * 0.2

        # Citation quality (10%)
        if answer.citations:
            score += 0.1

        return min(score, 1.0)

    def validate_citations(
        self, answer: AnswerWithCitations, available_ids: List[str]
    ) -> bool:
        """
        Validate that all citations reference valid chunk IDs

        Args:
            answer: Answer to validate
            available_ids: List of available chunk IDs

        Returns:
            True if all citations are valid
        """

        invalid_citations = []

        for citation in answer.citations:
            if citation not in available_ids:
                invalid_citations.append(citation)

        if invalid_citations:
            self.logger.warning(f"Invalid citations found: {invalid_citations}")
            return False

        return True

    def close(self):
        """Close the client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
