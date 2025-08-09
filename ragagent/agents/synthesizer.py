import json
import logging
from typing import List, Optional, Dict
from ..models import Chunk, AnswerWithCitations, AgentConfig
from ..unified_client import ClientFactory, UnifiedClient

logger = logging.getLogger(__name__)


class AnswerSynthesizerAgent:
    """Agent responsible for generating structured answers with citations"""

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

            # Bei Circuit Breaker Fehlern: Extrahiere echte Antwort aus Chunks
            if (
                "circuit breaker" in str(e).lower()
                and paragraphs
                and len(paragraphs) > 0
            ):
                self.logger.warning(
                    "Circuit Breaker Fehler - verwende direkte Chunk-Extraktion"
                )

                first_chunk = paragraphs[0]
                chunk_content = ""

                if hasattr(first_chunk, "content") and first_chunk.content:
                    chunk_content = first_chunk.content
                elif hasattr(first_chunk, "text") and first_chunk.text:
                    chunk_content = first_chunk.text
                else:
                    chunk_content = str(first_chunk)

                if chunk_content and len(chunk_content.strip()) > 10:
                    return AnswerWithCitations(
                        answer=f"**Direkte Antwort aus dem Dokument:**\n\n{chunk_content[:600]}{'...' if len(chunk_content) > 600 else ''}",
                        citations=[
                            chunk_content[:300] + "..."
                            if len(chunk_content) > 300
                            else chunk_content
                        ],
                        confidence_score=0.7,
                    )

            # Bei anderen Fehlern: Original Exception weiterwerfen
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
            response = self.client.chat_completion_sync(
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
            )

            # Extract content from OpenAI-compatible response format
            content = ""
            if "choices" in response and len(response["choices"]) > 0:
                choice = response["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]

            # Fallback for other response formats
            if not content:
                content = response.get("message", {}).get("content", "")

            # Clean up qwen3 think tags
            if content and "<think>" in content:
                import re

                # Remove <think>...</think> tags
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
                content = content.strip()

            # Log the raw response content
            print(f"DEBUG: Raw LLM response: '{content}'")
            print(f"DEBUG: Response type: {type(content)}")
            print(f"DEBUG: Response length: {len(content)}")

            # If response is empty, try with different approach
            if not content or content.strip() == "":
                print("DEBUG: Empty response, trying with simple text completion")
                # Try simple text completion instead
                simple_prompt = f"Beantworte diese Frage basierend auf dem Kontext: {messages[1]['content']}"
                response = self.client.chat_completion_sync(
                    messages=[{"role": "user", "content": simple_prompt}],
                    temperature=0.1,
                )
                # Extract content from OpenAI-compatible response format
                if "choices" in response and len(response["choices"]) > 0:
                    choice = response["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]

                # Fallback for other response formats
                if not content:
                    content = response.get("message", {}).get("content", "")

                # Clean up qwen3 think tags
                if content and "<think>" in content:
                    import re

                    # Remove <think>...</think> tags
                    content = re.sub(
                        r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
                    )
                    content = content.strip()
                print(f"DEBUG: Simple response: '{content}'")

            # Handle different response formats
            if isinstance(content, str):
                if not content.strip():
                    # Handle empty response
                    print("DEBUG: Empty response, using fallback")
                    result = {
                        "answer": "Keine Antwort generiert - LLM Response war leer.",
                        "citations": ["LLM-Fehler: Leere Antwort"],
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
                                    "citations": ["LLM-Fehler: JSON Parse Fehler"],
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
                                "citations": ["LLM-Fehler: Kein JSON im Response"],
                                "confidence_score": 0.0,
                            }
            else:
                result = content
                print(f"DEBUG: Using content as result: {result}")

            return AnswerWithCitations(
                answer=result.get("answer", ""),
                citations=result.get(
                    "citations", ["LLM-Fehler: Keine Citations vorhanden"]
                ),  # Ensure at least one citation
                confidence_score=result.get("confidence_score", 0.0),
            )

        except Exception as e:
            self.logger.error(f"Error generating structured answer: {e}")

            # Automatische Circuit Breaker Reparatur bei Circuit Breaker Fehlern
            if "circuit breaker is open" in str(e).lower():
                try:
                    self.logger.warning(
                        "Circuit Breaker Fehler erkannt - versuche automatische Reparatur"
                    )
                    # Reset Circuit Breaker über unified_client
                    if hasattr(self, "client") and hasattr(
                        self.client, "reset_circuit_breaker"
                    ):
                        self.client.reset_circuit_breaker()
                        self.logger.info("Circuit Breaker automatisch zurückgesetzt")

                    # Kurz warten und nochmal versuchen
                    import time

                    time.sleep(2)

                except Exception as reset_error:
                    self.logger.warning(
                        f"Circuit Breaker Reset fehlgeschlagen: {reset_error}"
                    )

            # Return informative fallback with actual error info
            return AnswerWithCitations(
                answer=f"LLM-Verarbeitungsfehler: {str(e)[:200]}...\n\nDas System hat relevante Dokumentabschnitte gefunden, aber die LLM-Antwortgenerierung ist fehlgeschlagen. Mögliche Ursachen:\n- Ollama ist überlastet oder nicht verfügbar\n- Netzwerk-Timeout\n- Modell qwen3:latest ist nicht geladen\n\n💡 Versuche es in 10-30 Sekunden erneut.",
                citations=[f"LLM-Fehler: {str(e)[:100]}"],
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
        """Close the client if we own it"""
        if self.owns_client:
            self.client.close_sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
