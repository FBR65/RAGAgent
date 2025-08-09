import logging
import json
from typing import List, Optional
from ..models import Chunk, AgentConfig, DocumentProcessingConfig, RoutingResult
from ..unified_client import ClientFactory, UnifiedClient

logger = logging.getLogger(__name__)


class DeepDiverAgent:
    """Agent responsible for hierarchical navigation through document chunks"""

    def __init__(
        self,
        config: AgentConfig,
        processing_config: DocumentProcessingConfig,
        client: Optional[UnifiedClient] = None,
    ):
        self.config = config
        self.processing_config = processing_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use provided client or create new one
        if client is not None:
            self.client = client
            self.owns_client = False  # Don't close shared client
            self.logger.info(f"DeepDiverAgent using shared client: {id(client)}")
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

    def navigate_to_paragraphs(
        self,
        question: str,
        document_chunks: List[Chunk],
        max_depth: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Navigate through document hierarchy to find most relevant paragraphs

        Args:
            question: User question
            document_chunks: Initial list of document chunks
            max_depth: Maximum navigation depth (default from config)

        Returns:
            List of most relevant chunks after navigation
        """
        if max_depth is None:
            max_depth = self.processing_config.max_navigation_depth

        self.logger.info(f"Starting deep navigation with max depth: {max_depth}")
        self.logger.info(f"Initial chunks: {len(document_chunks)}")

        current_chunks = document_chunks
        scratchpad = ""
        depth = 0

        while depth < max_depth:
            self.logger.info(
                f"Navigation depth {depth + 1}: Analyzing {len(current_chunks)} chunks"
            )

            try:
                # Debug: Zeige die aktuelle Navigation an
                print(
                    f"DEBUG: Navigation depth {depth + 1}: Analyzing {len(current_chunks)} chunks"
                )
                print(f"DEBUG: Question: {question}")
                for i, chunk in enumerate(
                    current_chunks[:2]
                ):  # Zeige nur ersten 2 chunks
                    print(f"DEBUG: Chunk {i}: {chunk.text[:100]}...")

                # Route to most relevant chunks
                routing_result = self._route_chunks(
                    question=question,
                    chunks=current_chunks,
                    depth=depth,
                    scratchpad=scratchpad,
                )

                selected_ids = routing_result.selected_ids
                scratchpad = routing_result.scratchpad

                # Get selected chunks
                selected_chunks = [
                    chunk for chunk in current_chunks if chunk.id in selected_ids
                ]

                self.logger.info(
                    f"Selected {len(selected_chunks)} chunks at depth {depth + 1}"
                )

                if not selected_chunks:
                    self.logger.warning("No chunks selected, stopping navigation")
                    break

                # Check if we should stop navigation
                if self._should_stop_navigation(selected_chunks):
                    self.logger.info("Navigation criteria met, stopping")
                    return selected_chunks

                # Split selected chunks further for deeper navigation
                current_chunks = self._split_chunks_further(selected_chunks)
                depth += 1

            except Exception as e:
                self.logger.error(f"Error at navigation depth {depth + 1}: {e}")
                break

        self.logger.info(f"Navigation completed after {depth} levels")
        self.logger.info(f"Final chunks: {len(current_chunks)}")

        return current_chunks

    def _should_stop_navigation(self, chunks: List[Chunk]) -> bool:
        """Determine if navigation should stop based on criteria"""

        # Check number of chunks
        if len(chunks) <= self.processing_config.max_initial_chunks // 2:
            self.logger.info("Stopping navigation: few chunks remaining")
            return True

        # Check total token count
        total_tokens = sum(chunk.token_count for chunk in chunks)
        if total_tokens < self.processing_config.min_chunk_size * 2:
            self.logger.info(f"Stopping navigation: low token count ({total_tokens})")
            return True

        # Check average chunk size
        avg_tokens = total_tokens / len(chunks)
        if avg_tokens < self.processing_config.min_chunk_size:
            self.logger.info(
                f"Stopping navigation: small average chunk size ({avg_tokens})"
            )
            return True

        return False

    def _split_chunks_further(self, chunks: List[Chunk]) -> List[Chunk]:
        """Split selected chunks into smaller paragraphs for deeper navigation"""

        sub_chunks = []

        for chunk in chunks:
            try:
                # Split by paragraphs
                paragraphs = self._split_into_paragraphs(chunk.text)

                for i, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Skip empty paragraphs
                        sub_chunk = self._create_sub_chunk(
                            paragraph=paragraph, parent_chunk=chunk, index=i
                        )
                        sub_chunks.append(sub_chunk)

            except Exception as e:
                self.logger.error(f"Error splitting chunk {chunk.id}: {e}")
                # Keep original chunk if splitting fails
                sub_chunks.append(chunk)

        self.logger.info(
            f"Split {len(chunks)} chunks into {len(sub_chunks)} sub-chunks"
        )
        return sub_chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        import re

        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean up each paragraph
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            # Remove leading/trailing whitespace
            paragraph = paragraph.strip()

            # Skip empty paragraphs
            if paragraph:
                # Further split by single newlines if paragraph is very long
                if len(paragraph) > 500:
                    line_paragraphs = re.split(r"\n", paragraph)
                    for line_para in line_paragraphs:
                        line_para = line_para.strip()
                        if line_para:
                            cleaned_paragraphs.append(line_para)
                else:
                    cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def _create_sub_chunk(
        self, paragraph: str, parent_chunk: Chunk, index: int
    ) -> Chunk:
        """Create a sub-chunk from a paragraph"""

        try:
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")
            token_count = len(tokenizer.encode(paragraph))
        except ImportError:
            # Fallback to word count
            token_count = len(paragraph.split())

        return Chunk(
            id=f"{parent_chunk.id}.{index}",
            text=paragraph,
            token_count=token_count,
            document_type=parent_chunk.document_type,
            metadata={
                **parent_chunk.metadata,
                "parent_id": parent_chunk.id,
                "is_sub_chunk": True,
                "sub_chunk_index": index,
            },
            parent_id=parent_chunk.id,
        )

    def hierarchical_search(
        self, question: str, document_chunks: List[Chunk], strategy: str = "balanced"
    ) -> List[Chunk]:
        """
        Perform hierarchical search with different strategies

        Args:
            question: User question
            document_chunks: Initial document chunks
            strategy: Search strategy ("balanced", "breadth_first", "depth_first")

        Returns:
            List of relevant chunks
        """

        if strategy == "breadth_first":
            return self._breadth_first_search(question, document_chunks)
        elif strategy == "depth_first":
            return self._depth_first_search(question, document_chunks)
        else:  # balanced
            return self.navigate_to_paragraphs(question, document_chunks)

    def _breadth_first_search(
        self, question: str, document_chunks: List[Chunk]
    ) -> List[Chunk]:
        """Breadth-first search strategy"""

        current_level = document_chunks
        scratchpad = ""
        depth = 0
        max_depth = self.processing_config.max_navigation_depth

        while depth < max_depth and len(current_level) > 1:
            self.logger.info(f"BFS Level {depth + 1}: {len(current_level)} chunks")

            try:
                routing_result = self._route_chunks(
                    question=question,
                    chunks=current_level,
                    depth=depth,
                    scratchpad=scratchpad,
                )

                selected_ids = routing_result.selected_ids
                scratchpad = routing_result.scratchpad

                selected_chunks = [
                    chunk for chunk in current_level if chunk.id in selected_ids
                ]

                if len(selected_chunks) == 1:
                    return selected_chunks

                current_level = self._split_chunks_further(selected_chunks)
                depth += 1

            except Exception as e:
                self.logger.error(f"Error in BFS level {depth + 1}: {e}")
                break

        return current_level

    def _depth_first_search(
        self, question: str, document_chunks: List[Chunk]
    ) -> List[Chunk]:
        """Depth-first search strategy"""

        stack = [(document_chunks, 0, "")]
        best_chunks = []
        best_score = 0

        while stack:
            current_chunks, depth, scratchpad = stack.pop()

            if depth >= self.processing_config.max_navigation_depth:
                continue

            try:
                routing_result = self._route_chunks(
                    question=question,
                    chunks=current_chunks,
                    depth=depth,
                    scratchpad=scratchpad,
                )

                selected_ids = routing_result.selected_ids
                scratchpad = routing_result.scratchpad

                selected_chunks = [
                    chunk for chunk in current_chunks if chunk.id in selected_ids
                ]

                # Calculate relevance score (simple: number of chunks)
                score = len(selected_chunks)

                if score > best_score:
                    best_score = score
                    best_chunks = selected_chunks

                # Push sub-chunks to stack for deeper exploration
                sub_chunks = self._split_chunks_further(selected_chunks)
                stack.append((sub_chunks, depth + 1, scratchpad))

            except Exception as e:
                self.logger.error(f"Error in DFS at depth {depth}: {e}")
                continue

        return best_chunks

    def close(self):
        """Close the client if we own it"""
        if self.owns_client:
            self.client.close_sync()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _route_chunks(
        self, question: str, chunks: List[Chunk], depth: int = 0, scratchpad: str = ""
    ) -> "RoutingResult":
        """Simple routing implementation for chunk selection"""

        # Debug: Zeige die Routing-Anfrage an
        print(f"DEBUG: Routing chunks for question: {question}")
        print(f"DEBUG: Available chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"DEBUG: Chunk {chunk.id}: {chunk.text[:50]}...")

        # Prepare chunk previews
        chunks_text = ""
        for chunk in chunks:
            preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            chunks_text += f"Chunk {chunk.id}: {preview}\n\n"

        # Build messages
        messages = [
            {
                "role": "system",
                "content": """You are a document navigation expert. Select the most relevant chunk IDs for the user's question.

IMPORTANT:
- Select chunks that contain keywords from the user's question
- Choose chunks that directly answer the question
- Respond with valid JSON format: {"selected_ids": ["chunk_id_1", "chunk_id_2"]}
- Always select at least one chunk

Example:
{"selected_ids": ["tmp0fjj9uaf_0"]}""",
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nChunks:\n{chunks_text}\n\nSelect the most relevant chunk IDs that directly answer the question. Respond with JSON format: {{'selected_ids': ['chunk_id_1', 'chunk_id_2']}}",
            },
        ]

        # Debug: Zeige die Routing-Nachrichten an
        print(f"DEBUG: Routing messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content'][:200]}...")

        # Get response from client
        response = self.client.chat_completion_sync(messages=messages)

        # Extract content from OpenAI-compatible response format
        content = ""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]

        # Fallback for other response formats
        if not content:
            content = response.get("message", {}).get("content", "{}")

        # Clean up qwen3 think tags
        if content and "<think>" in content:
            import re

            # Remove <think>...</think> tags
            content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
            content = content.strip()
        print(f"DEBUG: Raw routing response: '{content}'")

        # Initialize result as default fallback
        result = {"selected_ids": []}

        if isinstance(content, str):
            try:
                result = json.loads(content)
                self.logger.debug(f"Parsed routing result: {result}")
            except json.JSONDecodeError:
                # Fallback: try to extract IDs from text
                import re

                # Try to extract JSON from text
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                        self.logger.debug(f"Extracted routing result: {result}")
                    except json.JSONDecodeError:
                        # Look for various ID patterns
                        ids = re.findall(r'[\'"]?([a-zA-Z0-9_]+\.?\d+)[\'"]?', content)
                        if not ids and chunks:
                            ids = [chunks[0].id]  # Use first chunk ID
                        result = {"selected_ids": ids}
                        self.logger.warning(f"Extracted IDs from text: {ids}")
                else:
                    # Look for various ID patterns
                    ids = re.findall(r'[\'"]?([a-zA-Z0-9_]+\.?\d+)[\'"]?', content)
                    if not ids and chunks:
                        ids = [chunks[0].id]  # Use first chunk ID
                    result = {"selected_ids": ids}
                    self.logger.warning(f"Extracted IDs from text: {ids}")
        else:
            result = content if isinstance(content, dict) else {"selected_ids": []}

        selected_ids = result.get("selected_ids", [])
        self.logger.debug(f"Selected IDs: {selected_ids}")

        # Ensure we always select at least one chunk
        if not selected_ids and chunks:
            selected_ids = [chunks[0].id]
            self.logger.warning(
                f"No chunks selected, defaulting to first chunk: {selected_ids}"
            )

        return RoutingResult(
            selected_ids=selected_ids,
            scratchpad=f"Selected chunks {selected_ids} at depth {depth}",
            reasoning=f"Selected {len(selected_ids)} chunks at depth {depth}",
        )
