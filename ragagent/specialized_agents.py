"""
Specialized agents for different domains (legal, medical, technical)
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .agents import DeepDiverAgent, AnswerSynthesizerAgent, JudgeAgent
from .models import Chunk, AnswerWithCitations, VerificationResult, AgentConfig
from .config import get_config

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Supported domain types"""

    LEGAL = "legal"
    MEDICAL = "medical"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    ACADEMIC = "academic"


@dataclass
class DomainKnowledge:
    """Domain-specific knowledge and guidelines"""

    domain: DomainType
    terminology: Dict[str, str]
    guidelines: List[str]
    citation_format: str
    verification_rules: List[str]
    chunking_strategy: str
    navigation_focus: List[str]


class SpecializedAgentConfig(AgentConfig):
    """Configuration for specialized agents"""

    domain: DomainType
    domain_knowledge: DomainKnowledge
    use_domain_specific_prompts: bool = True
    enable_citation_validation: bool = True
    apply_domain_filtering: bool = True


class BaseSpecializedAgent(ABC):
    """Base class for specialized agents"""

    def __init__(self, config: SpecializedAgentConfig):
        self.config = config
        self.domain_knowledge = config.domain_knowledge
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize base agents
        self.deep_diver = self._create_deep_diver()
        self.synthesizer = self._create_synthesizer()
        self.judge = self._create_judge()

    @abstractmethod
    def _create_deep_diver(self) -> DeepDiverAgent:
        """Create specialized deep diver agent"""
        pass

    @abstractmethod
    def _create_synthesizer(self) -> AnswerSynthesizerAgent:
        """Create specialized synthesizer agent"""
        pass

    @abstractmethod
    def _create_judge(self) -> JudgeAgent:
        """Create specialized judge agent"""
        pass

    @abstractmethod
    def get_domain_prompt(self, base_prompt: str) -> str:
        """Add domain-specific context to prompts"""
        pass

    @abstractmethod
    def validate_domain_citations(
        self, citations: List[str], chunks: List[Chunk]
    ) -> List[str]:
        """Validate citations are appropriate for the domain"""
        pass

    @abstractmethod
    def apply_domain_filtering(self, chunks: List[Chunk], question: str) -> List[Chunk]:
        """Apply domain-specific filtering to chunks"""
        pass

    def process_question(self, question: str, chunks: List[Chunk]) -> Dict[str, Any]:
        """Process question using domain-specific approach"""

        # Apply domain filtering
        if self.config.apply_domain_filtering:
            chunks = self.apply_domain_filtering(chunks, question)
            self.logger.info(f"Domain filtering reduced chunks to {len(chunks)}")

        # Navigate to relevant chunks
        relevant_chunks = self.deep_diver.navigate_to_paragraphs(question, chunks)

        # Generate answer
        answer = self.synthesizer.generate_answer(question, relevant_chunks)

        # Validate domain-specific citations
        if self.config.enable_citation_validation:
            valid_citations = self.validate_domain_citations(
                answer.citations, relevant_chunks
            )
            answer.citations = valid_citations

        # Verify answer
        verification = self.judge.verify_answer(question, answer, relevant_chunks)

        return {
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "confidence_score": answer.confidence_score,
            "verification": verification.dict() if verification else None,
            "domain": self.config.domain.value,
            "chunks_processed": len(relevant_chunks),
        }


class LegalAgent(BaseSpecializedAgent):
    """Specialized agent for legal documents"""

    def __init__(self, config: SpecializedAgentConfig):
        # Legal domain knowledge
        legal_knowledge = DomainKnowledge(
            domain=DomainType.LEGAL,
            terminology={
                "statute": "law passed by legislature",
                "regulation": "rule created by administrative agency",
                "precedent": "previous court decision that guides current cases",
                "jurisdiction": "court's authority to hear a case",
                "plaintiff": "party who initiates a lawsuit",
                "defendant": "party against whom a lawsuit is brought",
            },
            guidelines=[
                "Always cite specific legal provisions (statutes, regulations, cases)",
                "Distinguish between binding and persuasive authority",
                "Consider the jurisdiction and applicable law",
                "Identify the level of judicial review",
                "Check for recent amendments or new cases",
            ],
            citation_format="[Case: Smith v. Jones, 2023] [Statute: 15 U.S.C. § 1] [Reg: 37 C.F.R. § 1.78]",
            verification_rules=[
                "Verify all legal citations are accurate and complete",
                "Check that legal conclusions follow from cited authorities",
                "Ensure statutory interpretation is consistent with legislative intent",
                "Confirm jurisdictional requirements are met",
            ],
            chunking_strategy="preserve legal sections and subsections",
            navigation_focus=[
                "legal provisions",
                "case law",
                "regulatory requirements",
                "procedural rules",
            ],
        )

        super().__init__(
            config
            or SpecializedAgentConfig(
                domain=DomainType.LEGAL,
                domain_knowledge=legal_knowledge,
                model_name="qwen2.5:latest",
                base_url="http://localhost:11434",
                temperature=0.1,
                max_tokens=4000,
                timeout=30,
            )
        )

    def _create_deep_diver(self) -> DeepDiverAgent:
        """Create legal-focused deep diver"""
        config = self.config
        config.document_processing.max_navigation_depth = (
            4  # Legal docs often need deeper navigation
        )
        return DeepDiverAgent(config, config.document_processing)

    def _create_synthesizer(self) -> AnswerSynthesizerAgent:
        """Create legal-focused synthesizer"""
        return AnswerSynthesizerAgent(self.config)

    def _create_judge(self) -> JudgeAgent:
        """Create legal-focused judge"""
        return JudgeAgent(self.config)

    def get_domain_prompt(self, base_prompt: str) -> str:
        """Add legal context to prompts"""
        legal_context = f"""
        LEGAL CONTEXT:
        - Domain: Legal Analysis
        - Citation Format: {self.domain_knowledge.citation_format}
        - Key Guidelines: {"; ".join(self.domain_knowledge.guidelines)}
        
        TERMINOLOGY:
        {chr(10).join(f"- {term}: {defn}" for term, defn in self.domain_knowledge.terminology.items())}
        
        {base_prompt}
        """
        return legal_context

    def validate_domain_citations(
        self, citations: List[str], chunks: List[Chunk]
    ) -> List[str]:
        """Validate legal citations"""
        valid_citations = []

        for citation in citations:
            # Check if citation follows legal format patterns
            if any(
                pattern in citation.lower()
                for pattern in ["case:", "statute:", "reg:", "§"]
            ):
                # Verify the cited chunk exists
                if any(
                    str(chunk.id) == citation.split(":")[1].strip() for chunk in chunks
                ):
                    valid_citations.append(citation)

        self.logger.info(
            f"Legal citation validation: {len(valid_citations)}/{len(citations)} valid"
        )
        return valid_citations

    def apply_domain_filtering(self, chunks: List[Chunk], question: str) -> List[Chunk]:
        """Apply legal-specific filtering"""
        legal_keywords = [
            "law",
            "statute",
            "regulation",
            "case",
            "court",
            "judge",
            "jury",
            "plaintiff",
            "defendant",
            "contract",
            "agreement",
            "liability",
            "jurisdiction",
            "precedent",
            "appeal",
            "motion",
            "brief",
        ]

        # Score chunks based on legal keyword density
        scored_chunks = []
        for chunk in chunks:
            score = sum(
                1 for keyword in legal_keywords if keyword.lower() in chunk.text.lower()
            )
            scored_chunks.append((chunk, score))

        # Keep chunks with legal relevance
        filtered_chunks = [chunk for chunk, score in scored_chunks if score > 0]

        if not filtered_chunks:
            # If no legal chunks found, return all chunks
            return chunks

        return filtered_chunks


class MedicalAgent(BaseSpecializedAgent):
    """Specialized agent for medical documents"""

    def __init__(self, config: SpecializedAgentConfig):
        # Medical domain knowledge
        medical_knowledge = DomainKnowledge(
            domain=DomainType.MEDICAL,
            terminology={
                "diagnosis": "identification of disease or condition",
                "prognosis": "predicted course and outcome of disease",
                "treatment": "medical management of disease or condition",
                "symptom": "subjective evidence of disease or condition",
                "sign": "objective evidence of disease or condition",
                "etiology": "cause or origin of disease",
            },
            guidelines=[
                "Always distinguish between established medical facts and research findings",
                "Consider the strength of medical evidence (systematic reviews > RCTs > observational studies)",
                "Note the level of evidence for medical recommendations",
                "Consider patient-specific factors and contraindications",
                "Be aware of conflicts of interest in medical literature",
            ],
            citation_format="[Study: Smith et al., 2023, JAMA] [Guideline: AMA 2023] [Drug: FDA Label]",
            verification_rules=[
                "Verify all medical claims are supported by evidence",
                "Check that treatment recommendations are evidence-based",
                "Ensure diagnostic criteria are current and accurate",
                "Confirm dosage and administration information is correct",
            ],
            chunking_strategy="preserve medical sections and clinical guidelines",
            navigation_focus=[
                "diagnostic criteria",
                "treatment protocols",
                "drug information",
                "clinical guidelines",
            ],
        )

        super().__init__(
            config
            or SpecializedAgentConfig(
                domain=DomainType.MEDICAL,
                domain_knowledge=medical_knowledge,
                model_name="qwen2.5:latest",
                base_url="http://localhost:11434",
                temperature=0.05,  # Lower temperature for medical accuracy
                max_tokens=4000,
                timeout=30,
            )
        )

    def _create_deep_diver(self) -> DeepDiverAgent:
        """Create medical-focused deep diver"""
        config = self.config
        config.document_processing.max_navigation_depth = 3
        return DeepDiverAgent(config, config.document_processing)

    def _create_synthesizer(self) -> AnswerSynthesizerAgent:
        """Create medical-focused synthesizer"""
        return AnswerSynthesizerAgent(self.config)

    def _create_judge(self) -> JudgeAgent:
        """Create medical-focused judge"""
        return JudgeAgent(self.config)

    def get_domain_prompt(self, base_prompt: str) -> str:
        """Add medical context to prompts"""
        medical_context = f"""
        MEDICAL CONTEXT:
        - Domain: Medical Analysis
        - Citation Format: {self.domain_knowledge.citation_format}
        - Key Guidelines: {"; ".join(self.domain_knowledge.guidelines)}
        
        TERMINOLOGY:
        {chr(10).join(f"- {term}: {defn}" for term, defn in self.domain_knowledge.terminology.items())}
        
        {base_prompt}
        """
        return medical_context

    def validate_domain_citations(
        self, citations: List[str], chunks: List[Chunk]
    ) -> List[str]:
        """Validate medical citations"""
        valid_citations = []

        for citation in citations:
            # Check if citation follows medical format patterns
            if any(
                pattern in citation.lower()
                for pattern in ["study:", "guideline:", "drug:", "journal:"]
            ):
                # Verify the cited chunk exists
                if any(
                    str(chunk.id) == citation.split(":")[1].strip() for chunk in chunks
                ):
                    valid_citations.append(citation)

        self.logger.info(
            f"Medical citation validation: {len(valid_citations)}/{len(citations)} valid"
        )
        return valid_citations

    def apply_domain_filtering(self, chunks: List[Chunk], question: str) -> List[Chunk]:
        """Apply medical-specific filtering"""
        medical_keywords = [
            "patient",
            "diagnosis",
            "treatment",
            "symptom",
            "disease",
            "therapy",
            "medication",
            "dosage",
            "clinical",
            "study",
            "research",
            "evidence",
            "guideline",
            "protocol",
            "healthcare",
            "medical",
            "surgical",
        ]

        # Score chunks based on medical keyword density
        scored_chunks = []
        for chunk in chunks:
            score = sum(
                1
                for keyword in medical_keywords
                if keyword.lower() in chunk.text.lower()
            )
            scored_chunks.append((chunk, score))

        # Keep chunks with medical relevance
        filtered_chunks = [chunk for chunk, score in scored_chunks if score > 0]

        if not filtered_chunks:
            # If no medical chunks found, return all chunks
            return chunks

        return filtered_chunks


class TechnicalAgent(BaseSpecializedAgent):
    """Specialized agent for technical documents"""

    def __init__(self, config: SpecializedAgentConfig):
        # Technical domain knowledge
        technical_knowledge = DomainKnowledge(
            domain=DomainType.TECHNICAL,
            terminology={
                "api": "application programming interface",
                "sdk": "software development kit",
                "framework": "software platform for developing applications",
                "library": "collection of pre-written code",
                "algorithm": "step-by-step procedure for calculations",
                "architecture": "structural design of system",
            },
            guidelines=[
                "Always specify version numbers for software and APIs",
                "Distinguish between documentation and implementation details",
                "Consider the context of code examples and snippets",
                "Verify technical specifications against official documentation",
                "Note the difference between deprecated and current practices",
            ],
            citation_format="[API: REST v1.2] [Doc: Python 3.11] [Standard: RFC 7231]",
            verification_rules=[
                "Verify all technical specifications are current",
                "Check that code examples are syntactically correct",
                "Ensure API calls match documented format",
                "Confirm version compatibility information",
            ],
            chunking_strategy="preserve technical sections and code blocks",
            navigation_focus=[
                "api documentation",
                "code examples",
                "technical specifications",
                "implementation details",
            ],
        )

        super().__init__(
            config
            or SpecializedAgentConfig(
                domain=DomainType.TECHNICAL,
                domain_knowledge=technical_knowledge,
                model_name="qwen2.5:latest",
                base_url="http://localhost:11434",
                temperature=0.1,
                max_tokens=4000,
                timeout=30,
            )
        )

    def _create_deep_diver(self) -> DeepDiverAgent:
        """Create technical-focused deep diver"""
        config = self.config
        config.document_processing.max_navigation_depth = 3
        return DeepDiverAgent(config, config.document_processing)

    def _create_synthesizer(self) -> AnswerSynthesizerAgent:
        """Create technical-focused synthesizer"""
        return AnswerSynthesizerAgent(self.config)

    def _create_judge(self) -> JudgeAgent:
        """Create technical-focused judge"""
        return JudgeAgent(self.config)

    def get_domain_prompt(self, base_prompt: str) -> str:
        """Add technical context to prompts"""
        technical_context = f"""
        TECHNICAL CONTEXT:
        - Domain: Technical Analysis
        - Citation Format: {self.domain_knowledge.citation_format}
        - Key Guidelines: {"; ".join(self.domain_knowledge.guidelines)}
        
        TERMINOLOGY:
        {chr(10).join(f"- {term}: {defn}" for term, defn in self.domain_knowledge.terminology.items())}
        
        {base_prompt}
        """
        return technical_context

    def validate_domain_citations(
        self, citations: List[str], chunks: List[Chunk]
    ) -> List[str]:
        """Validate technical citations"""
        valid_citations = []

        for citation in citations:
            # Check if citation follows technical format patterns
            if any(
                pattern in citation.lower()
                for pattern in ["api:", "doc:", "standard:", "version:"]
            ):
                # Verify the cited chunk exists
                if any(
                    str(chunk.id) == citation.split(":")[1].strip() for chunk in chunks
                ):
                    valid_citations.append(citation)

        self.logger.info(
            f"Technical citation validation: {len(valid_citations)}/{len(citations)} valid"
        )
        return valid_citations

    def apply_domain_filtering(self, chunks: List[Chunk], question: str) -> List[Chunk]:
        """Apply technical-specific filtering"""
        technical_keywords = [
            "api",
            "documentation",
            "code",
            "implementation",
            "function",
            "class",
            "method",
            "parameter",
            "return",
            "exception",
            "library",
            "framework",
            "version",
            "compatibility",
            "deprecated",
            "recommended",
            "example",
        ]

        # Score chunks based on technical keyword density
        scored_chunks = []
        for chunk in chunks:
            score = sum(
                1
                for keyword in technical_keywords
                if keyword.lower() in chunk.text.lower()
            )
            scored_chunks.append((chunk, score))

        # Keep chunks with technical relevance
        filtered_chunks = [chunk for chunk, score in scored_chunks if score > 0]

        if not filtered_chunks:
            # If no technical chunks found, return all chunks
            return chunks

        return filtered_chunks


class SpecializedAgentFactory:
    """Factory for creating specialized agents"""

    @staticmethod
    def create_agent(
        domain: DomainType, config: SpecializedAgentConfig = None
    ) -> BaseSpecializedAgent:
        """Create a specialized agent for the given domain"""

        if domain == DomainType.LEGAL:
            return LegalAgent(config)
        elif domain == DomainType.MEDICAL:
            return MedicalAgent(config)
        elif domain == DomainType.TECHNICAL:
            return TechnicalAgent(config)
        else:
            raise ValueError(f"Unsupported domain: {domain}")

    @staticmethod
    def detect_domain(question: str, chunks: List[Chunk]) -> DomainType:
        """Detect the domain based on question and content"""

        # Simple keyword-based domain detection
        domain_keywords = {
            DomainType.LEGAL: [
                "law",
                "legal",
                "court",
                "case",
                "contract",
                "agreement",
                "statute",
            ],
            DomainType.MEDICAL: [
                "patient",
                "diagnosis",
                "treatment",
                "symptom",
                "disease",
                "therapy",
                "medical",
            ],
            DomainType.TECHNICAL: [
                "api",
                "code",
                "documentation",
                "implementation",
                "function",
                "software",
            ],
        }

        # Count domain keywords in question
        question_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(
                1 for keyword in keywords if keyword.lower() in question.lower()
            )
            question_scores[domain] = score

        # Count domain keywords in chunks
        chunk_scores = {domain: 0 for domain in domain_keywords}
        for chunk in chunks:
            for domain, keywords in domain_keywords.items():
                score = sum(
                    1 for keyword in keywords if keyword.lower() in chunk.text.lower()
                )
                chunk_scores[domain] += score

        # Combine scores (question weighted more heavily)
        combined_scores = {}
        for domain in domain_keywords:
            combined_scores[domain] = (question_scores[domain] * 2) + chunk_scores[
                domain
            ]

        # Return domain with highest score
        detected_domain = max(combined_scores, key=combined_scores.get)

        # Only return detected domain if score is significant
        if combined_scores[detected_domain] > 0:
            return detected_domain
        else:
            return DomainType.TECHNICAL  # Default to technical
