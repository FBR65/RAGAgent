"""
Learning system with user preferences and adaptive behavior
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, deque
import pickle
import uuid

from .models import Chunk, AnswerWithCitations, ProcessingResponse
from .config import get_config
from .utils import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class UserPreference:
    """User preference data structure"""

    user_id: str
    preference_id: str
    name: str
    value: Any
    confidence: float = 0.5
    created_at: datetime = None
    updated_at: datetime = None
    usage_count: int = 0
    success_rate: float = 0.0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Interaction:
    """User interaction data structure"""

    interaction_id: str
    user_id: str
    question: str
    answer: str
    citations: List[str]
    confidence_score: float
    processing_time: float
    success: bool
    feedback_score: Optional[float] = None
    preferred_agent: Optional[str] = None
    preferred_domain: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningProfile:
    """User learning profile"""

    user_id: str
    preferences: Dict[str, UserPreference]
    interaction_history: List[Interaction]
    success_patterns: Dict[str, Any]
    preferred_agents: Dict[str, float]
    preferred_domains: Dict[str, float]
    response_style_preferences: Dict[str, float]
    chunking_preferences: Dict[str, float]
    navigation_preferences: Dict[str, float]
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class PreferenceLearner:
    """Learns user preferences from interaction history"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.cache = CacheManager()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Learning parameters
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.decay_factor = self.config.get("decay_factor", 0.95)
        self.min_interactions = self.config.get("min_interactions", 5)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)

        # Initialize preference models
        self.agent_preferences = defaultdict(lambda: 0.5)
        self.domain_preferences = defaultdict(lambda: 0.5)
        self.response_style_preferences = defaultdict(lambda: 0.5)
        self.chunking_preferences = defaultdict(lambda: 0.5)
        self.navigation_preferences = defaultdict(lambda: 0.5)

    def update_preferences(self, interaction: Interaction):
        """Update preferences based on user interaction"""

        # Update agent preferences
        if interaction.preferred_agent:
            self._update_preference(
                self.agent_preferences,
                interaction.preferred_agent,
                interaction.success,
                interaction.feedback_score,
            )

        # Update domain preferences
        if interaction.preferred_domain:
            self._update_preference(
                self.domain_preferences,
                interaction.preferred_domain,
                interaction.success,
                interaction.feedback_score,
            )

        # Analyze response style
        response_style = self._analyze_response_style(interaction.answer)
        self._update_preference(
            self.response_style_preferences,
            response_style,
            interaction.success,
            interaction.feedback_score,
        )

        # Analyze chunking preference
        chunking_style = self._analyze_chunking_preference(interaction)
        self._update_preference(
            self.chunking_preferences,
            chunking_style,
            interaction.success,
            interaction.feedback_score,
        )

        # Analyze navigation preference
        navigation_style = self._analyze_navigation_preference(interaction)
        self._update_preference(
            self.navigation_preferences,
            navigation_style,
            interaction.success,
            interaction.feedback_score,
        )

        self.logger.info(f"Updated preferences for user {interaction.user_id}")

    def _update_preference(
        self, preferences: Dict, key: str, success: bool, feedback_score: float
    ):
        """Update a specific preference"""
        current_score = preferences[key]

        # Calculate reward
        if feedback_score is not None:
            reward = feedback_score
        else:
            reward = 1.0 if success else 0.0

        # Update score with learning rate
        new_score = current_score + self.learning_rate * (reward - current_score)
        preferences[key] = max(0.0, min(1.0, new_score))

        # Apply decay to other preferences
        for k in preferences:
            if k != key:
                preferences[k] *= self.decay_factor

    def _analyze_response_style(self, answer: str) -> str:
        """Analyze response style from answer text"""
        # Simple heuristics for response style analysis
        if len(answer.split()) < 50:
            return "concise"
        elif len(answer.split()) > 200:
            return "detailed"
        else:
            return "balanced"

    def _analyze_chunking_preference(self, interaction: Interaction) -> str:
        """Analyze chunking preference from interaction"""
        # Based on number of citations and confidence
        if len(interaction.citations) <= 2:
            return "minimal"
        elif len(interaction.citations) >= 5:
            return "comprehensive"
        else:
            return "moderate"

    def _analyze_navigation_preference(self, interaction: Interaction) -> str:
        """Analyze navigation preference from interaction"""
        # Based on confidence score and processing time
        if interaction.confidence_score > 0.8 and interaction.processing_time < 5:
            return "fast"
        elif interaction.confidence_score > 0.9:
            return "thorough"
        else:
            return "balanced"

    def get_recommendations(
        self, user_id: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get personalized recommendations based on learned preferences"""

        recommendations = {
            "preferred_agent": max(
                self.agent_preferences, key=self.agent_preferences.get
            ),
            "preferred_domain": max(
                self.domain_preferences, key=self.domain_preferences.get
            ),
            "response_style": max(
                self.response_style_preferences, key=self.response_style_preferences.get
            ),
            "chunking_strategy": max(
                self.chunking_preferences, key=self.chunking_preferences.get
            ),
            "navigation_strategy": max(
                self.navigation_preferences, key=self.navigation_preferences.get
            ),
            "confidence_scores": {
                "agent": self.agent_preferences,
                "domain": self.domain_preferences,
                "response_style": self.response_style_preferences,
                "chunking": self.chunking_preferences,
                "navigation": self.navigation_preferences,
            },
        }

        self.logger.info(f"Generated recommendations for user {user_id}")
        return recommendations


class AdaptiveProcessor:
    """Adaptive processor that adjusts behavior based on user preferences"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.learner = PreferenceLearner(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Adaptive parameters
        self.adaptive_chunking = self.config.get("adaptive_chunking", True)
        self.adaptive_navigation = self.config.get("adaptive_navigation", True)
        self.adaptive_agent_selection = self.config.get(
            "adaptive_agent_selection", True
        )

        # Initialize profiles
        self.profiles: Dict[str, LearningProfile] = {}
        self.interactions: Dict[str, List[Interaction]] = defaultdict(list)

    def process_interaction(self, interaction: Interaction):
        """Process user interaction and update learning"""

        # Store interaction
        self.interactions[interaction.user_id].append(interaction)

        # Update learner
        self.learner.update_preferences(interaction)

        # Update user profile
        self._update_user_profile(interaction)

        self.logger.info(f"Processed interaction for user {interaction.user_id}")

    def _update_user_profile(self, interaction: Interaction):
        """Update user learning profile"""
        user_id = interaction.user_id

        if user_id not in self.profiles:
            self.profiles[user_id] = LearningProfile(
                user_id=user_id,
                preferences={},
                interaction_history=[],
                success_patterns={},
                preferred_agents={},
                preferred_domains={},
                response_style_preferences={},
                chunking_preferences={},
                navigation_preferences={},
            )

        profile = self.profiles[user_id]
        profile.interaction_history.append(interaction)
        profile.updated_at = datetime.now()

        # Update success patterns
        self._update_success_patterns(profile, interaction)

        # Get recommendations
        recommendations = self.learner.get_recommendations(user_id, {})

        # Update profile preferences
        profile.preferred_agents = recommendations["confidence_scores"]["agent"]
        profile.preferred_domains = recommendations["confidence_scores"]["domain"]
        profile.response_style_preferences = recommendations["confidence_scores"][
            "response_style"
        ]
        profile.chunking_preferences = recommendations["confidence_scores"]["chunking"]
        profile.navigation_preferences = recommendations["confidence_scores"][
            "navigation"
        ]

    def _update_success_patterns(
        self, profile: LearningProfile, interaction: Interaction
    ):
        """Update success patterns for user"""

        # Track success by agent
        if interaction.preferred_agent:
            if interaction.preferred_agent not in profile.success_patterns:
                profile.success_patterns[interaction.preferred_agent] = {
                    "success": 0,
                    "total": 0,
                }

            profile.success_patterns[interaction.preferred_agent]["total"] += 1
            if interaction.success:
                profile.success_patterns[interaction.preferred_agent]["success"] += 1

        # Track success by domain
        if interaction.preferred_domain:
            if interaction.preferred_domain not in profile.success_patterns:
                profile.success_patterns[interaction.preferred_domain] = {
                    "success": 0,
                    "total": 0,
                }

            profile.success_patterns[interaction.preferred_domain]["total"] += 1
            if interaction.success:
                profile.success_patterns[interaction.preferred_domain]["success"] += 1

    def get_adaptive_config(self, user_id: str) -> Dict[str, Any]:
        """Get adaptive configuration for user"""

        if user_id not in self.profiles:
            # Return default configuration for new users
            return self.config.get("default_adaptive_config", {})

        profile = self.profiles[user_id]
        recommendations = self.learner.get_recommendations(user_id, {})

        adaptive_config = {
            "agent_selection": {
                "preferred_agent": recommendations["preferred_agent"],
                "agent_confidence": recommendations["confidence_scores"]["agent"],
            },
            "chunking_strategy": {
                "strategy": recommendations["chunking_strategy"],
                "max_chunk_size": self._get_adaptive_chunk_size(
                    recommendations["chunking_strategy"]
                ),
                "overlap_tokens": self._get_adaptive_overlap(
                    recommendations["chunking_strategy"]
                ),
            },
            "navigation_strategy": {
                "strategy": recommendations["navigation_strategy"],
                "max_depth": self._get_adaptive_depth(
                    recommendations["navigation_strategy"]
                ),
                "timeout": self._get_adaptive_timeout(
                    recommendations["navigation_strategy"]
                ),
            },
            "response_style": {
                "style": recommendations["response_style"],
                "temperature": self._get_adaptive_temperature(
                    recommendations["response_style"]
                ),
            },
        }

        self.logger.info(f"Generated adaptive config for user {user_id}")
        return adaptive_config

    def _get_adaptive_chunk_size(self, strategy: str) -> int:
        """Get adaptive chunk size based on strategy"""
        chunk_sizes = {"minimal": 1000, "moderate": 2000, "comprehensive": 3000}
        return chunk_sizes.get(strategy, 2000)

    def _get_adaptive_overlap(self, strategy: str) -> int:
        """Get adaptive overlap based on strategy"""
        overlaps = {"minimal": 50, "moderate": 100, "comprehensive": 200}
        return overlaps.get(strategy, 100)

    def _get_adaptive_depth(self, strategy: str) -> int:
        """Get adaptive navigation depth based on strategy"""
        depths = {"fast": 2, "balanced": 3, "thorough": 4}
        return depths.get(strategy, 3)

    def _get_adaptive_timeout(self, strategy: str) -> int:
        """Get adaptive timeout based on strategy"""
        timeouts = {"fast": 10, "balanced": 30, "thorough": 60}
        return timeouts.get(strategy, 30)

    def _get_adaptive_temperature(self, style: str) -> float:
        """Get adaptive temperature based on response style"""
        temperatures = {"concise": 0.1, "balanced": 0.3, "detailed": 0.5}
        return temperatures.get(style, 0.3)

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user behavior and preferences"""

        if user_id not in self.profiles:
            return {"error": "User not found"}

        profile = self.profiles[user_id]
        interactions = self.interactions[user_id]

        # Calculate statistics
        total_interactions = len(interactions)
        successful_interactions = sum(1 for i in interactions if i.success)
        success_rate = (
            successful_interactions / total_interactions
            if total_interactions > 0
            else 0
        )

        # Calculate average confidence
        avg_confidence = (
            sum(i.confidence_score for i in interactions) / total_interactions
            if total_interactions > 0
            else 0
        )

        # Calculate average processing time
        avg_processing_time = (
            sum(i.processing_time for i in interactions) / total_interactions
            if total_interactions > 0
            else 0
        )

        # Get most active time
        if interactions:
            most_active_hour = max(
                set(i.timestamp.hour for i in interactions),
                key=list(i.timestamp.hour for i in interactions).count,
            )
        else:
            most_active_hour = None

        insights = {
            "user_id": user_id,
            "total_interactions": total_interactions,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "most_active_hour": most_active_hour,
            "preferred_agents": profile.preferred_agents,
            "preferred_domains": profile.preferred_domains,
            "response_style_preferences": profile.response_style_preferences,
            "success_patterns": profile.success_patterns,
            "first_interaction": interactions[0].timestamp.isoformat()
            if interactions
            else None,
            "last_interaction": interactions[-1].timestamp.isoformat()
            if interactions
            else None,
        }

        self.logger.info(f"Generated insights for user {user_id}")
        return insights

    def export_profile(self, user_id: str, file_path: str):
        """Export user profile to file"""

        if user_id not in self.profiles:
            raise ValueError(f"User {user_id} not found")

        profile = self.profiles[user_id]

        # Convert datetime objects to strings
        export_data = {
            "user_id": profile.user_id,
            "preferences": {
                k: asdict(v) if hasattr(v, "__dict__") else v
                for k, v in profile.preferences.items()
            },
            "interaction_history": [
                asdict(interaction) for interaction in profile.interaction_history
            ],
            "success_patterns": profile.success_patterns,
            "preferred_agents": profile.preferred_agents,
            "preferred_domains": profile.preferred_domains,
            "response_style_preferences": profile.response_style_preferences,
            "chunking_preferences": profile.chunking_preferences,
            "navigation_preferences": profile.navigation_preferences,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported profile for user {user_id} to {file_path}")

    def import_profile(self, user_id: str, file_path: str):
        """Import user profile from file"""

        with open(file_path, "r", encoding="utf-8") as f:
            import_data = json.load(f)

        # Convert datetime strings back to datetime objects
        profile = LearningProfile(
            user_id=import_data["user_id"],
            preferences={
                k: UserPreference(**v) for k, v in import_data["preferences"].items()
            },
            interaction_history=[
                Interaction(**i) for i in import_data["interaction_history"]
            ],
            success_patterns=import_data["success_patterns"],
            preferred_agents=import_data["preferred_agents"],
            preferred_domains=import_data["preferred_domains"],
            response_style_preferences=import_data["response_style_preferences"],
            chunking_preferences=import_data["chunking_preferences"],
            navigation_preferences=import_data["navigation_preferences"],
            created_at=datetime.fromisoformat(import_data["created_at"]),
            updated_at=datetime.fromisoformat(import_data["updated_at"]),
        )

        self.profiles[user_id] = profile
        self.interactions[user_id] = profile.interaction_history

        self.logger.info(f"Imported profile for user {user_id} from {file_path}")


class LearningSystemManager:
    """Main manager for the learning system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config().dict()
        self.processor = AdaptiveProcessor(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # File paths
        self.data_dir = Path(self.config.get("data_dir", "./learning_data"))
        self.data_dir.mkdir(exist_ok=True)

        self.profiles_file = self.data_dir / "profiles.json"
        self.interactions_file = self.data_dir / "interactions.pkl"

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing learning data"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, "r", encoding="utf-8") as f:
                    profiles_data = json.load(f)

                # Reconstruct profiles
                for user_id, profile_data in profiles_data.items():
                    profile = LearningProfile(
                        user_id=profile_data["user_id"],
                        preferences={
                            k: UserPreference(**v)
                            for k, v in profile_data["preferences"].items()
                        },
                        interaction_history=[
                            Interaction(**i)
                            for i in profile_data["interaction_history"]
                        ],
                        success_patterns=profile_data["success_patterns"],
                        preferred_agents=profile_data["preferred_agents"],
                        preferred_domains=profile_data["preferred_domains"],
                        response_style_preferences=profile_data[
                            "response_style_preferences"
                        ],
                        chunking_preferences=profile_data["chunking_preferences"],
                        navigation_preferences=profile_data["navigation_preferences"],
                        created_at=datetime.fromisoformat(profile_data["created_at"]),
                        updated_at=datetime.fromisoformat(profile_data["updated_at"]),
                    )
                    self.processor.profiles[user_id] = profile
                    self.processor.interactions[user_id] = profile.interaction_history

                self.logger.info(f"Loaded {len(profiles_data)} user profiles")

            if self.interactions_file.exists():
                with open(self.interactions_file, "rb") as f:
                    interactions_data = pickle.load(f)

                # Reconstruct interactions
                for user_id, interactions in interactions_data.items():
                    self.processor.interactions[user_id] = [
                        Interaction(**interaction_data)
                        for interaction_data in interactions
                    ]

                self.logger.info(
                    f"Loaded interactions for {len(interactions_data)} users"
                )

        except Exception as e:
            self.logger.error(f"Error loading learning data: {e}")

    def _save_data(self):
        """Save learning data"""
        try:
            # Save profiles
            profiles_data = {}
            for user_id, profile in self.processor.profiles.items():
                profiles_data[user_id] = {
                    "user_id": profile.user_id,
                    "preferences": {
                        k: asdict(v) if hasattr(v, "__dict__") else v
                        for k, v in profile.preferences.items()
                    },
                    "interaction_history": [
                        asdict(interaction)
                        for interaction in profile.interaction_history
                    ],
                    "success_patterns": profile.success_patterns,
                    "preferred_agents": profile.preferred_agents,
                    "preferred_domains": profile.preferred_domains,
                    "response_style_preferences": profile.response_style_preferences,
                    "chunking_preferences": profile.chunking_preferences,
                    "navigation_preferences": profile.navigation_preferences,
                    "created_at": profile.created_at.isoformat(),
                    "updated_at": profile.updated_at.isoformat(),
                }

            with open(self.profiles_file, "w", encoding="utf-8") as f:
                json.dump(profiles_data, f, indent=2, ensure_ascii=False)

            # Save interactions
            interactions_data = {}
            for user_id, interactions in self.processor.interactions.items():
                interactions_data[user_id] = [
                    asdict(interaction) for interaction in interactions
                ]

            with open(self.interactions_file, "wb") as f:
                pickle.dump(interactions_data, f)

            self.logger.info("Saved learning data")

        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")

    def process_interaction(self, interaction: Interaction):
        """Process user interaction"""
        self.processor.process_interaction(interaction)
        self._save_data()

    def get_user_profile(self, user_id: str) -> Optional[LearningProfile]:
        """Get user profile"""
        return self.processor.profiles.get(user_id)

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get user insights"""
        return self.processor.get_user_insights(user_id)

    def get_adaptive_config(self, user_id: str) -> Dict[str, Any]:
        """Get adaptive configuration"""
        return self.processor.get_adaptive_config(user_id)

    def export_user_profile(self, user_id: str, file_path: str):
        """Export user profile"""
        self.processor.export_profile(user_id, file_path)

    def import_user_profile(self, user_id: str, file_path: str):
        """Import user profile"""
        self.processor.import_profile(user_id, file_path)
        self._save_data()

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        total_users = len(self.processor.profiles)
        total_interactions = sum(
            len(interactions) for interactions in self.processor.interactions.values()
        )

        # Calculate overall success rate
        successful_interactions = sum(
            sum(1 for interaction in interactions if interaction.success)
            for interactions in self.processor.interactions.values()
        )
        overall_success_rate = (
            successful_interactions / total_interactions
            if total_interactions > 0
            else 0
        )

        # Get most popular agents and domains
        all_agents = []
        all_domains = []

        for profile in self.processor.profiles.values():
            all_agents.extend(profile.preferred_agents.keys())
            all_domains.extend(profile.preferred_domains.keys())

        popular_agents = (
            max(set(all_agents), key=all_agents.count) if all_agents else None
        )
        popular_domains = (
            max(set(all_domains), key=all_domains.count) if all_domains else None
        )

        stats = {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "overall_success_rate": overall_success_rate,
            "popular_agent": popular_agents,
            "popular_domain": popular_domains,
            "data_directory": str(self.data_dir),
            "profiles_file": str(self.profiles_file),
            "interactions_file": str(self.interactions_file),
        }

        return stats
