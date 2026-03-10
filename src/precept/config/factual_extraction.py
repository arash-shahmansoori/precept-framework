"""
Factual Extraction Configuration.

Configuration for extracting factual statements from experiential lessons
to enable semantic conflict detection with static knowledge.

NO HARDCODED VALUES: All patterns and thresholds are loaded from:
1. Environment variables
2. Configuration files
3. Runtime configuration

Usage:
    from precept.config import FactualExtractionConfig, get_factual_extraction_config

    # Get default configuration (from environment or defaults)
    config = get_factual_extraction_config()

    # Create custom configuration
    config = FactualExtractionConfig.from_env()

    # Override at runtime
    config = FactualExtractionConfig(
        status_negative_keywords=["blocked", "closed"],
        min_lesson_length=15,
    )
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FactualExtractionConfig:
    """
    Configuration for extracting factual statements from lessons.

    SINGLE SOURCE OF TRUTH: All defaults are defined here.
    Can be overridden via environment variables.

    Environment Variables:
        PRECEPT_FACTUAL_STATUS_NEGATIVE: Comma-separated negative status keywords
        PRECEPT_FACTUAL_STATUS_POSITIVE: Comma-separated positive status keywords
        PRECEPT_FACTUAL_DELAY_KEYWORDS: Comma-separated delay keywords
        PRECEPT_FACTUAL_REQUIREMENT_KEYWORDS: Comma-separated requirement keywords
        PRECEPT_FACTUAL_ALTERNATIVE_KEYWORDS: Comma-separated alternative keywords
        PRECEPT_FACTUAL_STRIKE_INDICATORS: Comma-separated strike indicators
        PRECEPT_FACTUAL_CONGESTION_INDICATORS: Comma-separated congestion indicators
        PRECEPT_FACTUAL_CUSTOMS_INDICATORS: Comma-separated customs indicators
        PRECEPT_FACTUAL_MIN_LESSON_LENGTH: Minimum lesson length (int)
        PRECEPT_FACTUAL_INCLUDE_ORIGINAL: Include original as fallback (bool)
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # DEFAULT VALUES - Single source of truth (no external JSON needed)
    # ═══════════════════════════════════════════════════════════════════════════

    # Keyword patterns for status classification
    status_negative_keywords: List[str] = field(
        default_factory=lambda: [
            "blocked",
            "strike",
            "closed",
            "unavailable",
            "congested",
            "shutdown",
            "suspended",
            "halted",
        ]
    )
    status_positive_keywords: List[str] = field(
        default_factory=lambda: [
            "operational",
            "working",
            "open",
            "available",
            "clear",
            "running",
            "active",
            "functioning",
        ]
    )
    delay_keywords: List[str] = field(
        default_factory=lambda: [
            "delayed",
            "delay",
            "slow",
            "backlog",
            "waiting",
            "queue",
            "bottleneck",
        ]
    )
    requirement_keywords: List[str] = field(
        default_factory=lambda: [
            "required",
            "must",
            "mandatory",
            "always",
            "necessary",
            "essential",
            "compulsory",
        ]
    )
    alternative_keywords: List[str] = field(
        default_factory=lambda: [
            "use",
            "alternative",
            "instead",
            "fallback",
            "switch",
            "redirect",
            "reroute",
        ]
    )

    # Sub-category indicators
    strike_indicators: List[str] = field(
        default_factory=lambda: ["strike", "labor", "dispute", "union", "walkout"]
    )
    congestion_indicators: List[str] = field(
        default_factory=lambda: [
            "congested",
            "congestion",
            "capacity",
            "overflow",
            "backed up",
        ]
    )
    customs_indicators: List[str] = field(
        default_factory=lambda: [
            "customs",
            "clearance",
            "declaration",
            "import",
            "export",
        ]
    )

    # Thresholds
    min_lesson_length: int = 10
    include_original_as_fallback: bool = True

    # Factual statement verbs (for detecting if lesson is already factual)
    factual_verbs: List[str] = field(
        default_factory=lambda: [
            "is",
            "are",
            "has",
            "have",
            "cannot",
            "must",
            "will",
            "should",
        ]
    )

    @classmethod
    def from_env(cls) -> "FactualExtractionConfig":
        """
        Load configuration with environment variable overrides.

        Priority:
        1. Environment variables (override defaults)
        2. Default values defined in this class

        No external JSON file needed - single source of truth.
        """

        def parse_list(env_var: str, default: List[str]) -> List[str]:
            """Parse comma-separated environment variable into list."""
            value = os.environ.get(env_var, "")
            if not value:
                return default
            return [item.strip() for item in value.split(",") if item.strip()]

        def parse_bool(env_var: str, default: bool) -> bool:
            """Parse boolean environment variable."""
            value = os.environ.get(env_var, "").lower()
            if value in ("true", "1", "yes"):
                return True
            elif value in ("false", "0", "no"):
                return False
            return default

        def parse_int(env_var: str, default: int) -> int:
            """Parse integer environment variable."""
            try:
                return int(os.environ.get(env_var, str(default)))
            except ValueError:
                return default

        # Create instance with defaults, then check for env overrides
        default_instance = cls()

        return cls(
            status_negative_keywords=parse_list(
                "PRECEPT_FACTUAL_STATUS_NEGATIVE",
                default_instance.status_negative_keywords,
            ),
            status_positive_keywords=parse_list(
                "PRECEPT_FACTUAL_STATUS_POSITIVE",
                default_instance.status_positive_keywords,
            ),
            delay_keywords=parse_list(
                "PRECEPT_FACTUAL_DELAY_KEYWORDS", default_instance.delay_keywords
            ),
            requirement_keywords=parse_list(
                "PRECEPT_FACTUAL_REQUIREMENT_KEYWORDS",
                default_instance.requirement_keywords,
            ),
            alternative_keywords=parse_list(
                "PRECEPT_FACTUAL_ALTERNATIVE_KEYWORDS",
                default_instance.alternative_keywords,
            ),
            strike_indicators=parse_list(
                "PRECEPT_FACTUAL_STRIKE_INDICATORS", default_instance.strike_indicators
            ),
            congestion_indicators=parse_list(
                "PRECEPT_FACTUAL_CONGESTION_INDICATORS",
                default_instance.congestion_indicators,
            ),
            customs_indicators=parse_list(
                "PRECEPT_FACTUAL_CUSTOMS_INDICATORS",
                default_instance.customs_indicators,
            ),
            factual_verbs=parse_list(
                "PRECEPT_FACTUAL_VERBS", default_instance.factual_verbs
            ),
            min_lesson_length=parse_int(
                "PRECEPT_FACTUAL_MIN_LESSON_LENGTH", default_instance.min_lesson_length
            ),
            include_original_as_fallback=parse_bool(
                "PRECEPT_FACTUAL_INCLUDE_ORIGINAL",
                default_instance.include_original_as_fallback,
            ),
        )

    @classmethod
    def from_file(cls, path: str) -> "FactualExtractionConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            status_negative_keywords=data.get("status_negative_keywords", []),
            status_positive_keywords=data.get("status_positive_keywords", []),
            delay_keywords=data.get("delay_keywords", []),
            requirement_keywords=data.get("requirement_keywords", []),
            alternative_keywords=data.get("alternative_keywords", []),
            strike_indicators=data.get("strike_indicators", []),
            congestion_indicators=data.get("congestion_indicators", []),
            customs_indicators=data.get("customs_indicators", []),
            factual_verbs=data.get("factual_verbs", []),
            min_lesson_length=data.get("min_lesson_length", 0),
            include_original_as_fallback=data.get("include_original_as_fallback", True),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "status_negative_keywords": self.status_negative_keywords,
            "status_positive_keywords": self.status_positive_keywords,
            "delay_keywords": self.delay_keywords,
            "requirement_keywords": self.requirement_keywords,
            "alternative_keywords": self.alternative_keywords,
            "strike_indicators": self.strike_indicators,
            "congestion_indicators": self.congestion_indicators,
            "customs_indicators": self.customs_indicators,
            "factual_verbs": self.factual_verbs,
            "min_lesson_length": self.min_lesson_length,
            "include_original_as_fallback": self.include_original_as_fallback,
        }

    def save_to_file(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def is_configured(self) -> bool:
        """Check if configuration has any patterns defined."""
        return bool(
            self.status_negative_keywords
            or self.status_positive_keywords
            or self.delay_keywords
            or self.requirement_keywords
            or self.alternative_keywords
        )


# Global configuration instance (lazy-loaded)
_factual_extraction_config: Optional[FactualExtractionConfig] = None


def get_factual_extraction_config() -> FactualExtractionConfig:
    """
    Get the factual extraction configuration.

    Loads from environment variables on first call.
    Use configure_factual_extraction() to override.
    """
    global _factual_extraction_config
    if _factual_extraction_config is None:
        _factual_extraction_config = FactualExtractionConfig.from_env()
    return _factual_extraction_config


def configure_factual_extraction(config: FactualExtractionConfig) -> None:
    """
    Configure the factual extraction system at runtime.

    Args:
        config: FactualExtractionConfig instance with desired settings
    """
    global _factual_extraction_config
    _factual_extraction_config = config


def reset_factual_extraction_config() -> None:
    """Reset the configuration to reload from environment."""
    global _factual_extraction_config
    _factual_extraction_config = None
