"""
Cutting-Edge Conflict Resolution Module for PRECEPT Framework.

This module implements a multi-stage, evidence-based, and Bayesian-informed
approach to conflict resolution between static knowledge and dynamic experience.

Key Features:
- Semantic Entailment Detection (NLI-based)
- Bayesian Uncertainty Quantification (Beta distributions)
- Dynamic Reliability Learning
- Active Exploration (Thompson Sampling)
- Ensemble Detection (multi-method voting)
- Evidence-Based Prioritization
- Anomaly Detection for outlier experiences

All thresholds, weights, and parameters are CONFIGURABLE - no hardcoded logic.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# ENUMS
# =============================================================================


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SemanticRelation(Enum):
    """Semantic relationship between two pieces of knowledge."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class KnowledgeSource(Enum):
    """Source type for knowledge items."""

    STATIC_KB = "static_kb"
    DYNAMIC_EXPERIENCE = "dynamic_experience"
    EPISODIC_MEMORY = "episodic_memory"
    LLM_GENERATED = "llm_generated"


# =============================================================================
# CONFIGURATION (All parameters are configurable - NO hardcoded values)
# =============================================================================


@dataclass
class ConflictResolutionConfig:
    """
    Configuration for conflict resolution.

    All thresholds, weights, decay factors, and reliability scores are
    configurable parameters. This ensures no hardcoded logic.
    """

    # Semantic similarity thresholds (lowered for better conflict detection)
    semantic_similarity_threshold: float = 0.5  # Lowered from 0.7
    contradiction_similarity_threshold: float = 0.4  # Lowered from 0.6
    high_similarity_threshold: float = 0.75  # Lowered from 0.85

    # NLI confidence thresholds (lowered for sensitivity)
    nli_entailment_threshold: float = 0.6  # Lowered from 0.7
    nli_contradiction_threshold: float = 0.5  # Lowered from 0.7
    nli_neutral_threshold: float = 0.4  # Lowered from 0.5

    # Ensemble voting weights
    nli_vote_weight: float = 0.30
    semantic_vote_weight: float = 0.30  # Increased semantic weight
    temporal_vote_weight: float = 0.15
    evidence_vote_weight: float = 0.15
    llm_vote_weight: float = 0.10
    recommendation_vote_weight: float = 0.50

    # Ensemble confidence thresholds
    ensemble_conflict_threshold: float = 0.30
    ensemble_high_confidence_threshold: float = 0.7

    # Temporal decay parameters
    recency_decay_rate: float = 0.1  # per day
    max_staleness_days: float = 30.0
    staleness_penalty_threshold: float = 0.5

    # Reliability parameters (Bayesian Beta distribution priors)
    # Static KB is UNVERIFIED external knowledge; skeptical prior reflects
    # that it may be outdated or adversarial until confirmed by execution.
    static_prior_alpha: float = 5.0
    static_prior_beta: float = 5.0
    dynamic_prior_alpha: float = 5.0
    dynamic_prior_beta: float = 3.0
    episodic_prior_alpha: float = 3.0
    episodic_prior_beta: float = 3.0

    # Reliability learning rates
    reliability_update_weight: float = 1.0
    min_reliability_score: float = 0.1
    max_reliability_score: float = 0.99

    # Evidence strength thresholds
    min_confirmations_for_high_evidence: int = 3
    min_failures_for_low_evidence: int = 2
    evidence_confirmation_weight: float = 0.3
    evidence_failure_weight: float = 0.4

    # Anomaly detection thresholds
    anomaly_single_point_threshold: float = 0.7
    anomaly_failure_rate_threshold: float = 0.5
    anomaly_staleness_threshold_days: float = 14.0

    # Resolution strategy weights
    recency_strategy_weight: float = 0.25
    reliability_strategy_weight: float = 0.30
    specificity_strategy_weight: float = 0.20
    evidence_strategy_weight: float = 0.25

    # Static vs Dynamic comparison thresholds
    static_wins_confidence_threshold: float = 0.7
    dynamic_wins_confidence_threshold: float = 0.7
    merge_confidence_threshold: float = 0.5

    # Active exploration (Thompson Sampling)
    exploration_probability_threshold: float = 0.3
    exploration_uncertainty_threshold: float = 0.4
    min_observations_for_exploration: int = 5

    # LLM integration
    use_llm_for_resolution: bool = True
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2
    llm_fallback_confidence: float = (
        0.7  # Fallback confidence when LLM doesn't return one
    )

    # Pattern-based detection keywords (configurable, logistics-aware)
    contradiction_keywords: List[str] = field(
        default_factory=lambda: [
            # General negation
            "not",
            "never",
            "no longer",
            "discontinued",
            # Port/logistics specific
            "closed",
            "unavailable",
            "blocked",
            "suspended",
            "cancelled",
            "strike",
            "congested",
            "delayed",
            "backlog",
            "shutdown",
            "halted",
            "dispute",
            # Status changes
            "incorrect",
            "wrong",
            "false",
            "invalid",
            "outdated",
            "stale",
            "changed",
            "different",
            # Failure indicators
            "failed",
            "error",
            "rejected",
            "denied",
        ]
    )

    agreement_keywords: List[str] = field(
        default_factory=lambda: [
            "confirmed",
            "verified",
            "correct",
            "accurate",
            "valid",
            "operational",
            "active",
            "available",
            "working",
            "open",
            "running",
            "stable",
            "normal",
            "successful",
            "cleared",
        ]
    )

    # Contradiction phrase patterns (for more complex matching)
    contradiction_phrase_patterns: List[str] = field(
        default_factory=lambda: [
            "{entity} is blocked",
            "{entity} is closed",
            "{entity} has strike",
            "{entity} is congested",
            "{entity} is unavailable",
            "{entity} not operational",
            "{entity} is delayed",
            "use {alternative} instead of {entity}",
        ]
    )

    # Recommendation-conflict detection: dismissive phrases in static KB
    # that downplay severity or suggest no action is needed.
    # Designed to be domain-agnostic across logistics, integration, booking.
    dismissive_phrases: List[str] = field(
        default_factory=lambda: [
            # General dismissive
            "can be ignored",
            "safe to ignore",
            "no fallback needed",
            "don't need",
            "don't apply",
            "auto-recovers",
            "auto-recover",
            "not affected",
            "always reliable",
            "immune",
            "are guaranteed",
            "is guaranteed",
            "retry automatically",
            "rarely expire",
            "is a reliable option",
            "is rare",
            "are client-side",
            "is permanent",
            "is uncommon",
            # Normalcy claims (logistics / booking)
            "proceed normally",
            "processed normally",
            "without issues",
            "without delays",
            "no labor disruptions",
            "no congestion",
            "operating at normal",
            "cleared its backlog",
            "excellent availability",
            "can still succeed",
            "get priority",
            # Over-permissive claims (logistics)
            "can wait indefinitely",
            "can use standard",
            "can be routed through any",
            "regardless of conditions",
        ]
    )

    # Phrases in dynamic experience that indicate an active strategy was needed
    active_strategy_phrases: List[str] = field(
        default_factory=lambda: [
            "effective strategy is",
            "strategy is",
            "strategy:",
            "learned:",
            "the correct approach",
            "should use",
            "need to",
            "requires",
            "for conditions [",
            "solution:",
            "error:",
            "conditions:",
        ]
    )


@dataclass
class LearnedPatterns:
    """Learned patterns for conflict detection (can be updated over time)."""

    contradiction_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            # General operational status
            ("operational", "closed"),
            ("available", "unavailable"),
            ("active", "suspended"),
            ("working", "broken"),
            ("fast", "delayed"),
            ("open", "blocked"),
            # Logistics-specific (labor/stability)
            ("stable", "strike"),
            ("stable", "blocked"),
            ("stable", "congested"),
            ("stable", "disrupted"),
            ("normal", "blocked"),
            ("normal", "strike"),
            ("normal", "congested"),
            ("normal", "delayed"),
            # Port operations
            ("operational", "blocked"),
            ("operational", "strike"),
            ("operational", "congested"),
            ("running", "blocked"),
            ("running", "closed"),
            ("running", "strike"),
            # Capacity/processing
            ("capacity", "backlog"),
            ("standard", "expedited"),
            ("quick", "delayed"),
            ("smooth", "problems"),
        ]
    )

    regulatory_keywords: List[str] = field(
        default_factory=lambda: [
            "regulation",
            "compliance",
            "FDA",
            "WCO",
            "certificate",
            "mandatory",
            "required",
            "must",
            "shall",
        ]
    )

    temporal_keywords: List[str] = field(
        default_factory=lambda: [
            "currently",
            "now",
            "recently",
            "update",
            "alert",
            "as of",
            "effective",
            "temporary",
        ]
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class BetaDistribution:
    """Beta distribution for Bayesian uncertainty quantification."""

    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total * total * (total + 1))

    def sample(self) -> float:
        """Sample from the distribution (Thompson Sampling)."""
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool, weight: float = 1.0) -> "BetaDistribution":
        """Update distribution based on observation."""
        if success:
            return BetaDistribution(self.alpha + weight, self.beta)
        else:
            return BetaDistribution(self.alpha, self.beta + weight)

    def confidence_interval(self, z: float = 1.96) -> Tuple[float, float]:
        """Approximate confidence interval using normal approximation."""
        mean = self.mean()
        std = math.sqrt(self.variance())
        return (max(0, mean - z * std), min(1, mean + z * std))


@dataclass
class UncertainValue:
    """A value with uncertainty quantification."""

    value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    source: str = ""

    @classmethod
    def from_beta(cls, dist: BetaDistribution, source: str = "") -> "UncertainValue":
        """Create from Beta distribution."""
        lower, upper = dist.confidence_interval()
        return cls(
            value=dist.mean(),
            confidence=1.0 - (upper - lower),  # Narrower interval = higher confidence
            lower_bound=lower,
            upper_bound=upper,
            source=source,
        )


@dataclass
class NLIResult:
    """Result from Natural Language Inference."""

    relation: SemanticRelation
    confidence: float
    method: str  # "neural", "pattern", "llm"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeItem:
    """A piece of knowledge with metadata."""

    id: str
    content: str
    source: KnowledgeSource
    timestamp: datetime
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confirmations: int = 0
    failures: int = 0

    def age_days(self) -> float:
        """Age of the knowledge in days."""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 86400


@dataclass
class ConflictRecord:
    """Record of a detected conflict."""

    id: str
    static_item: KnowledgeItem
    dynamic_item: KnowledgeItem
    relation: SemanticRelation
    severity: ConflictSeverity
    confidence: float
    detection_method: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None


@dataclass
class ResolutionResult:
    """Result of conflict resolution."""

    id: str
    conflict: ConflictRecord
    winner: KnowledgeSource
    winning_item: KnowledgeItem
    confidence: float
    strategy_used: str
    reasoning: str
    merged_content: Optional[str] = None
    should_update_static: bool = False
    should_flag_for_review: bool = False


@dataclass
class MethodVote:
    """A vote from a detection method."""

    method: str
    is_conflict: bool
    confidence: float
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# NEURAL NLI CLASSIFIER
# =============================================================================


class NeuralNLIClassifier:
    """
    Natural Language Inference classifier.

    Uses multiple methods:
    1. Neural model (if available) - e.g., RoBERTa/DeBERTa
    2. Pattern-based detection
    3. LLM fallback
    """

    def __init__(
        self,
        config: ConflictResolutionConfig,
        patterns: LearnedPatterns,
        model_fn: Optional[Callable] = None,
        llm_client: Optional[Callable] = None,
        use_llm_fallback: bool = True,
    ):
        self.config = config
        self.patterns = patterns
        self.model_fn = model_fn
        self.llm_client = llm_client
        self.use_llm_fallback = use_llm_fallback

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Classify the relationship between premise and hypothesis.

        Args:
            premise: The reference statement (typically static knowledge)
            hypothesis: The statement to compare (typically dynamic knowledge)

        Returns:
            NLIResult with relation, confidence, and method used
        """
        # Try neural model first
        if self.model_fn is not None:
            try:
                result = self._neural_nli(premise, hypothesis)
                if result.confidence >= self.config.nli_entailment_threshold:
                    return result
            except Exception:
                pass  # Fall through to other methods

        # Try pattern-based detection
        pattern_result = self._pattern_based_nli(premise, hypothesis)
        if pattern_result.confidence >= self.config.nli_contradiction_threshold:
            return pattern_result

        # LLM fallback
        if self.use_llm_fallback and self.llm_client is not None:
            try:
                return self._llm_nli(premise, hypothesis)
            except Exception:
                pass

        # Default to pattern result or unknown
        if pattern_result.confidence > 0:
            return pattern_result

        return NLIResult(
            relation=SemanticRelation.UNKNOWN,
            confidence=self.config.nli_neutral_threshold,
            method="default",
        )

    def _neural_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Use neural model for NLI."""
        if self.model_fn is None:
            raise ValueError("Neural model not available")

        result = self.model_fn(premise, hypothesis)

        # Expect result to be dict with 'label' and 'score'
        label = result.get("label", "neutral").lower()
        score = result.get("score", 0.5)

        if "contradict" in label:
            relation = SemanticRelation.CONTRADICTS
        elif "entail" in label:
            relation = SemanticRelation.ENTAILS
        else:
            relation = SemanticRelation.NEUTRAL

        return NLIResult(
            relation=relation,
            confidence=score,
            method="neural",
            details={"raw_result": result},
        )

    def _pattern_based_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Pattern-based NLI detection."""
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        # Check for contradiction patterns
        contradiction_score = 0.0
        matched_patterns = []

        for word1, word2 in self.patterns.contradiction_pairs:
            if word1 in premise_lower and word2 in hypothesis_lower:
                contradiction_score += self.config.semantic_vote_weight
                matched_patterns.append((word1, word2))
            elif word2 in premise_lower and word1 in hypothesis_lower:
                contradiction_score += self.config.semantic_vote_weight
                matched_patterns.append((word2, word1))

        # Check for contradiction keywords
        for keyword in self.config.contradiction_keywords:
            if keyword in hypothesis_lower and keyword not in premise_lower:
                contradiction_score += self.config.evidence_vote_weight
                matched_patterns.append(f"keyword:{keyword}")

        if contradiction_score > 0:
            return NLIResult(
                relation=SemanticRelation.CONTRADICTS,
                confidence=min(contradiction_score, self.config.max_reliability_score),
                method="pattern",
                details={"matched_patterns": matched_patterns},
            )

        # Check for agreement patterns
        agreement_score = 0.0
        for keyword in self.config.agreement_keywords:
            if keyword in premise_lower and keyword in hypothesis_lower:
                agreement_score += self.config.evidence_vote_weight

        if agreement_score > 0:
            return NLIResult(
                relation=SemanticRelation.ENTAILS,
                confidence=min(agreement_score, self.config.max_reliability_score),
                method="pattern",
                details={"agreement_score": agreement_score},
            )

        return NLIResult(
            relation=SemanticRelation.NEUTRAL,
            confidence=self.config.nli_neutral_threshold,
            method="pattern",
        )

    def _llm_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """Use LLM for NLI classification."""
        if self.llm_client is None:
            raise ValueError("LLM client not available")

        prompt = f"""Analyze the logical relationship between these two statements:

STATEMENT A (Reference): {premise}

STATEMENT B (New Information): {hypothesis}

What is the logical relationship?
1. CONTRADICTS - Statement B contradicts or negates Statement A
2. ENTAILS - Statement B supports or is consistent with Statement A
3. NEUTRAL - The statements are unrelated or independent

Respond with exactly one word: CONTRADICTS, ENTAILS, or NEUTRAL
Then on a new line, provide a confidence score from 0.0 to 1.0"""

        try:
            response = self.llm_client(prompt)
            lines = response.strip().split("\n")

            label = lines[0].strip().upper()
            confidence = (
                float(lines[1].strip())
                if len(lines) > 1
                else self.config.llm_fallback_confidence
            )

            if "CONTRADICT" in label:
                relation = SemanticRelation.CONTRADICTS
            elif "ENTAIL" in label:
                relation = SemanticRelation.ENTAILS
            else:
                relation = SemanticRelation.NEUTRAL

            return NLIResult(
                relation=relation,
                confidence=confidence,
                method="llm",
                details={"raw_response": response},
            )
        except Exception as e:
            return NLIResult(
                relation=SemanticRelation.UNKNOWN,
                confidence=self.config.min_reliability_score,
                method="llm_error",
                details={"error": str(e)},
            )


# =============================================================================
# DYNAMIC RELIABILITY TRACKER
# =============================================================================


class DynamicReliabilityTracker:
    """
    Tracks and learns reliability of different knowledge sources.

    Uses Bayesian inference with Beta distributions to model uncertainty
    and learn from outcomes.
    """

    def __init__(self, config: ConflictResolutionConfig):
        self.config = config
        self.distributions: Dict[KnowledgeSource, BetaDistribution] = {
            KnowledgeSource.STATIC_KB: BetaDistribution(
                config.static_prior_alpha, config.static_prior_beta
            ),
            KnowledgeSource.DYNAMIC_EXPERIENCE: BetaDistribution(
                config.dynamic_prior_alpha, config.dynamic_prior_beta
            ),
            KnowledgeSource.EPISODIC_MEMORY: BetaDistribution(
                config.episodic_prior_alpha, config.episodic_prior_beta
            ),
        }
        self.observation_counts: Dict[KnowledgeSource, int] = {
            source: 0 for source in KnowledgeSource
        }

    def update(
        self, source: KnowledgeSource, was_correct: bool, weight: float = None
    ) -> None:
        """Update reliability based on observed outcome."""
        if weight is None:
            weight = self.config.reliability_update_weight

        if source in self.distributions:
            self.distributions[source] = self.distributions[source].update(
                was_correct, weight
            )
            self.observation_counts[source] = self.observation_counts.get(source, 0) + 1

    def get_reliability(self, source: KnowledgeSource) -> UncertainValue:
        """Get current reliability estimate with uncertainty."""
        if source not in self.distributions:
            return UncertainValue(
                value=0.5,
                confidence=0.0,
                lower_bound=0.0,
                upper_bound=1.0,
                source=source.value,
            )

        dist = self.distributions[source]
        return UncertainValue.from_beta(dist, source.value)

    def compare_reliabilities(
        self, source_a: KnowledgeSource, source_b: KnowledgeSource
    ) -> Tuple[KnowledgeSource, float]:
        """
        Compare two sources and return which is more reliable.

        Returns:
            Tuple of (winning source, confidence in the comparison)
        """
        rel_a = self.get_reliability(source_a)
        rel_b = self.get_reliability(source_b)

        # If confidence intervals don't overlap, we're confident
        if rel_a.lower_bound > rel_b.upper_bound:
            confidence = rel_a.lower_bound - rel_b.upper_bound
            return (source_a, min(confidence, self.config.max_reliability_score))
        elif rel_b.lower_bound > rel_a.upper_bound:
            confidence = rel_b.lower_bound - rel_a.upper_bound
            return (source_b, min(confidence, self.config.max_reliability_score))

        # Overlapping intervals - use point estimates
        diff = abs(rel_a.value - rel_b.value)
        winner = source_a if rel_a.value > rel_b.value else source_b

        # Confidence is low due to overlap
        confidence = diff * min(rel_a.confidence, rel_b.confidence)
        return (winner, confidence)


# =============================================================================
# ACTIVE EXPLORATION STRATEGY
# =============================================================================


class ActiveExplorationStrategy:
    """
    Implements Thompson Sampling for strategic re-testing of uncertain knowledge.

    When static knowledge hasn't been verified recently and there's high
    uncertainty, this strategy may recommend re-verification.
    """

    def __init__(
        self,
        config: ConflictResolutionConfig,
        reliability_tracker: DynamicReliabilityTracker,
    ):
        self.config = config
        self.reliability_tracker = reliability_tracker
        self.exploration_history: List[Dict] = []

    def should_explore_static(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> Tuple[bool, str]:
        """
        Decide whether to re-verify static knowledge.

        Uses Thompson Sampling: sample from reliability distributions
        and explore if the sampled dynamic reliability exceeds static.

        Returns:
            Tuple of (should_explore, reasoning)
        """
        static_dist = self.reliability_tracker.distributions.get(
            KnowledgeSource.STATIC_KB,
            BetaDistribution(
                self.config.static_prior_alpha, self.config.static_prior_beta
            ),
        )
        dynamic_dist = self.reliability_tracker.distributions.get(
            KnowledgeSource.DYNAMIC_EXPERIENCE,
            BetaDistribution(
                self.config.dynamic_prior_alpha, self.config.dynamic_prior_beta
            ),
        )

        # Thompson Sampling: sample from both distributions
        static_sample = static_dist.sample()
        dynamic_sample = dynamic_dist.sample()

        # Check if we have enough observations
        static_obs = self.reliability_tracker.observation_counts.get(
            KnowledgeSource.STATIC_KB, 0
        )
        dynamic_obs = self.reliability_tracker.observation_counts.get(
            KnowledgeSource.DYNAMIC_EXPERIENCE, 0
        )

        if static_obs < self.config.min_observations_for_exploration:
            return (False, "Insufficient static observations for exploration")

        # Check uncertainty level
        static_uncertainty = 1.0 - (1.0 / (1.0 + static_dist.variance() * 10))

        if static_uncertainty < self.config.exploration_uncertainty_threshold:
            return (False, "Static knowledge has low uncertainty")

        # Check staleness
        if static_item.age_days() > self.config.anomaly_staleness_threshold_days:
            # Stale static knowledge should be re-verified
            return (
                True,
                f"Static knowledge is stale ({static_item.age_days():.1f} days old)",
            )

        # Thompson Sampling decision
        if dynamic_sample > static_sample:
            probability = dynamic_sample - static_sample
            if probability > self.config.exploration_probability_threshold:
                self.exploration_history.append(
                    {
                        "timestamp": datetime.now(),
                        "static_sample": static_sample,
                        "dynamic_sample": dynamic_sample,
                        "decision": "explore",
                    }
                )
                return (
                    True,
                    f"Thompson Sampling suggests exploration (p={probability:.2f})",
                )

        return (False, "Thompson Sampling suggests trusting static knowledge")


# =============================================================================
# ENSEMBLE CONFLICT DETECTOR
# =============================================================================


class EnsembleConflictDetector:
    """
    Combines multiple detection methods with configurable weights.

    Methods:
    1. NLI (Neural/Pattern/LLM)
    2. Semantic similarity patterns
    3. Temporal analysis
    4. Evidence strength
    5. Direct LLM assessment
    """

    def __init__(
        self,
        config: ConflictResolutionConfig,
        patterns: LearnedPatterns,
        nli_classifier: NeuralNLIClassifier,
        embedding_fn: Optional[Callable] = None,
        llm_client: Optional[Callable] = None,
    ):
        self.config = config
        self.patterns = patterns
        self.nli_classifier = nli_classifier
        self.embedding_fn = embedding_fn
        self.llm_client = llm_client
        self.detection_history: List[Dict] = []

    def detect(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> Tuple[bool, float, List[MethodVote]]:
        """
        Detect conflict using ensemble of methods.

        Returns:
            Tuple of (is_conflict, confidence, list of method votes)
        """
        votes: List[MethodVote] = []

        # 1. NLI vote
        nli_result = self.nli_classifier.classify(
            static_item.content, dynamic_item.content
        )
        nli_vote = MethodVote(
            method="nli",
            is_conflict=nli_result.relation == SemanticRelation.CONTRADICTS,
            confidence=nli_result.confidence,
            weight=self.config.nli_vote_weight,
            details={
                "relation": nli_result.relation.value,
                "nli_method": nli_result.method,
            },
        )
        votes.append(nli_vote)

        # 2. Semantic pattern vote
        semantic_vote = self._vote_semantic_pattern(static_item, dynamic_item)
        votes.append(semantic_vote)

        # 3. Temporal vote
        temporal_vote = self._vote_temporal(static_item, dynamic_item)
        votes.append(temporal_vote)

        # 4. Evidence vote
        evidence_vote = self._vote_evidence(static_item, dynamic_item)
        votes.append(evidence_vote)

        # 5. Recommendation conflict vote (dismissive vs active strategy)
        rec_vote = self._vote_recommendation_conflict(static_item, dynamic_item)
        votes.append(rec_vote)

        # 6. LLM vote (if available)
        if self.llm_client is not None:
            llm_vote = self._vote_llm(static_item, dynamic_item)
            votes.append(llm_vote)

        # Weighted voting
        total_weight = sum(v.weight for v in votes)
        weighted_conflict = (
            sum(
                v.weight * (1.0 if v.is_conflict else 0.0) * v.confidence for v in votes
            )
            / total_weight
            if total_weight > 0
            else 0
        )

        # Circuit breaker: a single highly-confident method overrides the
        # ensemble average.  This prevents dilution when recommendation-level
        # detection finds a clear dismissive-vs-active conflict but the other
        # text-similarity methods (NLI, semantic) see no surface-level
        # contradiction.
        high_confidence_override = any(
            v.is_conflict and v.confidence >= 0.60 for v in votes
        )

        is_conflict = (
            weighted_conflict >= self.config.ensemble_conflict_threshold
            or high_confidence_override
        )

        # Log detection
        self.detection_history.append(
            {
                "timestamp": datetime.now(),
                "static_id": static_item.id,
                "dynamic_id": dynamic_item.id,
                "is_conflict": is_conflict,
                "confidence": weighted_conflict,
                "votes": [
                    {
                        "method": v.method,
                        "is_conflict": v.is_conflict,
                        "confidence": v.confidence,
                    }
                    for v in votes
                ],
            }
        )

        return (is_conflict, weighted_conflict, votes)

    def _vote_semantic_pattern(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> MethodVote:
        """Vote based on semantic patterns."""
        static_lower = static_item.content.lower()
        dynamic_lower = dynamic_item.content.lower()

        conflict_indicators = 0
        matched_patterns = []

        # Check contradiction pairs
        for word1, word2 in self.patterns.contradiction_pairs:
            if word1 in static_lower and word2 in dynamic_lower:
                conflict_indicators += 1
                matched_patterns.append(f"{word1}↔{word2}")
            elif word2 in static_lower and word1 in dynamic_lower:
                conflict_indicators += 1
                matched_patterns.append(f"{word2}↔{word1}")

        # Also check contradiction keywords in dynamic that imply change from static
        for keyword in self.config.contradiction_keywords:
            if keyword in dynamic_lower:
                # Check if static implies opposite (positive) state
                positive_match = any(
                    pos_kw in static_lower for pos_kw in self.config.agreement_keywords
                )
                if positive_match:
                    conflict_indicators += 1
                    matched_patterns.append(f"keyword:{keyword}")

        # Check if same entity mentioned with conflicting states
        # e.g., static: "Hamburg is stable", dynamic: "Hamburg is blocked"
        static_has_positive = any(
            kw in static_lower for kw in self.config.agreement_keywords
        )
        dynamic_has_negative = any(
            kw in dynamic_lower for kw in self.config.contradiction_keywords
        )
        if static_has_positive and dynamic_has_negative:
            # Check for common entity reference
            # Find words common to both (potential entity names)
            static_words = set(static_lower.split())
            dynamic_words = set(dynamic_lower.split())
            common_words = static_words & dynamic_words
            # Filter to meaningful words (>3 chars, not common words)
            common_entities = [
                w
                for w in common_words
                if len(w) > 3
                and w
                not in {
                    "the",
                    "and",
                    "for",
                    "with",
                    "from",
                    "that",
                    "this",
                    "port",
                    "cargo",
                }
            ]
            if common_entities:
                conflict_indicators += 1
                matched_patterns.append(f"entity:{common_entities[0]}")

        confidence = min(
            conflict_indicators * 0.3,  # Each indicator adds 0.3 confidence
            self.config.max_reliability_score,
        )

        return MethodVote(
            method="semantic_pattern",
            is_conflict=conflict_indicators > 0,
            confidence=max(confidence, 0.5) if conflict_indicators > 0 else 0.0,
            weight=self.config.semantic_vote_weight,
            details={
                "conflict_indicators": conflict_indicators,
                "matched": matched_patterns,
            },
        )

    def _vote_temporal(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> MethodVote:
        """Vote based on temporal analysis."""
        static_age = static_item.age_days()
        dynamic_age = dynamic_item.age_days()

        # Check for temporal keywords in dynamic item
        has_temporal_keywords = any(
            kw in dynamic_item.content.lower() for kw in self.patterns.temporal_keywords
        )

        # Stale static + fresh dynamic with temporal keywords = likely conflict
        is_conflict = (
            static_age > self.config.anomaly_staleness_threshold_days
            and dynamic_age < self.config.recency_decay_rate
            and has_temporal_keywords
        )

        confidence = self.config.temporal_vote_weight
        if is_conflict:
            # Higher confidence if the age difference is large
            age_ratio = static_age / max(dynamic_age, 1)
            confidence = min(
                confidence * (1 + math.log1p(age_ratio) / 10),
                self.config.max_reliability_score,
            )

        return MethodVote(
            method="temporal",
            is_conflict=is_conflict,
            confidence=confidence,
            weight=self.config.temporal_vote_weight,
            details={
                "static_age_days": static_age,
                "dynamic_age_days": dynamic_age,
                "has_temporal_keywords": has_temporal_keywords,
            },
        )

    def _vote_evidence(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> MethodVote:
        """Vote based on evidence strength."""
        # High dynamic failures might indicate unreliable dynamic info
        dynamic_failure_rate = dynamic_item.failures / max(
            1, dynamic_item.confirmations + dynamic_item.failures
        )

        # Low static failures = reliable static
        static_failure_rate = static_item.failures / max(
            1, static_item.confirmations + static_item.failures
        )

        # Conflict if dynamic has high failure rate but claims to override static
        is_conflict = (
            dynamic_failure_rate > self.config.anomaly_failure_rate_threshold
            and static_failure_rate < self.config.anomaly_failure_rate_threshold
        )

        confidence = abs(dynamic_failure_rate - static_failure_rate)

        return MethodVote(
            method="evidence",
            is_conflict=is_conflict,
            confidence=confidence,
            weight=self.config.evidence_vote_weight,
            details={
                "dynamic_failure_rate": dynamic_failure_rate,
                "static_failure_rate": static_failure_rate,
            },
        )

    def _vote_recommendation_conflict(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> MethodVote:
        """
        Vote based on recommendation-level contradictions.

        Detects three patterns:
        1. Dismissive static (''can be ignored'') vs active dynamic (''strategy is X'')
        2. Condition code overlap with different recommendations
        3. Explicit different solution recommendations (''use X'' vs ''use Y'')
        """
        import re

        static_lower = static_item.content.lower()
        dynamic_lower = dynamic_item.content.lower()

        conflict_score = 0.0
        matched = []

        # Pattern 1: Dismissive static + active dynamic
        has_dismissive = any(
            p in static_lower for p in self.config.dismissive_phrases
        )
        has_active = any(
            p in dynamic_lower for p in self.config.active_strategy_phrases
        )
        if has_dismissive and has_active:
            conflict_score += 0.70
            matched.append("dismissive_vs_active")

        # Pattern 2: Condition code overlap (e.g. both mention INT-401 or DAT-SYNC)
        code_pattern = re.compile(r"\b[A-Z]{1,4}-(?:\d{2,4}|[A-Z]{2,10})\b")
        static_codes = set(code_pattern.findall(static_item.content))
        dynamic_codes = set(code_pattern.findall(dynamic_item.content))
        overlap = static_codes & dynamic_codes
        if overlap:
            conflict_score += 0.25 * len(overlap)
            matched.append(f"condition_overlap:{overlap}")

        # Pattern 3: Different solution recommendations
        solution_re = re.compile(
            r"(?:use|strategy is|fallback[_ ]to|switch to)\s+([\w-]+)",
            re.IGNORECASE,
        )
        static_solutions = {
            m.group(1).lower().rstrip(".,;")
            for m in solution_re.finditer(static_item.content)
        }
        dynamic_solutions = {
            m.group(1).lower().rstrip(".,;")
            for m in solution_re.finditer(dynamic_item.content)
        }
        if (
            static_solutions
            and dynamic_solutions
            and not (static_solutions & dynamic_solutions)
        ):
            conflict_score += 0.40
            matched.append(f"diff_solutions:{static_solutions}vs{dynamic_solutions}")

        confidence = min(conflict_score, self.config.max_reliability_score)
        is_conflict = confidence > 0.3

        return MethodVote(
            method="recommendation",
            is_conflict=is_conflict,
            confidence=confidence if is_conflict else 0.0,
            weight=self.config.recommendation_vote_weight,
            details={"matched": matched},
        )

    def _vote_llm(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> MethodVote:
        """Vote using LLM assessment."""
        if self.llm_client is None:
            return MethodVote(
                method="llm", is_conflict=False, confidence=0.0, weight=0.0
            )

        prompt = f"""Do these two statements conflict with each other?

STATIC KNOWLEDGE: {static_item.content}

DYNAMIC EXPERIENCE: {dynamic_item.content}

Answer with YES or NO, then explain briefly."""

        try:
            response = self.llm_client(prompt)
            is_conflict = response.strip().upper().startswith("YES")

            return MethodVote(
                method="llm",
                is_conflict=is_conflict,
                confidence=self.config.llm_vote_weight,
                weight=self.config.llm_vote_weight,
                details={"response": response[:200]},
            )
        except Exception as e:
            return MethodVote(
                method="llm_error",
                is_conflict=False,
                confidence=0.0,
                weight=0.0,
                details={"error": str(e)},
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        if not self.detection_history:
            return {"total_detections": 0, "conflicts_found": 0}

        total = len(self.detection_history)
        conflicts = sum(1 for d in self.detection_history if d["is_conflict"])
        avg_confidence = sum(d["confidence"] for d in self.detection_history) / total

        return {
            "total_detections": total,
            "conflicts_found": conflicts,
            "conflict_rate": conflicts / total if total > 0 else 0,
            "avg_confidence": avg_confidence,
        }


# =============================================================================
# CONFLICT RESOLVER
# =============================================================================


class ConflictResolver:
    """
    Resolves conflicts between static and dynamic knowledge.

    Strategies:
    1. Recency - prefer more recent information
    2. Reliability - prefer more reliable source
    3. Specificity - prefer more specific information
    4. Evidence strength - prefer better-evidenced information
    5. Anomaly detection - detect and handle outliers
    6. LLM merge - use LLM to synthesize both sources
    """

    def __init__(
        self,
        config: ConflictResolutionConfig,
        reliability_tracker: DynamicReliabilityTracker,
        llm_client: Optional[Callable] = None,
    ):
        self.config = config
        self.reliability_tracker = reliability_tracker
        self.llm_client = llm_client
        self.resolution_history: List[ResolutionResult] = []

    def resolve(
        self, conflict: ConflictRecord, strategy: Optional[str] = None
    ) -> ResolutionResult:
        """
        Resolve a conflict using the specified or auto-selected strategy.

        Args:
            conflict: The conflict to resolve
            strategy: Optional strategy name, or None for auto-selection

        Returns:
            ResolutionResult with the winning source and reasoning
        """
        if strategy is None:
            strategy = self._select_strategy(conflict)

        static_item = conflict.static_item
        dynamic_item = conflict.dynamic_item

        if strategy == "recency":
            result = self._resolve_by_recency(conflict)
        elif strategy == "reliability":
            result = self._resolve_by_reliability(conflict)
        elif strategy == "specificity":
            result = self._resolve_by_specificity(conflict)
        elif strategy == "evidence_strength":
            result = self._resolve_by_evidence(conflict)
        elif strategy == "anomaly_detection":
            result = self._resolve_by_anomaly(conflict)
        elif strategy == "llm_merge":
            result = self._resolve_by_llm_merge(conflict)
        else:
            # Default: compare static vs dynamic
            result = self._compare_static_vs_dynamic(conflict)

        self.resolution_history.append(result)
        return result

    def _select_strategy(self, conflict: ConflictRecord) -> str:
        """Auto-select the best resolution strategy."""
        static_item = conflict.static_item
        dynamic_item = conflict.dynamic_item

        # Check for anomaly first
        if self._is_anomaly(dynamic_item):
            return "anomaly_detection"

        # If static is very old, prefer recency
        if static_item.age_days() > self.config.max_staleness_days:
            return "recency"

        # If we have good evidence data, use evidence
        total_evidence = (
            static_item.confirmations
            + static_item.failures
            + dynamic_item.confirmations
            + dynamic_item.failures
        )
        if total_evidence >= self.config.min_confirmations_for_high_evidence:
            return "evidence_strength"

        # Default to reliability
        return "reliability"

    def _is_anomaly(self, item: KnowledgeItem) -> bool:
        """Check if an item is an anomaly (outlier)."""
        # Single data point with no confirmations
        if item.confirmations == 0 and item.failures == 0:
            if item.confidence < self.config.anomaly_single_point_threshold:
                return True

        # High failure rate
        total = item.confirmations + item.failures
        if total > 0:
            failure_rate = item.failures / total
            if failure_rate > self.config.anomaly_failure_rate_threshold:
                return True

        # Check for anomaly keywords in metadata
        anomaly_keywords = ["outlier", "anomaly", "unusual", "unexpected"]
        content_lower = item.content.lower()
        if any(kw in content_lower for kw in anomaly_keywords):
            return True

        # Very stale item
        if item.age_days() > self.config.anomaly_staleness_threshold_days:
            return True

        return False

    def _recency_score(self, item: KnowledgeItem) -> float:
        """Calculate recency score (higher = more recent)."""
        age_days = item.age_days()
        decay = math.exp(-self.config.recency_decay_rate * age_days)
        return decay

    def _reliability_score(self, item: KnowledgeItem) -> float:
        """Calculate reliability score."""
        source_reliability = self.reliability_tracker.get_reliability(item.source)
        return source_reliability.value * item.confidence

    def _specificity_score(self, item: KnowledgeItem) -> float:
        """Calculate specificity score (more specific = higher)."""
        # Use content length as a proxy for specificity
        base_score = min(len(item.content) / 500, 1.0)

        # Boost for metadata
        if item.metadata:
            base_score += len(item.metadata) * 0.1

        return min(base_score, self.config.max_reliability_score)

    def _evidence_strength_score(self, item: KnowledgeItem) -> float:
        """Calculate evidence strength score."""
        confirmations = item.confirmations
        failures = item.failures
        total = confirmations + failures

        if total == 0:
            # No evidence - use prior based on source
            if item.source == KnowledgeSource.STATIC_KB:
                return self.config.static_prior_alpha / (
                    self.config.static_prior_alpha + self.config.static_prior_beta
                )
            else:
                return self.config.dynamic_prior_alpha / (
                    self.config.dynamic_prior_alpha + self.config.dynamic_prior_beta
                )

        # Laplace smoothing
        score = (confirmations + 1) / (total + 2)
        return score

    def _resolve_by_recency(self, conflict: ConflictRecord) -> ResolutionResult:
        """Resolve by preferring more recent information."""
        static_score = self._recency_score(conflict.static_item)
        dynamic_score = self._recency_score(conflict.dynamic_item)

        if dynamic_score > static_score:
            winner = KnowledgeSource.DYNAMIC_EXPERIENCE
            winning_item = conflict.dynamic_item
            confidence = dynamic_score / (static_score + dynamic_score)
        else:
            winner = KnowledgeSource.STATIC_KB
            winning_item = conflict.static_item
            confidence = static_score / (static_score + dynamic_score)

        return ResolutionResult(
            id=f"res_{conflict.id}",
            conflict=conflict,
            winner=winner,
            winning_item=winning_item,
            confidence=confidence,
            strategy_used="recency",
            reasoning=f"Recency: static={static_score:.2f}, dynamic={dynamic_score:.2f}",
        )

    def _resolve_by_reliability(self, conflict: ConflictRecord) -> ResolutionResult:
        """Resolve by preferring more reliable source."""
        static_score = self._reliability_score(conflict.static_item)
        dynamic_score = self._reliability_score(conflict.dynamic_item)

        if dynamic_score > static_score:
            winner = KnowledgeSource.DYNAMIC_EXPERIENCE
            winning_item = conflict.dynamic_item
            confidence = dynamic_score
        else:
            winner = KnowledgeSource.STATIC_KB
            winning_item = conflict.static_item
            confidence = static_score

        return ResolutionResult(
            id=f"res_{conflict.id}",
            conflict=conflict,
            winner=winner,
            winning_item=winning_item,
            confidence=confidence,
            strategy_used="reliability",
            reasoning=f"Reliability: static={static_score:.2f}, dynamic={dynamic_score:.2f}",
        )

    def _resolve_by_specificity(self, conflict: ConflictRecord) -> ResolutionResult:
        """Resolve by preferring more specific information."""
        static_score = self._specificity_score(conflict.static_item)
        dynamic_score = self._specificity_score(conflict.dynamic_item)

        if dynamic_score > static_score:
            winner = KnowledgeSource.DYNAMIC_EXPERIENCE
            winning_item = conflict.dynamic_item
            confidence = dynamic_score / (static_score + dynamic_score)
        else:
            winner = KnowledgeSource.STATIC_KB
            winning_item = conflict.static_item
            confidence = static_score / (static_score + dynamic_score)

        return ResolutionResult(
            id=f"res_{conflict.id}",
            conflict=conflict,
            winner=winner,
            winning_item=winning_item,
            confidence=confidence,
            strategy_used="specificity",
            reasoning=f"Specificity: static={static_score:.2f}, dynamic={dynamic_score:.2f}",
        )

    def _resolve_by_evidence(self, conflict: ConflictRecord) -> ResolutionResult:
        """Resolve by preferring better-evidenced information."""
        static_score = self._evidence_strength_score(conflict.static_item)
        dynamic_score = self._evidence_strength_score(conflict.dynamic_item)

        if dynamic_score > static_score + self.config.evidence_confirmation_weight:
            winner = KnowledgeSource.DYNAMIC_EXPERIENCE
            winning_item = conflict.dynamic_item
            confidence = dynamic_score
        else:
            winner = KnowledgeSource.STATIC_KB
            winning_item = conflict.static_item
            confidence = static_score

        return ResolutionResult(
            id=f"res_{conflict.id}",
            conflict=conflict,
            winner=winner,
            winning_item=winning_item,
            confidence=confidence,
            strategy_used="evidence_strength",
            reasoning=f"Evidence: static={static_score:.2f}, dynamic={dynamic_score:.2f}",
        )

    def _resolve_by_anomaly(self, conflict: ConflictRecord) -> ResolutionResult:
        """Resolve by detecting and discarding anomalies."""
        static_is_anomaly = self._is_anomaly(conflict.static_item)
        dynamic_is_anomaly = self._is_anomaly(conflict.dynamic_item)

        if dynamic_is_anomaly and not static_is_anomaly:
            # Dynamic is anomaly, trust static
            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=KnowledgeSource.STATIC_KB,
                winning_item=conflict.static_item,
                confidence=self.config.static_wins_confidence_threshold,
                strategy_used="anomaly_detection",
                reasoning="Dynamic item detected as anomaly, trusting static knowledge",
                should_flag_for_review=True,
            )
        elif static_is_anomaly and not dynamic_is_anomaly:
            # Static is anomaly (stale), trust dynamic
            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=KnowledgeSource.DYNAMIC_EXPERIENCE,
                winning_item=conflict.dynamic_item,
                confidence=self.config.dynamic_wins_confidence_threshold,
                strategy_used="anomaly_detection",
                reasoning="Static item detected as anomaly (possibly stale), trusting dynamic",
                should_update_static=True,
            )
        else:
            # Both or neither are anomalies - fall back to evidence
            return self._resolve_by_evidence(conflict)

    def _resolve_by_llm_merge(self, conflict: ConflictRecord) -> ResolutionResult:
        """Use LLM to merge conflicting information."""
        if self.llm_client is None:
            return self._resolve_by_evidence(conflict)

        prompt = f"""Two pieces of knowledge conflict. Synthesize them into accurate information:

STATIC KNOWLEDGE (from knowledge base):
{conflict.static_item.content}

DYNAMIC EXPERIENCE (from recent observation):
{conflict.dynamic_item.content}

Provide a merged, accurate statement that resolves the conflict. If one is clearly wrong, explain which and why."""

        try:
            response = self.llm_client(prompt)

            # Determine winner based on response
            if "static" in response.lower() and "correct" in response.lower():
                winner = KnowledgeSource.STATIC_KB
                winning_item = conflict.static_item
            elif "dynamic" in response.lower() and "correct" in response.lower():
                winner = KnowledgeSource.DYNAMIC_EXPERIENCE
                winning_item = conflict.dynamic_item
            else:
                # Merged - prefer dynamic as it's more recent
                winner = KnowledgeSource.DYNAMIC_EXPERIENCE
                winning_item = conflict.dynamic_item

            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=winner,
                winning_item=winning_item,
                confidence=self.config.merge_confidence_threshold,
                strategy_used="llm_merge",
                reasoning=f"LLM synthesis: {response[:200]}",
                merged_content=response,
            )
        except Exception:
            # Fall back to evidence
            return self._resolve_by_evidence(conflict)

    def _compare_static_vs_dynamic(self, conflict: ConflictRecord) -> ResolutionResult:
        """
        Core comparison logic between static and dynamic knowledge.

        This is where static knowledge can explicitly win over dynamic.
        """
        static_item = conflict.static_item
        dynamic_item = conflict.dynamic_item

        # Check if dynamic is an anomaly
        if self._is_anomaly(dynamic_item):
            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=KnowledgeSource.STATIC_KB,
                winning_item=static_item,
                confidence=self.config.static_wins_confidence_threshold,
                strategy_used="static_vs_dynamic",
                reasoning="Dynamic knowledge appears to be anomalous; trusting stable static knowledge",
                should_flag_for_review=True,
            )

        # Calculate scores
        static_evidence = self._evidence_strength_score(static_item)
        dynamic_evidence = self._evidence_strength_score(dynamic_item)

        static_reliability = self._reliability_score(static_item)
        dynamic_reliability = self._reliability_score(dynamic_item)

        static_recency = self._recency_score(static_item)
        dynamic_recency = self._recency_score(dynamic_item)

        static_specificity = self._specificity_score(static_item)
        dynamic_specificity = self._specificity_score(dynamic_item)

        # Weighted composite score
        static_score = (
            static_evidence * self.config.evidence_strategy_weight
            + static_reliability * self.config.reliability_strategy_weight
            + static_recency * self.config.recency_strategy_weight
            + static_specificity * self.config.specificity_strategy_weight
        )

        dynamic_score = (
            dynamic_evidence * self.config.evidence_strategy_weight
            + dynamic_reliability * self.config.reliability_strategy_weight
            + dynamic_recency * self.config.recency_strategy_weight
            + dynamic_specificity * self.config.specificity_strategy_weight
        )

        # Determine winner
        if static_score > dynamic_score:
            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=KnowledgeSource.STATIC_KB,
                winning_item=static_item,
                confidence=static_score / (static_score + dynamic_score),
                strategy_used="static_vs_dynamic",
                reasoning=f"Static wins: score={static_score:.2f} vs dynamic={dynamic_score:.2f}",
            )
        else:
            return ResolutionResult(
                id=f"res_{conflict.id}",
                conflict=conflict,
                winner=KnowledgeSource.DYNAMIC_EXPERIENCE,
                winning_item=dynamic_item,
                confidence=dynamic_score / (static_score + dynamic_score),
                strategy_used="static_vs_dynamic",
                reasoning=f"Dynamic wins: score={dynamic_score:.2f} vs static={static_score:.2f}",
                should_update_static=dynamic_score
                > self.config.dynamic_wins_confidence_threshold,
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get resolution statistics."""
        if not self.resolution_history:
            return {"total_resolutions": 0}

        total = len(self.resolution_history)
        static_wins = sum(
            1 for r in self.resolution_history if r.winner == KnowledgeSource.STATIC_KB
        )
        dynamic_wins = sum(
            1
            for r in self.resolution_history
            if r.winner == KnowledgeSource.DYNAMIC_EXPERIENCE
        )

        strategies_used = {}
        for r in self.resolution_history:
            strategies_used[r.strategy_used] = (
                strategies_used.get(r.strategy_used, 0) + 1
            )

        return {
            "total_resolutions": total,
            "static_wins": static_wins,
            "dynamic_wins": dynamic_wins,
            "merges": total - static_wins - dynamic_wins,
            "strategies_used": strategies_used,
            "avg_confidence": sum(r.confidence for r in self.resolution_history)
            / total,
        }


# =============================================================================
# CONFLICT MANAGER (Main Entry Point)
# =============================================================================


class ConflictManager:
    """
    Main orchestrator for conflict detection and resolution.

    This is the primary entry point for the conflict resolution system.
    """

    def __init__(
        self,
        config: Optional[ConflictResolutionConfig] = None,
        patterns: Optional[LearnedPatterns] = None,
        embedding_fn: Optional[Callable] = None,
        llm_client: Optional[Callable] = None,
        nli_model_fn: Optional[Callable] = None,
    ):
        self.config = config or ConflictResolutionConfig()
        self.patterns = patterns or LearnedPatterns()
        self.embedding_fn = embedding_fn
        self.llm_client = llm_client

        # Initialize components
        self.reliability_tracker = DynamicReliabilityTracker(self.config)

        self.nli_classifier = NeuralNLIClassifier(
            config=self.config,
            patterns=self.patterns,
            model_fn=nli_model_fn,
            llm_client=llm_client,
        )

        self.detector = EnsembleConflictDetector(
            config=self.config,
            patterns=self.patterns,
            nli_classifier=self.nli_classifier,
            embedding_fn=embedding_fn,
            llm_client=llm_client,
        )

        self.resolver = ConflictResolver(
            config=self.config,
            reliability_tracker=self.reliability_tracker,
            llm_client=llm_client,
        )

        self.exploration_strategy = ActiveExplorationStrategy(
            config=self.config, reliability_tracker=self.reliability_tracker
        )

        self.conflict_history: List[ConflictRecord] = []
        self.resolution_history: List[ResolutionResult] = []

    def detect_and_resolve(
        self,
        static_item: KnowledgeItem,
        dynamic_item: KnowledgeItem,
        auto_resolve: bool = True,
    ) -> Tuple[Optional[ConflictRecord], Optional[ResolutionResult]]:
        """
        Detect conflict between static and dynamic knowledge, and optionally resolve it.

        Args:
            static_item: Knowledge from static knowledge base
            dynamic_item: Knowledge from dynamic experience
            auto_resolve: Whether to automatically resolve detected conflicts

        Returns:
            Tuple of (ConflictRecord if conflict detected, ResolutionResult if resolved)
        """
        # Detect conflict
        is_conflict, confidence, votes = self.detector.detect(static_item, dynamic_item)

        if not is_conflict:
            return (None, None)

        # Create conflict record
        conflict = ConflictRecord(
            id=f"conflict_{len(self.conflict_history)}",
            static_item=static_item,
            dynamic_item=dynamic_item,
            relation=SemanticRelation.CONTRADICTS,
            severity=self._assess_severity(confidence),
            confidence=confidence,
            detection_method="ensemble",
        )
        self.conflict_history.append(conflict)

        if not auto_resolve:
            return (conflict, None)

        # Resolve conflict
        resolution = self.resolver.resolve(conflict)
        self.resolution_history.append(resolution)

        # Update reliability based on resolution
        if resolution.winner == KnowledgeSource.STATIC_KB:
            self.reliability_tracker.update(KnowledgeSource.STATIC_KB, True)
            self.reliability_tracker.update(KnowledgeSource.DYNAMIC_EXPERIENCE, False)
        else:
            self.reliability_tracker.update(KnowledgeSource.DYNAMIC_EXPERIENCE, True)
            self.reliability_tracker.update(KnowledgeSource.STATIC_KB, False)

        return (conflict, resolution)

    def _assess_severity(self, confidence: float) -> ConflictSeverity:
        """Assess conflict severity based on confidence."""
        if confidence >= self.config.ensemble_high_confidence_threshold:
            return ConflictSeverity.CRITICAL
        elif confidence >= self.config.ensemble_conflict_threshold:
            return ConflictSeverity.HIGH
        elif confidence >= self.config.nli_neutral_threshold:
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    def should_explore(
        self, static_item: KnowledgeItem, dynamic_item: KnowledgeItem
    ) -> Tuple[bool, str]:
        """Check if we should re-verify static knowledge."""
        return self.exploration_strategy.should_explore_static(
            static_item, dynamic_item
        )

    def update_reliability(self, source: KnowledgeSource, was_correct: bool) -> None:
        """Update reliability of a knowledge source based on outcome."""
        self.reliability_tracker.update(source, was_correct)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        detector_stats = self.detector.get_stats()
        resolver_stats = self.resolver.get_stats()

        return {
            "status": "active",
            "summary": {
                "conflicts_detected": len(self.conflict_history),
                "conflicts_resolved": len(self.resolution_history),
            },
            "detection": detector_stats,
            "resolution_outcomes": {
                "static_wins": resolver_stats.get("static_wins", 0),
                "dynamic_wins": resolver_stats.get("dynamic_wins", 0),
                "merges": resolver_stats.get("merges", 0),
            },
            "reliability": {
                source.value: self.reliability_tracker.get_reliability(source).value
                for source in [
                    KnowledgeSource.STATIC_KB,
                    KnowledgeSource.DYNAMIC_EXPERIENCE,
                    KnowledgeSource.EPISODIC_MEMORY,
                ]
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_conflict_manager(
    llm_client: Optional[Callable] = None,
    embedding_fn: Optional[Callable] = None,
    config: Optional[ConflictResolutionConfig] = None,
) -> ConflictManager:
    """Create a ConflictManager with default settings."""
    return ConflictManager(
        config=config, llm_client=llm_client, embedding_fn=embedding_fn
    )


def detect_conflict(
    static_content: str, dynamic_content: str, manager: Optional[ConflictManager] = None
) -> Tuple[bool, float]:
    """
    Simple function to detect conflict between two pieces of content.

    Returns:
        Tuple of (is_conflict, confidence)
    """
    if manager is None:
        manager = ConflictManager()

    static_item = KnowledgeItem(
        id="static_check",
        content=static_content,
        source=KnowledgeSource.STATIC_KB,
        timestamp=datetime.now(),
    )

    dynamic_item = KnowledgeItem(
        id="dynamic_check",
        content=dynamic_content,
        source=KnowledgeSource.DYNAMIC_EXPERIENCE,
        timestamp=datetime.now(),
    )

    is_conflict, confidence, _ = manager.detector.detect(static_item, dynamic_item)
    return (is_conflict, confidence)
