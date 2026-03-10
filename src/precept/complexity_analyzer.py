"""
Generalized Complexity Analyzer for PRECEPT.

This module extends COMPASS's ML-based query analysis to handle:
1. Tool execution chains (tool A → tool B → tool C)
2. Retrieval hops (document A → document B → document C)
3. Reasoning steps (premise → inference → conclusion)
4. API call sequences (check → validate → execute)
5. Memory lookups (retrieve → verify → apply)

Key COMPASS Advantages Integrated:
- ML-based complexity detection (adaptive vs fixed)
- Smart rollout allocation (2 vs 15 rollouts)
- Multi-strategy approach (vector + keyword + graph → tool + retrieval + reason)
- Caching for efficiency
- Early stopping when confident
"""

import re
import logging
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# TASK COMPLEXITY TYPES
# =============================================================================

class ComplexityDimension(Enum):
    """Dimensions of task complexity in PRECEPT."""
    RETRIEVAL = "retrieval"       # Information retrieval hops
    TOOL_USE = "tool_use"         # Tool execution chains
    REASONING = "reasoning"        # Logical reasoning steps
    API_CALLS = "api_calls"       # External API sequences
    MEMORY = "memory"             # Memory lookup chains
    VERIFICATION = "verification" # Validation steps


@dataclass
class ComplexityEstimate:
    """Estimate of task complexity across multiple dimensions."""
    
    # Estimated steps per dimension
    retrieval_hops: int = 1
    tool_steps: int = 1
    reasoning_steps: int = 1
    api_calls: int = 1
    memory_lookups: int = 1
    verification_steps: int = 1
    
    # Aggregates
    total_estimated_steps: int = 0
    confidence: float = 0.5
    
    # Analysis
    dominant_dimension: ComplexityDimension = ComplexityDimension.REASONING
    reasoning_path: List[str] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)
    detected_tools: List[str] = field(default_factory=list)
    detected_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.total_estimated_steps == 0:
            self.total_estimated_steps = max(
                self.retrieval_hops,
                self.tool_steps,
                self.reasoning_steps,
                self.api_calls,
                self.memory_lookups,
                self.verification_steps,
            )


@dataclass
class RolloutDecision:
    """Decision about rollout allocation (from COMPASS smart rollouts)."""
    use_rollouts: bool
    num_rollouts: int
    focus: str  # "diversity", "consistency", "exploration", "skip"
    reason: str
    early_stop_threshold: float = 0.98


# =============================================================================
# PATTERN DETECTORS
# =============================================================================

class ToolPatternDetector:
    """Detect tool-related patterns in tasks."""
    
    # Common tool action patterns
    TOOL_PATTERNS = {
        "check": ["check", "verify", "validate", "inspect", "examine"],
        "query": ["find", "search", "lookup", "retrieve", "get"],
        "execute": ["book", "order", "schedule", "reserve", "submit"],
        "analyze": ["analyze", "compare", "evaluate", "assess"],
        "transform": ["convert", "format", "parse", "extract"],
        "communicate": ["send", "notify", "alert", "report"],
    }
    
    # Tool dependency patterns (if A, then likely need B)
    TOOL_DEPENDENCIES = {
        "check_port": ["calculate_route", "book_shipment"],
        "check_carrier": ["book_shipment"],
        "check_compliance": ["book_shipment"],
        "authenticate": ["*"],  # Required before any action
        "validate": ["execute"],
    }
    
    @classmethod
    def detect_tools(cls, task: str) -> List[str]:
        """Detect likely tools needed for task."""
        task_lower = task.lower()
        detected = []
        
        for tool_type, patterns in cls.TOOL_PATTERNS.items():
            if any(p in task_lower for p in patterns):
                detected.append(tool_type)
        
        return detected
    
    @classmethod
    def estimate_tool_chain(cls, task: str, available_tools: Optional[List[str]] = None) -> int:
        """Estimate the length of tool chain needed."""
        detected = cls.detect_tools(task)
        
        if not detected:
            return 1
        
        # Count dependencies
        chain_length = len(detected)
        for tool in detected:
            deps = cls.TOOL_DEPENDENCIES.get(tool, [])
            if "*" in deps:  # Wildcard dependency
                chain_length += 1
            else:
                chain_length += len([d for d in deps if d in detected])
        
        return min(chain_length, 10)  # Cap at 10


class ReasoningPatternDetector:
    """Detect reasoning complexity patterns."""
    
    # Reasoning complexity indicators
    COMPLEXITY_INDICATORS = {
        "high": [
            "however", "although", "despite", "whereas", "nevertheless",
            "on the other hand", "in contrast", "conversely",
            "if and only if", "necessary and sufficient",
        ],
        "medium": [
            "because", "therefore", "thus", "hence", "consequently",
            "as a result", "leads to", "implies", "suggests",
        ],
        "conditional": [
            "if", "when", "unless", "provided that", "assuming",
            "in case", "given that", "suppose",
        ],
        "comparative": [
            "more than", "less than", "compared to", "versus",
            "better", "worse", "faster", "slower", "larger", "smaller",
        ],
        "temporal": [
            "before", "after", "during", "while", "until",
            "since", "following", "prior to",
        ],
    }
    
    @classmethod
    def estimate_reasoning_steps(cls, task: str) -> Tuple[int, List[str]]:
        """Estimate reasoning steps needed."""
        task_lower = task.lower()
        detected_patterns = []
        complexity_score = 1
        
        for complexity_type, patterns in cls.COMPLEXITY_INDICATORS.items():
            matches = [p for p in patterns if p in task_lower]
            if matches:
                detected_patterns.extend(matches)
                if complexity_type == "high":
                    complexity_score += 2
                elif complexity_type == "medium":
                    complexity_score += 1
                elif complexity_type in ["conditional", "comparative"]:
                    complexity_score += 1
        
        # Count question marks and sub-clauses
        complexity_score += task.count("?")
        complexity_score += len(re.findall(r",\s*(and|or|but)", task_lower))
        
        return min(complexity_score, 8), detected_patterns


class EntityPatternDetector:
    """Detect entities and relationships."""
    
    # Entity patterns
    ENTITY_PATTERNS = [
        r"(?:from|to|at|in|on)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Locations
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:port|carrier|company|airport)",  # Organizations
        r"(?:via|through|using)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Routes
    ]
    
    # Relationship indicators
    RELATIONSHIP_INDICATORS = [
        "from", "to", "via", "through", "using", "with",
        "instead of", "alternative to", "replacing",
    ]
    
    @classmethod
    def extract_entities(cls, task: str) -> List[str]:
        """Extract named entities from task."""
        entities = []
        for pattern in cls.ENTITY_PATTERNS:
            matches = re.findall(pattern, task)
            entities.extend(matches)
        
        # Also look for capitalized words
        capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", task)
        entities.extend([e for e in capitalized if len(e) > 2])
        
        return list(set(entities))
    
    @classmethod
    def count_relationships(cls, task: str) -> int:
        """Count relationship indicators suggesting multi-hop."""
        task_lower = task.lower()
        return sum(1 for ind in cls.RELATIONSHIP_INDICATORS if ind in task_lower)


# =============================================================================
# GENERALIZED COMPLEXITY ANALYZER
# =============================================================================

class PRECEPTComplexityAnalyzer:
    """
    Generalized complexity analyzer for PRECEPT tasks.
    
    Extends COMPASS's ML hop detection to handle:
    - Tool chains
    - Reasoning complexity
    - API sequences
    - Memory lookups
    
    Key COMPASS features integrated:
    - Adaptive step estimation (not fixed)
    - Learning from history
    - Confidence calibration
    """
    
    def __init__(
        self,
        use_ml: bool = True,
        cache_enabled: bool = True,
        learning_enabled: bool = True,
    ):
        self.use_ml = use_ml
        self.cache_enabled = cache_enabled
        self.learning_enabled = learning_enabled
        
        # Caching (COMPASS advantage: query result caching)
        self._cache: Dict[str, ComplexityEstimate] = {}
        
        # Learning from history (COMPASS advantage: continuous learning)
        self.task_history: List[Dict[str, Any]] = []
        self.successful_patterns: Dict[str, int] = defaultdict(int)
        self.step_adjustments: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Try to import COMPASS ML analyzer
        self._compass_analyzer = None
        self._init_compass_ml()
        
        # Statistics
        self.stats = {
            "analyses_performed": 0,
            "cache_hits": 0,
            "ml_analyses": 0,
            "pattern_analyses": 0,
            "history_adjustments": 0,
        }
    
    def _init_compass_ml(self):
        """Initialize COMPASS ML analyzer if available."""
        self._ml_init_attempted = True
        self._ml_init_success = False
        self._ml_init_error = None
        
        try:
            # Try COMPASS ML analyzer (from compass library)
            from compass.ml_query_analyzer import MLQueryAnalyzer
            self._compass_analyzer = MLQueryAnalyzer()
            self._ml_init_success = True
            logger.info("✓ COMPASS ML analyzer initialized")
        except ImportError:
            # COMPASS library not installed - use pattern-based (still effective)
            self._ml_init_error = "COMPASS library not installed"
            logger.info("COMPASS ML analyzer not available, using pattern-based analysis")
        except Exception as e:
            self._ml_init_error = str(e)
            logger.warning(f"Could not initialize COMPASS ML: {e}")
    
    @property
    def ml_initialized(self) -> bool:
        """Check if ML analyzer was successfully initialized."""
        return self._ml_init_success
    
    @property
    def ml_status(self) -> Dict[str, Any]:
        """Get detailed ML initialization status."""
        return {
            "attempted": getattr(self, '_ml_init_attempted', False),
            "success": getattr(self, '_ml_init_success', False),
            "error": getattr(self, '_ml_init_error', None),
            "using_ml": self._compass_analyzer is not None,
            "fallback_active": self._compass_analyzer is None,
        }
    
    def _get_cache_key(self, task: str) -> str:
        """Generate cache key for task."""
        return hashlib.md5(task.encode()).hexdigest()
    
    def analyze(
        self,
        task: str,
        goal: Optional[str] = None,
        available_tools: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> ComplexityEstimate:
        """
        Analyze task complexity across all dimensions.
        
        Args:
            task: The task description
            goal: Optional goal description
            available_tools: List of available tools
            domain: Optional domain context
        
        Returns:
            ComplexityEstimate with multi-dimensional analysis
        """
        self.stats["analyses_performed"] += 1
        
        # Check cache (COMPASS advantage: caching)
        if self.cache_enabled:
            cache_key = self._get_cache_key(task)
            if cache_key in self._cache:
                self.stats["cache_hits"] += 1
                return self._cache[cache_key]
        
        # Combine task and goal for analysis
        full_text = f"{task} {goal or ''}"
        
        # Step 1: Tool chain estimation
        tool_steps = ToolPatternDetector.estimate_tool_chain(task, available_tools)
        detected_tools = ToolPatternDetector.detect_tools(task)
        
        # Step 2: Reasoning complexity
        reasoning_steps, reasoning_patterns = ReasoningPatternDetector.estimate_reasoning_steps(full_text)
        
        # Step 3: Entity extraction (for retrieval estimation)
        entities = EntityPatternDetector.extract_entities(task)
        relationships = EntityPatternDetector.count_relationships(task)
        retrieval_hops = max(1, len(entities) // 2 + relationships)
        
        # Step 4: ML-based analysis (COMPASS advantage: ML hop detection)
        ml_estimate = None
        ml_confidence = 0.5
        if self.use_ml and self._compass_analyzer:
            try:
                ml_result = self._compass_analyzer.analyze_query(task)
                ml_estimate = ml_result.estimated_hops
                ml_confidence = ml_result.confidence
                retrieval_hops = ml_estimate  # Use ML estimate for retrieval
                self.stats["ml_analyses"] += 1
            except Exception as e:
                logger.debug(f"ML analysis failed: {e}")
        else:
            self.stats["pattern_analyses"] += 1
        
        # Step 5: Determine dominant dimension
        dimension_scores = {
            ComplexityDimension.TOOL_USE: tool_steps * (1.5 if detected_tools else 0.5),
            ComplexityDimension.RETRIEVAL: retrieval_hops * (1.3 if entities else 0.5),
            ComplexityDimension.REASONING: reasoning_steps * (1.2 if reasoning_patterns else 0.5),
            ComplexityDimension.VERIFICATION: 1.0 if "verify" in task.lower() or "check" in task.lower() else 0.5,
        }
        dominant = max(dimension_scores, key=dimension_scores.get)
        
        # Step 6: Calculate total and confidence
        total_steps = max(tool_steps, retrieval_hops, reasoning_steps)
        confidence = ml_confidence if ml_estimate else 0.6
        
        # Step 7: Adjust from history (COMPASS advantage: continuous learning)
        if self.learning_enabled and domain:
            adjustment = self.step_adjustments.get(domain, 1.0)
            total_steps = int(total_steps * adjustment)
            if adjustment != 1.0:
                self.stats["history_adjustments"] += 1
        
        # Build estimate
        estimate = ComplexityEstimate(
            retrieval_hops=retrieval_hops,
            tool_steps=tool_steps,
            reasoning_steps=reasoning_steps,
            api_calls=max(1, tool_steps - 1),  # API calls often parallel tool steps
            memory_lookups=max(1, retrieval_hops),
            verification_steps=1 if "verify" in task.lower() else 0,
            total_estimated_steps=total_steps,
            confidence=confidence,
            dominant_dimension=dominant,
            reasoning_path=[f"Detected {len(entities)} entities", f"Found {len(reasoning_patterns)} reasoning patterns"],
            detected_entities=entities,
            detected_tools=detected_tools,
            detected_patterns=reasoning_patterns,
        )
        
        # Cache result
        if self.cache_enabled:
            self._cache[cache_key] = estimate
        
        return estimate
    
    def learn_from_execution(
        self,
        task: str,
        actual_steps: int,
        success: bool,
        domain: Optional[str] = None,
    ):
        """
        Learn from actual task execution to improve future estimates.
        
        COMPASS advantage: Continuous learning from successful retrievals.
        """
        if not self.learning_enabled:
            return
        
        # Record history
        self.task_history.append({
            "task": task,
            "actual_steps": actual_steps,
            "success": success,
            "domain": domain,
        })
        
        # Update domain adjustments
        if domain and success:
            # Get previous estimate
            cache_key = self._get_cache_key(task)
            if cache_key in self._cache:
                estimated = self._cache[cache_key].total_estimated_steps
                if estimated > 0:
                    # Exponential moving average of adjustment factor
                    adjustment = actual_steps / estimated
                    current = self.step_adjustments[domain]
                    self.step_adjustments[domain] = 0.7 * current + 0.3 * adjustment
        
        # Track successful patterns
        if success:
            patterns = ToolPatternDetector.detect_tools(task)
            for pattern in patterns:
                self.successful_patterns[pattern] += 1
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()


# =============================================================================
# SMART ROLLOUT STRATEGY (FROM COMPASS)
# =============================================================================

class SmartRolloutStrategy:
    """
    Smart rollout allocation strategy.
    
    Adapted from COMPASS's smart_rollout_strategy.py for generalized PRECEPT use.
    
    Key insight: Don't waste rollouts on simple tasks or already-good solutions.
    """
    
    def __init__(
        self,
        diversity_threshold: float = 0.7,
        confidence_threshold: float = 0.9,
        min_rollouts: int = 1,
        max_rollouts: int = 15,
        diversity_rollouts: int = 5,
        consistency_rollouts: int = 3,
    ):
        self.diversity_threshold = diversity_threshold
        self.confidence_threshold = confidence_threshold
        self.min_rollouts = min_rollouts
        self.max_rollouts = max_rollouts
        self.diversity_rollouts = diversity_rollouts
        self.consistency_rollouts = consistency_rollouts
        
        # Caching
        self._decision_cache: Dict[str, RolloutDecision] = {}
        
        # Statistics
        self.stats = {
            "decisions_made": 0,
            "rollouts_saved": 0,
            "early_stops": 0,
        }
    
    def decide(
        self,
        task_complexity: ComplexityEstimate,
        current_score: float,
        diversity_score: Optional[float] = None,
        previous_attempts: int = 0,
    ) -> RolloutDecision:
        """
        Decide rollout allocation based on complexity and current performance.
        
        From COMPASS smart rollouts:
        - If score is perfect (≥0.98), skip additional rollouts
        - If score is high (≥0.9) but diversity low, focus on diversity
        - If score is high and diversity OK, verify consistency
        - Otherwise, use complexity-based allocation
        """
        self.stats["decisions_made"] += 1
        
        # Rule 1: Perfect score - skip (COMPASS early stopping)
        if current_score >= 0.98:
            self.stats["early_stops"] += 1
            self.stats["rollouts_saved"] += self.max_rollouts
            return RolloutDecision(
                use_rollouts=False,
                num_rollouts=0,
                focus="skip",
                reason=f"Near-perfect score ({current_score:.3f}), no rollouts needed",
                early_stop_threshold=0.98,
            )
        
        # Rule 2: High score but low diversity - focus on diversity
        if current_score >= self.confidence_threshold:
            if diversity_score is not None and diversity_score < self.diversity_threshold:
                return RolloutDecision(
                    use_rollouts=True,
                    num_rollouts=self.diversity_rollouts,
                    focus="diversity",
                    reason=f"High score ({current_score:.3f}) but low diversity ({diversity_score:.3f})",
                )
            else:
                # High score, good diversity - just verify consistency
                return RolloutDecision(
                    use_rollouts=True,
                    num_rollouts=self.consistency_rollouts,
                    focus="consistency",
                    reason=f"Verifying consistency of high score ({current_score:.3f})",
                )
        
        # Rule 3: Complexity-based allocation
        complexity = task_complexity.total_estimated_steps
        confidence = task_complexity.confidence
        
        if complexity <= 2 and confidence > 0.8:
            # Simple task with high confidence - minimal rollouts
            num_rollouts = self.min_rollouts
            focus = "minimal"
        elif complexity <= 4:
            # Medium complexity - moderate rollouts
            num_rollouts = min(complexity + 2, self.max_rollouts // 2)
            focus = "exploration"
        else:
            # High complexity - more rollouts
            num_rollouts = min(complexity * 2, self.max_rollouts)
            focus = "thorough"
        
        # Adjust for previous attempts
        if previous_attempts > 0:
            num_rollouts = min(num_rollouts + previous_attempts, self.max_rollouts)
            focus = "recovery"
        
        return RolloutDecision(
            use_rollouts=True,
            num_rollouts=num_rollouts,
            focus=focus,
            reason=f"Complexity {complexity}, confidence {confidence:.2f}, focus: {focus}",
        )


# =============================================================================
# MULTI-STRATEGY COORDINATOR
# =============================================================================

class MultiStrategyCoordinator:
    """
    Coordinate multiple strategies for task execution.
    
    Adapted from COMPASS's multi-strategy retrieval for general PRECEPT use:
    - Tool strategy: Which tools to use in what order
    - Retrieval strategy: Vector, keyword, graph, or hybrid
    - Reasoning strategy: Chain-of-thought, tree-of-thought, etc.
    """
    
    class Strategy(Enum):
        DIRECT = "direct"           # Single-step execution
        SEQUENTIAL = "sequential"   # Step-by-step chain
        PARALLEL = "parallel"       # Multiple paths simultaneously
        ADAPTIVE = "adaptive"       # Adjust based on feedback
        CACHED = "cached"           # Use cached results
    
    def __init__(self):
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "avg_steps": 0}
        )
        
        # Strategy selection cache
        self._strategy_cache: Dict[str, str] = {}
    
    def select_strategy(
        self,
        complexity: ComplexityEstimate,
        domain: Optional[str] = None,
        previous_strategy: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Select best strategy based on complexity and history.
        
        Returns:
            Tuple of (strategy_name, reasoning)
        """
        dominant = complexity.dominant_dimension
        total_steps = complexity.total_estimated_steps
        
        # Check if we have cached success for this domain
        if domain and domain in self._strategy_cache:
            cached = self._strategy_cache[domain]
            return cached, f"Using cached successful strategy for {domain}"
        
        # Select based on dominant dimension
        if dominant == ComplexityDimension.TOOL_USE:
            if total_steps <= 2:
                return "direct", "Simple tool chain - direct execution"
            elif total_steps <= 4:
                return "sequential", "Medium tool chain - sequential with verification"
            else:
                return "adaptive", "Complex tool chain - adaptive with feedback"
        
        elif dominant == ComplexityDimension.RETRIEVAL:
            if total_steps <= 2:
                return "cached", "Simple retrieval - use cache if available"
            else:
                return "parallel", "Multi-hop retrieval - parallel exploration"
        
        elif dominant == ComplexityDimension.REASONING:
            if total_steps <= 3:
                return "sequential", "Linear reasoning - chain-of-thought"
            else:
                return "adaptive", "Complex reasoning - tree-of-thought"
        
        # Default
        return "sequential", "Default sequential strategy"
    
    def record_outcome(
        self,
        strategy: str,
        domain: str,
        success: bool,
        steps: int,
    ):
        """Record strategy outcome for learning."""
        perf = self.strategy_performance[f"{strategy}_{domain}"]
        
        if success:
            perf["successes"] += 1
            # Cache successful strategy
            self._strategy_cache[domain] = strategy
        else:
            perf["failures"] += 1
        
        # Update average steps
        total = perf["successes"] + perf["failures"]
        perf["avg_steps"] = (perf["avg_steps"] * (total - 1) + steps) / total


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instances
_analyzer: Optional[PRECEPTComplexityAnalyzer] = None
_rollout_strategy: Optional[SmartRolloutStrategy] = None
_strategy_coordinator: Optional[MultiStrategyCoordinator] = None


def get_complexity_analyzer() -> PRECEPTComplexityAnalyzer:
    """Get or create global complexity analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = PRECEPTComplexityAnalyzer()
    return _analyzer


def get_rollout_strategy() -> SmartRolloutStrategy:
    """Get or create global rollout strategy."""
    global _rollout_strategy
    if _rollout_strategy is None:
        _rollout_strategy = SmartRolloutStrategy()
    return _rollout_strategy


def get_strategy_coordinator() -> MultiStrategyCoordinator:
    """Get or create global strategy coordinator."""
    global _strategy_coordinator
    if _strategy_coordinator is None:
        _strategy_coordinator = MultiStrategyCoordinator()
    return _strategy_coordinator


def analyze_task_complexity(
    task: str,
    goal: Optional[str] = None,
    available_tools: Optional[List[str]] = None,
    domain: Optional[str] = None,
) -> ComplexityEstimate:
    """
    Convenience function to analyze task complexity.
    
    Uses COMPASS ML analyzer if available, falls back to pattern-based.
    """
    return get_complexity_analyzer().analyze(task, goal, available_tools, domain)


def decide_rollouts(
    task: str,
    current_score: float,
    diversity_score: Optional[float] = None,
    previous_attempts: int = 0,
    domain: Optional[str] = None,
) -> RolloutDecision:
    """
    Convenience function to decide rollout allocation.
    
    Implements COMPASS smart rollout strategy.
    """
    analyzer = get_complexity_analyzer()
    strategy = get_rollout_strategy()
    
    complexity = analyzer.analyze(task, domain=domain)
    return strategy.decide(complexity, current_score, diversity_score, previous_attempts)

