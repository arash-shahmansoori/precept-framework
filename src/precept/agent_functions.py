"""
PRECEPT Agent Pure Functions Module.

Contains pure functions for PRECEPT agent operations, enabling:
- Dependency injection for testability
- Separation of concerns
- Easier unit testing
- Functional composition

Usage:
    from precept.agent_functions import (
        parse_llm_response,
        build_reasoning_prompt,
        run_task_with_dependencies,
    )

    # Parse LLM response
    result = parse_llm_response(response_text)

    # Build prompt for LLM reasoning
    prompt = build_reasoning_prompt(task, parsed_task, memories, rules)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypeVar

from .config import AgentConfig, PromptTemplates  # noqa: I001
from .constraints import RefineInterceptor

# =============================================================================
# LOGGING SETUP
# =============================================================================

_logger = None


def _get_logger():
    """Get or create the module logger (lazy initialization)."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger("precept.agent_functions")
    return _logger


# =============================================================================
# TYPE DEFINITIONS AND PROTOCOLS
# =============================================================================

T = TypeVar("T")


class MCPClientProtocol(Protocol):
    """Protocol defining the interface for MCP clients."""

    async def retrieve_memories(self, query: str, top_k: int = 3) -> str: ...

    async def get_procedure(self, task_type: str) -> str: ...

    async def get_learned_rules(self) -> str: ...

    async def get_rule_hybrid(
        self,
        condition_key: str,
        task_description: str = "",
        similarity_threshold: float = 0.5,
        top_k: int = 3,
    ) -> str: ...

    async def record_error(
        self, error_code: str, context: str, solution: str = ""
    ) -> str: ...

    async def record_solution(
        self, error_code: str, solution: str, context: str
    ) -> str: ...

    async def store_experience(
        self,
        task: str,
        outcome: str,
        strategy: str,
        lessons: str,
        domain: str,
        error_code: str = "",
        solution: str = "",
        failed_options: str = "",
        task_type: str = "",
        condition_key: str = "",
    ) -> str: ...

    async def trigger_consolidation(self) -> str: ...

    async def trigger_compass_evolution(self, failure_context: str) -> str: ...

    async def get_evolved_prompt(self, include_rules: bool = True) -> str: ...

    async def update_memory_usefulness(
        self, feedback: float, task_succeeded: bool
    ) -> str: ...

    def analyze_complexity(self, task: str, action: str) -> str: ...


class DomainStrategyProtocol(Protocol):
    """Protocol defining the interface for domain strategies."""

    domain_name: str

    def parse_task(self, task: str) -> Any: ...

    def get_options_for_task(self, parsed_task: Any) -> List[str]: ...

    async def execute_action(self, mcp_client: Any, parsed_task: Any) -> Any: ...

    def get_available_actions(self) -> List[str]: ...


class LLMClientProtocol(Protocol):
    """Protocol defining the interface for LLM clients."""

    async def create(self, messages: List[Any], extra_create_args: Dict) -> Any: ...


# =============================================================================
# DATA CLASSES FOR FUNCTION RESULTS
# =============================================================================


@dataclass
class LLMSuggestion:
    """Result of parsing an LLM response."""

    suggested_solution: Optional[str]
    reasoning: str
    confidence: str


@dataclass
class TaskResult:
    """Result of running a task."""

    success: bool
    steps: int
    overhead: int
    duration: float
    response: str
    strategy: str
    complexity: str
    domain: str
    error_code: Optional[str] = None


@dataclass
class ContextFetchResult:
    """Result of fetching context from MCP."""

    memories: str
    procedure: str
    rules: str
    # ═══════════════════════════════════════════════════════════════════════════
    # HYBRID MATCH: Direct solution from 3-tier lookup
    # When set, this should be applied DIRECTLY without LLM reasoning.
    # This is PRECEPT's key advantage - deterministic rule application.
    # ═══════════════════════════════════════════════════════════════════════════
    exact_match_solution: Optional[str] = None  # e.g., "antwerp", "hamburg"
    exact_match_key: Optional[str] = None  # The condition key that was matched
    match_tier: Optional[int] = None  # 1=exact, 2=vector, 3=jaccard
    # ═══════════════════════════════════════════════════════════════════════════
    # PARTIAL PROGRESS: Resume from failed training
    # Options that already failed - skip them during testing to save steps
    # ═══════════════════════════════════════════════════════════════════════════
    failed_options: Optional[List[str]] = None  # e.g., ["shanghai", "ningbo"]


# =============================================================================
# PURE FUNCTIONS FOR LLM RESPONSE PARSING
# =============================================================================


def parse_llm_response(response_text: str) -> Optional[LLMSuggestion]:
    """
    Parse LLM response with robust error handling.

    This is a pure function that handles variations in LLM output format.

    Args:
        response_text: Raw text response from LLM

    Returns:
        LLMSuggestion if parsed successfully, None otherwise
    """
    if not response_text:
        return None

    # Normalize response
    response_upper = response_text.upper()

    # Check for explicit EXPLORE or EXHAUSTED
    if "EXPLORE" in response_upper:
        return None
    if "EXHAUSTED" in response_upper:
        return LLMSuggestion(
            suggested_solution="EXHAUSTED",
            reasoning="All options exhausted",
            confidence="low",
        )

    # Try to extract structured response
    solution = None
    reasoning = ""
    confidence = "medium"

    # Match SOLUTION line (flexible matching)
    solution_match = re.search(r"SOLUTION[:\s]+([^\n]+)", response_text, re.IGNORECASE)
    if solution_match:
        solution = solution_match.group(1).strip()
        # Clean up common artifacts
        solution = solution.strip("\"'`")
        if solution.upper() == "EXPLORE":
            return None

    # Match REASONING line
    reasoning_match = re.search(
        r"REASONING[:\s]+([^\n]+)", response_text, re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Match CONFIDENCE line
    confidence_match = re.search(r"CONFIDENCE[:\s]+(\w+)", response_text, re.IGNORECASE)
    if confidence_match:
        confidence = confidence_match.group(1).strip().lower()

    if solution:
        return LLMSuggestion(
            suggested_solution=solution,
            reasoning=reasoning,
            confidence=confidence,
        )

    return None


def parse_reflexion_response(
    response_text: str, valid_options: List[str]
) -> Dict[str, Optional[str]]:
    """
    Parse Reflexion-style LLM response.

    Args:
        response_text: Raw text response from LLM
        valid_options: List of valid options to match against

    Returns:
        Dict with solution, reflection, lesson, reasoning, confidence
    """
    result: Dict[str, Optional[str]] = {
        "solution": None,
        "reflection": None,
        "lesson": None,
        "reasoning": None,
        "confidence": None,
    }

    # Extract REFLECTION
    reflection_match = re.search(
        r"REFLECTION:\s*(.+?)(?=LESSON:|SOLUTION:|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if reflection_match:
        result["reflection"] = reflection_match.group(1).strip()

    # Extract LESSON
    lesson_match = re.search(
        r"LESSON:\s*(.+?)(?=SOLUTION:|$)", response_text, re.IGNORECASE | re.DOTALL
    )
    if lesson_match:
        result["lesson"] = lesson_match.group(1).strip()

    # Extract SOLUTION
    solution_match = re.search(r"SOLUTION:\s*(\S+)", response_text, re.IGNORECASE)
    if solution_match:
        suggested = solution_match.group(1).strip().lower()
        for opt in valid_options:
            if opt.lower() == suggested:
                result["solution"] = opt
                break
        if not result["solution"]:
            result["solution"] = suggested

    # Extract REASONING
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=CONFIDENCE:|$)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract CONFIDENCE
    confidence_match = re.search(r"CONFIDENCE:\s*(\w+)", response_text, re.IGNORECASE)
    if confidence_match:
        result["confidence"] = confidence_match.group(1).strip().lower()

    # Fallback: look for any option in response
    if not result["solution"]:
        response_lower = response_text.lower()
        for opt in valid_options:
            if opt.lower() in response_lower:
                result["solution"] = opt
                break

    return result


# =============================================================================
# PURE FUNCTIONS FOR PROMPT BUILDING
# =============================================================================


def filter_rules_by_relevance(
    learned_rules: str,
    parsed_task: Any,
    task: str,
    max_chars: int = 0,
) -> str:
    """
    Smart filtering of learned rules by relevance to the current task.

    This ensures the LLM sees the MOST RELEVANT rules first, rather than
    arbitrary truncation that might hide important rules.

    Filtering strategy:
    1. Extract entity from task (e.g., "AA-999" from "Book AA-999 for conference")
    2. Filter rules that mention the entity or related error codes
    3. If no relevant rules found, return all rules (up to max_chars if set)
    4. If max_chars=0, return all relevant rules without truncation

    Args:
        learned_rules: All learned rules as a string (newline-separated)
        parsed_task: The parsed task object with entity info
        task: The raw task string
        max_chars: Maximum characters (0 = unlimited)

    Returns:
        Filtered rules string, prioritizing relevance
    """
    if not learned_rules or learned_rules == "No rules learned yet.":
        return "No learned rules yet."

    # Split rules into individual lines
    lines = learned_rules.strip().split("\n")
    if not lines:
        return "No learned rules yet."

    # Extract search terms from task
    entity = getattr(parsed_task, "entity", "") or ""
    action = getattr(parsed_task, "action", "") or ""
    source = getattr(parsed_task, "source", "") or ""
    target = getattr(parsed_task, "target", "") or ""

    # Build search terms (case-insensitive)
    search_terms = set()
    for term in [entity, action, source, target]:
        if term and len(term) > 2:  # Skip very short terms
            search_terms.add(term.lower())

    # ═══════════════════════════════════════════════════════════════════════════
    # FIX: Include condition_key and conditions from parsed_task.parameters
    # This ensures multi-condition rules are matched correctly
    # ═══════════════════════════════════════════════════════════════════════════
    params = getattr(parsed_task, "parameters", {}) or {}

    # Add condition_key (e.g., "C-HIGH+R-482+T-PEAK")
    condition_key = params.get("condition_key")
    if condition_key:
        search_terms.add(condition_key.lower())
        # Also add individual conditions from the key
        for cond in condition_key.split("+"):
            if cond.strip():
                search_terms.add(cond.strip().lower())

    # Add individual conditions list (e.g., ["C-HIGH", "R-482", "T-PEAK"])
    conditions = params.get("conditions", [])
    for cond in conditions:
        if cond and len(cond) > 1:
            search_terms.add(cond.lower())

    # Also extract potential entities from the raw task (e.g., flight numbers)
    import re

    # ═══════════════════════════════════════════════════════════════════════════
    # FIX: Comprehensive pattern for ALL domain condition codes
    # Matches: R-482, BK-401, C-HIGH, SVC-CRIT, NET-TIMEOUT, K8S-101, etc.
    # Pattern: 1-4 alphanumeric chars + dash + 2-8 alphanumeric chars
    # ═══════════════════════════════════════════════════════════════════════════
    entity_pattern = r"[A-Z0-9]{1,4}-[A-Z0-9]{2,8}"
    found_entities = re.findall(entity_pattern, task.upper())
    for e in found_entities:
        search_terms.add(e.lower())

    # Separate header from rules
    header_lines = []
    rule_lines = []
    for line in lines:
        if line.startswith("===") or line.startswith("---"):
            header_lines.append(line)
        elif line.strip():
            rule_lines.append(line)

    # Filter rules by relevance
    relevant_rules = []
    other_rules = []

    for rule in rule_lines:
        rule_lower = rule.lower()
        is_relevant = any(term in rule_lower for term in search_terms)
        if is_relevant:
            relevant_rules.append(rule)
        else:
            other_rules.append(rule)

    # Build result: relevant rules first, then others
    result_lines = header_lines.copy()

    if relevant_rules:
        result_lines.append("📌 RELEVANT TO YOUR TASK:")
        result_lines.extend(relevant_rules)
        if other_rules:
            result_lines.append("")
            result_lines.append("📚 OTHER LEARNED RULES:")
            result_lines.extend(other_rules)
    else:
        # No specifically relevant rules - include all
        result_lines.extend(rule_lines)

    result = "\n".join(result_lines)

    # Apply max_chars limit only if set (0 = unlimited)
    if max_chars > 0 and len(result) > max_chars:
        # Truncate but try to end at a complete line
        truncated = result[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars * 0.8:  # Keep at least 80% of allowed chars
            truncated = truncated[:last_newline]
        return truncated + "\n... (more rules available)"

    return result


def build_reasoning_prompt(
    task: str,
    parsed_task: Any,
    memories: str,
    learned_rules: str,
    forbidden_section: str = "",
    error_feedback: str = "",
    available_options: Optional[List[str]] = None,
    prompts: Optional[PromptTemplates] = None,
    max_rules_chars: int = 0,
    max_memories_chars: int = 2000,
    enable_smart_rule_filtering: bool = True,
    procedure: str = "",
) -> str:
    """
    Build the reasoning prompt for LLM.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        memories: Retrieved memories
        learned_rules: Learned rules as text
        forbidden_section: FORBIDDEN options injection
        error_feedback: Error feedback from previous attempt
        available_options: List of valid options the LLM can suggest
        prompts: Optional prompt templates
        max_rules_chars: Max chars for rules (0 = unlimited)
        max_memories_chars: Max chars for memories (0 = unlimited)
        enable_smart_rule_filtering: If True, filter rules by relevance first
        procedure: Procedural memory (step-by-step how-to strategy)

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    # Determine task type
    task_type = (
        "CUSTOMS"
        if getattr(parsed_task, "parameters", {}).get("is_customs")
        else "BOOKING"
    )

    # Format parameters (exclude internal ones)
    params = getattr(parsed_task, "parameters", {})
    params_to_show = {
        k: v
        for k, v in params.items()
        if k not in ["procedure_hint", "preferred_solution"]
    }

    # Build error feedback section
    error_feedback_section = ""
    if error_feedback:
        error_feedback_section = f"""
ERROR FEEDBACK (from previous failed attempt):
{error_feedback}
DO NOT suggest any solution that appears in "Tried" above!
"""

    # NOTE: We intentionally do NOT include available_options in the prompt.
    # This would defeat the black swan scenario where the agent must learn
    # through trial and error, not by being given a list of options.
    # PRECEPT learns solutions through handle_error exploration, not upfront hints.

    # =========================================================================
    # SMART RULE FILTERING: Prioritize relevant rules, then include others
    # =========================================================================
    if enable_smart_rule_filtering and learned_rules:
        filtered_rules = filter_rules_by_relevance(
            learned_rules=learned_rules,
            parsed_task=parsed_task,
            task=task,
            max_chars=max_rules_chars,
        )
    else:
        # Simple truncation if smart filtering disabled
        if max_rules_chars > 0 and learned_rules:
            filtered_rules = learned_rules[:max_rules_chars]
        else:
            filtered_rules = learned_rules or "No learned rules yet."

    # Memory truncation (simple, as memories are already relevance-filtered by retrieval)
    if max_memories_chars > 0 and memories:
        filtered_memories = memories[:max_memories_chars]
    else:
        filtered_memories = memories or "No relevant memories found."

    # ═══════════════════════════════════════════════════════════════════════════
    # PROCEDURAL MEMORY: Format step-by-step strategies for LLM
    # ═══════════════════════════════════════════════════════════════════════════
    procedure_section = ""
    if procedure and "No procedure found" not in procedure:
        procedure_section = f"""
═══════════════════════════════════════════════════════════════════════════════
📋 PROCEDURAL MEMORY (Step-by-Step Strategy)
═══════════════════════════════════════════════════════════════════════════════
{procedure}
"""

    return prompts.reasoning_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=task_type,
        parameters=str(params_to_show),
        memories=filtered_memories,
        learned_rules=filtered_rules,
        forbidden_section=forbidden_section or "",
        error_feedback_section=error_feedback_section,
        procedure_section=procedure_section,
    )


def build_baseline_prompt(
    task: str,
    parsed_task: Any,
    memories: str,
    options: List[str],
    error_context: str = "",
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """
    Build the prompt for baseline LLM reasoning.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        memories: Retrieved memories
        options: Available options
        error_context: Error context from previous attempt
        prompts: Optional prompt templates

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    return prompts.baseline_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=getattr(parsed_task, "task_type", "general"),
        options=", ".join(options),
        memories=memories or "No relevant memories found.",
        error_context=error_context,
    )


def build_reflexion_prompt(
    task: str,
    parsed_task: Any,
    memories: str,
    options: List[str],
    reflection_section: str = "",
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """
    Build the prompt for Reflexion-style reasoning.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        memories: Retrieved memories
        options: Available options
        reflection_section: Previous attempts and reflections
        prompts: Optional prompt templates

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    return prompts.reflexion_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=getattr(parsed_task, "task_type", "general"),
        options=", ".join(options),
        memories=memories or "No relevant memories found.",
        reflection_section=reflection_section,
    )


def build_full_reflexion_prompt(
    task: str,
    parsed_task: Any,
    options: List[str],
    task_type: str,
    accumulated_reflections: str,
    current_episode_context: str = "",
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """
    Build the prompt for Full Reflexion with cross-episode memory.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        options: Available options
        task_type: The task type identifier
        accumulated_reflections: Formatted reflections from previous episodes
        current_episode_context: Current episode attempts
        prompts: Optional prompt templates

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    return prompts.full_reflexion_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=task_type,
        options=", ".join(options),
        accumulated_reflections=accumulated_reflections,
        current_episode_context=current_episode_context,
    )


# =============================================================================
# ASYNC HELPER FUNCTIONS
# =============================================================================


async def parallel_fetch(
    *coros,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Any]:
    """
    Run multiple coroutines in parallel, optionally limited by semaphore.

    This enables efficient parallel fetching of memories, rules, procedures
    while respecting the max_internal_workers limit.

    Args:
        *coros: Coroutines to run in parallel
        semaphore: Optional semaphore for concurrency control

    Returns:
        List of results (including any exceptions)
    """
    if semaphore is None:
        return await asyncio.gather(*coros, return_exceptions=True)

    async def wrap(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*[wrap(c) for c in coros], return_exceptions=True)


async def fetch_context(
    mcp_client: MCPClientProtocol,
    query: str,
    task_type: str,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> ContextFetchResult:
    """
    Fetch all context needed for task execution.

    Args:
        mcp_client: The MCP client
        query: Query for memory retrieval
        task_type: Task type for procedure lookup
        semaphore: Optional semaphore for concurrency control

    Returns:
        ContextFetchResult with memories, procedure, and rules
    """
    results = await parallel_fetch(
        mcp_client.retrieve_memories(query, top_k=3),
        mcp_client.get_procedure(task_type),
        mcp_client.get_learned_rules(),
        semaphore=semaphore,
    )

    return ContextFetchResult(
        memories=results[0] if not isinstance(results[0], Exception) else "",
        procedure=results[1] if not isinstance(results[1], Exception) else "",
        rules=results[2] if not isinstance(results[2], Exception) else "",
    )


async def fetch_context_with_hybrid(
    mcp_client: MCPClientProtocol,
    query: str,
    task_type: str,
    task_description: str = "",
    condition_key: Optional[str] = None,
    similarity_threshold: float = 0.5,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> ContextFetchResult:
    """
    Enhanced context fetch with 3-TIER HYBRID RULE RETRIEVAL.

    This is PRECEPT's secret weapon for both matched and random test scenarios:

    TIER 1: O(1) hash lookup (instant, exact match)
    TIER 2: Vector similarity (semantic, like ExpeL)
    TIER 3: Jaccard similarity (structural fallback)

    Args:
        mcp_client: The MCP client
        query: Query for memory retrieval
        task_type: Task type for procedure lookup
        task_description: Full task text for semantic matching (Tier 2)
        condition_key: Optional multi-condition key for hybrid matching
        similarity_threshold: Minimum similarity for partial matches (0.0-1.0)
        semaphore: Optional semaphore for concurrency control

    Returns:
        ContextFetchResult with memories, procedure, and enhanced rules
    """
    import json

    # Initialize match variables (will be set if any Tier finds a match)
    exact_match_solution = None
    exact_match_key = None
    match_tier = None  # 1=exact, 2=vector, 3=jaccard
    failed_options = None  # Options that already failed during training

    # Prepare coroutines for parallel fetch
    coros = [
        mcp_client.retrieve_memories(query, top_k=3),
        mcp_client.get_procedure(task_type),
        mcp_client.get_learned_rules(),
    ]

    # Add hybrid lookup if condition_key is available
    if condition_key:
        coros.append(
            mcp_client.get_rule_hybrid(
                condition_key=condition_key,
                task_description=task_description,  # For Tier 2 vector similarity
                similarity_threshold=similarity_threshold,
                top_k=3,
            )
        )

    results = await parallel_fetch(*coros, semaphore=semaphore)

    # Extract base results
    memories = results[0] if not isinstance(results[0], Exception) else ""
    procedure = results[1] if not isinstance(results[1], Exception) else ""
    rules = results[2] if not isinstance(results[2], Exception) else ""

    # Process hybrid results if available (3-tier HIERARCHICAL strategy)
    # CRITICAL: Only ONE tier should be applied per lookup to avoid context pollution
    if condition_key and len(results) > 3 and not isinstance(results[3], Exception):
        try:
            hybrid_result = json.loads(results[3])
            hybrid_rules = []

            # ═══════════════════════════════════════════════════════════════════
            # PARTIAL PROGRESS: Extract failed options to skip during testing
            # This allows resuming from where training left off
            # ═══════════════════════════════════════════════════════════════════
            failed_options = hybrid_result.get("failed_options", [])

            # ═══════════════════════════════════════════════════════════════════
            # HIERARCHICAL TIERS: Mutually exclusive - first success wins
            # ═══════════════════════════════════════════════════════════════════

            # TIER 1: Exact match (O(1) hash lookup) - HIGHEST PRIORITY
            if hybrid_result.get("exact_match"):
                exact = hybrid_result["exact_match"]
                hybrid_rules.append(
                    f"\n=== EXACT MATCH (O(1) Lookup - Tier 1) ===\n"
                    f"• APPLY THIS RULE: {exact['solution']} [confidence: 100%]\n"
                    f"  Key: {exact['key'][:60]}..."
                )
                # Set for DIRECT APPLICATION - bypasses LLM reasoning
                exact_match_solution = exact.get("solution")
                exact_match_key = exact.get("key")
                match_tier = 1
                # STOP HERE - don't check lower tiers

            # TIER 2: Vector similarity (semantic) - ONLY if Tier 1 failed
            elif hybrid_result.get("vector_matches"):
                hybrid_rules.append(
                    "\n=== SEMANTIC MATCH (Vector Similarity - Tier 2) ==="
                )
                best = hybrid_result["vector_matches"][0]
                hybrid_rules.append(
                    f"• SUGGESTED: {best['solution']} [semantic match]\n"
                    f"  Key: {best['key'][:60]}..."
                )
                # Set for DIRECT APPLICATION - bypasses unreliable LLM interpretation
                exact_match_solution = best.get("solution")
                exact_match_key = best.get("key")
                match_tier = 2
                # STOP HERE - don't check Tier 3

            # TIER 3: Jaccard similarity (structural) - ONLY if Tier 1 & 2 failed
            elif hybrid_result.get("jaccard_matches"):
                hybrid_rules.append(
                    "\n=== PARTIAL MATCH (Jaccard Similarity - Tier 3) ==="
                )
                best = hybrid_result["jaccard_matches"][0]
                overlap = best.get("overlap_count", 0)
                sim = best.get("similarity", 0)
                hybrid_rules.append(
                    f"• SUGGESTED: {best['solution']} "
                    f"[{sim:.0%} match, {overlap} conditions overlap]\n"
                    f"  Key: {best['key'][:60]}..."
                )
                # Include which conditions match for transparency
                common = best.get("common_conditions", [])
                if common:
                    hybrid_rules.append(
                        f"  Matching conditions: {', '.join(common[:5])}"
                    )
                # Set for DIRECT APPLICATION - bypasses unreliable LLM interpretation
                exact_match_solution = best.get("solution")
                exact_match_key = best.get("key")
                match_tier = 3

            # Merge ONLY the successful tier's rules with standard rules
            if hybrid_rules:
                rules = "\n".join(hybrid_rules) + "\n\n" + rules

        except (json.JSONDecodeError, TypeError):
            pass  # Fall back to standard rules

    return ContextFetchResult(
        memories=memories,
        procedure=procedure,
        rules=rules,
        exact_match_solution=exact_match_solution if condition_key else None,
        exact_match_key=exact_match_key if condition_key else None,
        match_tier=match_tier if condition_key else None,
        failed_options=failed_options if condition_key else None,
    )


# =============================================================================
# COMPOSITIONAL GENERALIZATION (Atomic Constraint Stacking)
# =============================================================================


@dataclass
class CompositionalContext:
    """Result of compositional precept retrieval."""

    # Original context
    memories: str
    procedure: str
    rules: str

    # Compositional data
    constraint_stack: List[str]  # Stacked constraints for LLM injection
    precepts_found: List[Dict[str, Any]]  # Atomic precepts retrieved
    precepts_missing: List[str]  # Conditions without precepts
    synthesis_mode: str  # "full_compositional", "partial_compositional", "exploration_needed"
    coverage: float  # Fraction of conditions with precepts

    # Conflict resolution (Constitution of Constraints)
    conflicts: Optional[Dict[str, Any]] = None  # Detected conflicts
    resolution: Optional[Dict[str, Any]] = None  # Hierarchical resolution applied

    # Standard hybrid matching (fallback)
    exact_match_solution: Optional[str] = None
    exact_match_key: Optional[str] = None
    match_tier: Optional[int] = None
    failed_options: Optional[List[str]] = None


async def fetch_context_compositional(
    mcp_client: MCPClientProtocol,
    query: str,
    task_type: str,
    task_description: str = "",
    condition_key: Optional[str] = None,
    similarity_threshold: float = 0.5,
    min_precept_confidence: float = 0.3,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> CompositionalContext:
    """
    Compositional context fetch with ATOMIC CONSTRAINT STACKING.

    This is PRECEPT's key advantage for compositional generalization:
    1. Decompose composite condition into atomic components
    2. Retrieve atomic precepts for each component independently
    3. Stack constraints in LLM context for compositional synthesis

    This enables O(2^N) compositional adaptation from N learned precepts.

    Args:
        mcp_client: The MCP client
        query: Query for memory retrieval
        task_type: Task type for procedure lookup
        task_description: Full task text for semantic matching
        condition_key: Composite condition key for decomposition
        similarity_threshold: Minimum similarity for fallback matching
        min_precept_confidence: Minimum confidence for atomic precepts
        semaphore: Optional semaphore for concurrency control

    Returns:
        CompositionalContext with constraint stack and synthesis mode
    """
    import json

    # Initialize compositional fields
    constraint_stack = []
    precepts_found = []
    precepts_missing = []
    synthesis_mode = "exploration_needed"
    coverage = 0.0

    # Conflict resolution fields (Constitution of Constraints)
    conflicts = None
    resolution = None

    # Fallback fields (from hybrid matching)
    exact_match_solution = None
    exact_match_key = None
    match_tier = None
    failed_options = None

    # Prepare coroutines for parallel fetch
    coros = [
        mcp_client.retrieve_memories(query, top_k=3),
        mcp_client.get_procedure(task_type),
        mcp_client.get_learned_rules(),
    ]

    # Add compositional retrieval if condition_key is available
    if condition_key:
        coros.append(
            mcp_client.call_tool(
                "retrieve_atomic_precepts",
                {
                    "condition_key": condition_key,
                    "min_confidence": min_precept_confidence,
                }
            )
        )
        # Also add hybrid lookup as fallback
        coros.append(
            mcp_client.get_rule_hybrid(
                condition_key=condition_key,
                task_description=task_description,
                similarity_threshold=similarity_threshold,
                top_k=3,
            )
        )

    results = await parallel_fetch(*coros, semaphore=semaphore)

    # Extract base results
    memories = results[0] if not isinstance(results[0], Exception) else ""
    procedure = results[1] if not isinstance(results[1], Exception) else ""
    rules = results[2] if not isinstance(results[2], Exception) else ""

    # Process compositional results (atomic precepts)
    if condition_key and len(results) > 3 and not isinstance(results[3], Exception):
        try:
            compositional_result = json.loads(results[3])

            # Extract constraint stack for LLM injection
            constraint_stack = compositional_result.get("constraint_stack", [])
            precepts_found = compositional_result.get("precepts_found", [])
            precepts_missing = compositional_result.get("precepts_missing", [])
            synthesis_mode = compositional_result.get("synthesis_mode", "exploration_needed")
            coverage = compositional_result.get("coverage", 0.0)

            # Extract conflict resolution data (Constitution of Constraints)
            conflicts = compositional_result.get("conflicts")
            resolution = compositional_result.get("resolution")

        except (json.JSONDecodeError, TypeError):
            pass  # Fall back to hybrid matching

    # Process hybrid results as fallback
    if condition_key and len(results) > 4 and not isinstance(results[4], Exception):
        try:
            hybrid_result = json.loads(results[4])
            failed_options = hybrid_result.get("failed_options", [])

            # If compositional didn't find enough precepts, use hybrid matching
            if synthesis_mode == "exploration_needed":
                if hybrid_result.get("exact_match"):
                    exact = hybrid_result["exact_match"]
                    exact_match_solution = exact.get("solution")
                    exact_match_key = exact.get("key")
                    match_tier = 1
                elif hybrid_result.get("vector_matches"):
                    best = hybrid_result["vector_matches"][0]
                    exact_match_solution = best.get("solution")
                    exact_match_key = best.get("key")
                    match_tier = 2
                elif hybrid_result.get("jaccard_matches"):
                    best = hybrid_result["jaccard_matches"][0]
                    exact_match_solution = best.get("solution")
                    exact_match_key = best.get("key")
                    match_tier = 3

        except (json.JSONDecodeError, TypeError):
            pass

    return CompositionalContext(
        memories=memories,
        procedure=procedure,
        rules=rules,
        constraint_stack=constraint_stack,
        precepts_found=precepts_found,
        precepts_missing=precepts_missing,
        synthesis_mode=synthesis_mode,
        coverage=coverage,
        conflicts=conflicts,
        resolution=resolution,
        exact_match_solution=exact_match_solution,
        exact_match_key=exact_match_key,
        match_tier=match_tier,
        failed_options=failed_options,
    )


def build_constraint_stack_prompt(
    constraint_stack: List[str],
    resolution: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a constraint stacking prompt section for LLM synthesis.

    This is the "Refine Layer" that constructs a compound prompt
    with all applicable constraints stacked for the LLM.

    Handles:
    - Normal constraint stacking for compositional synthesis
    - Conflict resolution with overridden constraints
    - Synthesis opportunities when conflicts have alternative solutions

    Args:
        constraint_stack: List of constraint strings to stack
        resolution: Optional conflict resolution data

    Returns:
        Formatted prompt section for LLM injection
    """
    if not constraint_stack:
        return ""

    lines = [
        "\n═══════════════════════════════════════════════════════════════════════",
        "⚛️ COMPOSITIONAL CONSTRAINTS (Constitution of Constraints)",
        "═══════════════════════════════════════════════════════════════════════",
        "",
    ]

    # Add tier hierarchy explanation if there are multiple constraints
    if len(constraint_stack) > 1:
        lines.append("CONSTRAINT HIERARCHY (higher tier overrides lower):")
        lines.append("  PHYSICS (tier=3) > POLICY (tier=2) > INSTRUCTION (tier=1)")
        lines.append("")

    # Check if there are conflicts/resolutions
    has_conflicts = resolution and resolution.get("conflicts_found", 0) > 0
    has_synthesis = resolution and resolution.get("synthesis_opportunities")

    if has_conflicts:
        lines.append("⚠️ CONFLICT DETECTED: Some constraints conflict. Resolution applied:")
        lines.append("")

    # Add constraints (already include [OVERRIDDEN] markers from retrieval)
    active_count = 0
    for constraint in constraint_stack:
        if constraint.startswith("💡"):
            # This is synthesis opportunity section
            lines.append(constraint)
        elif constraint.startswith("[OVERRIDDEN"):
            lines.append(f"   ⛔ {constraint}")
        elif constraint.strip():
            active_count += 1
            lines.append(f"   {active_count}. {constraint}")
        else:
            lines.append("")

    lines.append("")

    # Add synthesis instruction based on conflict status
    if has_synthesis:
        lines.extend([
            "SYNTHESIS INSTRUCTION (Conflict Resolution Mode):",
            "1. FIRST: Try the synthesis opportunities above to satisfy BOTH constraints",
            "2. If synthesis not possible, apply ACTIVE constraints only (ignore OVERRIDDEN)",
            "3. Higher-tier constraints (PHYSICS, POLICY) MUST be satisfied",
            "4. Report 'HARD_CONSTRAINT_VIOLATION' if critical constraints cannot be met",
        ])
    elif has_conflicts:
        lines.extend([
            "SYNTHESIS INSTRUCTION (Hierarchical Resolution Applied):",
            "1. Apply ACTIVE constraints only (OVERRIDDEN constraints have been pruned)",
            "2. Higher-tier precepts have already overridden lower-tier conflicts",
            "3. Combine remaining solution hints logically",
        ])
    else:
        lines.extend([
            "SYNTHESIS INSTRUCTION:",
            "You must find a solution that satisfies ALL constraints above.",
            "Combine the solution hints logically. Constraints are ordered by priority.",
        ])

    lines.extend([
        "═══════════════════════════════════════════════════════════════════════",
        "",
    ])

    return "\n".join(lines)


async def extract_and_store_atomic_precepts(
    mcp_client: MCPClientProtocol,
    condition_key: str,
    solution: str,
    domain: str = "general",
    tier: int = None,
    semantic_meaning: str = None,
) -> None:
    """
    Extract atomic precepts from a successful composite solution.

    Called when PRECEPT learns a composite rule to enable future
    compositional generalization.

    Args:
        mcp_client: The MCP client
        condition_key: The composite condition key
        solution: The solution that worked
        domain: Domain this applies to
        tier: Priority tier for hierarchical constraint resolution (1=lowest, 3=highest)
              Used for compositional reasoning: higher tier wins when constraints conflict
        semantic_meaning: Human-readable description of what this condition means
    """
    try:
        tier_str = f" (tier={tier})" if tier else ""
        _get_logger().info(
            f"⚛️ STORING ATOMIC PRECEPT: {condition_key} → {solution}{tier_str}"
        )

        tool_args = {
            "condition_key": condition_key,
            "solution": solution,
            "domain": domain,
        }
        if tier is not None:
            tool_args["tier"] = tier
        if semantic_meaning:
            tool_args["semantic_meaning"] = semantic_meaning

        result = await mcp_client.call_tool(
            "extract_atomic_precepts_from_solution",
            tool_args
        )
        _get_logger().debug(f"   Result: {result}")
    except Exception as e:
        _get_logger().warning(f"   ⚠️ Failed to store atomic precept: {e}")


# =============================================================================
# TASK EXECUTION FUNCTIONS
# =============================================================================


async def record_error(
    mcp_client: MCPClientProtocol,
    error_code: str,
    context: str,
    solution: str = "",
) -> None:
    """
    Record an error for learning purposes.

    This is a pure side-effect function that records the error to the MCP server.
    If solution is provided, a rule is learned IMMEDIATELY (no count requirement).

    Args:
        mcp_client: The MCP client
        error_code: The error code to record
        context: Context for recording
        solution: (Optional) The working solution for this error
    """
    await mcp_client.record_error(error_code, context, solution)


def add_error_constraint(
    interceptor: RefineInterceptor,
    error_code: str,
    error_message: str,
    failed_solution: str,
) -> Any:
    """
    Add a constraint to the interceptor to prune failed solutions.

    This is a pure function that returns the created constraint.

    Args:
        interceptor: The RefineInterceptor
        error_code: The error code
        error_message: The error message
        failed_solution: The solution that failed

    Returns:
        The created constraint
    """
    return interceptor.add_constraint(
        solution=failed_solution,
        error_code=error_code,
        error_message=error_message,
    )


async def record_error_and_add_constraint(
    mcp_client: MCPClientProtocol,
    interceptor: RefineInterceptor,
    error_code: str,
    error_message: str,
    failed_solution: str,
    context: str,
    verbose: bool = False,
) -> None:
    """
    Record an error and add it as a constraint.

    DEPRECATED: Use record_error() and add_error_constraint() separately
    for better testability and single responsibility.

    Args:
        mcp_client: The MCP client
        interceptor: The RefineInterceptor
        error_code: The error code
        error_message: The error message
        failed_solution: The solution that failed
        context: Context for recording
        verbose: Whether to print debug info
    """
    await record_error(mcp_client, error_code, context)
    constraint = add_error_constraint(
        interceptor, error_code, error_message, failed_solution
    )

    if verbose:
        print(
            f"    🚫 CONSTRAINT: {failed_solution} → {constraint.constraint_type.value}"
        )


async def report_rule_failure(
    mcp_client: MCPClientProtocol,
    condition_key: str,
    failed_solution: str,
    error_message: str = "",
    verbose: bool = False,
) -> Optional[str]:
    """
    Report that a learned rule produced a failure.

    Call this when an action based on a learned rule FAILS. This triggers
    PRECEPT's unlearning mechanism which invalidates stale rules after
    repeated failures.

    This enables drift adaptation - rules learned during training that
    no longer work at test time (e.g., due to environment changes) will
    be automatically invalidated.

    Args:
        mcp_client: The MCP client
        condition_key: The condition key whose rule failed
        failed_solution: The solution that was tried and failed
        error_message: Optional error message from the failure
        verbose: Whether to print debug info

    Returns:
        Invalidation message if rule was deleted, None otherwise
    """
    import logging

    logger = logging.getLogger("precept.unlearning")

    try:
        result = await mcp_client.call_tool(
            "report_rule_failure",
            {
                "condition_key": condition_key,
                "failed_solution": failed_solution,
                "error_message": error_message,
            },
        )

        # Check if rule was invalidated
        if "INVALIDATED" in result:
            logger.warning(f"Rule invalidated: {condition_key} → {failed_solution}")
            if verbose:
                print(f"    🗑️ RULE INVALIDATED: {condition_key[:40]}...")
            return result
        else:
            logger.info(f"Rule failure recorded: {condition_key} → {failed_solution}")
            if verbose:
                print(f"    ⚠️ RULE FAILURE: {condition_key[:40]}... ({failed_solution})")
            return None

    except Exception as e:
        logger.debug(f"Failed to report rule failure (non-critical): {e}")
        return None


async def handle_rule_based_failure(
    mcp_client: MCPClientProtocol,
    interceptor: RefineInterceptor,
    condition_key: str,
    error_code: str,
    error_message: str,
    failed_solution: str,
    context: str,
    was_from_rule: bool = False,
    verbose: bool = False,
) -> Optional[str]:
    """
    Handle a failure, with special processing if the solution came from a rule.

    This is the unified failure handler that:
    1. Records the error for constraint learning
    2. Adds a constraint to prune the failed solution
    3. Reports rule failure if the solution came from a learned rule

    Args:
        mcp_client: The MCP client
        interceptor: The RefineInterceptor
        condition_key: The composite condition key (e.g., "A+B+C")
        error_code: The error code from the failure
        error_message: The error message
        failed_solution: The solution that failed
        context: Context for recording
        was_from_rule: Whether the failed solution came from a learned rule
        verbose: Whether to print debug info

    Returns:
        Invalidation message if a rule was deleted, None otherwise
    """
    # Standard error recording
    await record_error(mcp_client, error_code, context)
    constraint = add_error_constraint(
        interceptor, error_code, error_message, failed_solution
    )

    if verbose:
        print(
            f"    🚫 CONSTRAINT: {failed_solution} → {constraint.constraint_type.value}"
        )

    # If solution came from a rule, report the failure for unlearning
    invalidation_msg = None
    if was_from_rule and condition_key:
        invalidation_msg = await report_rule_failure(
            mcp_client=mcp_client,
            condition_key=condition_key,
            failed_solution=failed_solution,
            error_message=error_message,
            verbose=verbose,
        )

    return invalidation_msg


async def record_successful_solution(
    mcp_client: MCPClientProtocol,
    error_code: str,
    solution: str,
    context: str,
    verbose: bool = False,
    domain: str = "general",
    skip_atomic_in_learned_rules: bool = False,
) -> bool:
    """
    Record a successful solution after a pivot.

    This is CRITICAL for PRECEPT learning - it triggers save_rules()!

    Args:
        mcp_client: The MCP client
        error_code: The original error code (condition_key)
        solution: The working solution
        context: Context for recording
        verbose: Whether to print debug info
        domain: The domain name for domain mappings
        skip_atomic_in_learned_rules: If True, skip storing atomic conditions
            (no "+" in error_code) in learned_rules.json. Atomic precepts
            should be stored separately via extract_and_store_atomic_precepts.

    Returns:
        True if solution was recorded successfully, False otherwise
    """
    import logging

    logger = logging.getLogger("precept.learning")
    success = False

    # ═══════════════════════════════════════════════════════════════════════════
    # CLEAN SEPARATION: Atomic vs Composite Storage
    # ═══════════════════════════════════════════════════════════════════════════
    # - Atomic conditions (no "+") → stored in atomic_precepts.json ONLY
    # - Composite conditions (has "+") → stored in learned_rules.json
    # This prevents duplication and enables clean compositional generalization.
    # ═══════════════════════════════════════════════════════════════════════════
    is_atomic = "+" not in error_code
    if skip_atomic_in_learned_rules and is_atomic:
        if verbose:
            print(f"    ⚛️ ATOMIC SKIP: {error_code} → stored in atomic_precepts only")
        logger.debug(f"Skipping atomic condition in learned_rules: {error_code}")
        # Still store domain mapping for fallback retrieval
        try:
            await mcp_client.call_tool(
                "store_domain_mapping",
                {
                    "domain": domain,
                    "mapping_type": "error_solutions",
                    "key": error_code,
                    "value": solution,
                },
            )
        except Exception:
            pass
        return True  # Considered success (will be stored via extract_and_store_atomic_precepts)

    try:
        # Primary: Record solution (triggers save_rules) - COMPOSITE only
        # CRITICAL: Pass task_succeeded=True since this function is only called
        # for solutions from SUCCESSFUL task completions
        await mcp_client.record_solution(
            error_code=error_code,
            solution=solution,
            context=context,
            task_succeeded=True,  # This function is only called for successful solutions
        )
        success = True
        if verbose:
            print(f"    🎓 SOLUTION RECORDED: {error_code} → {solution}")
        logger.info(f"Rule persisted: {error_code} → {solution}")

    except Exception as e:
        # FIX: Don't silently swallow exceptions - LOG them!
        logger.warning(f"Failed to record solution {error_code} → {solution}: {e}")

    # Secondary: Also store as domain mapping for redundancy
    try:
        await mcp_client.call_tool(
            "store_domain_mapping",
            {
                "domain": domain,
                "mapping_type": "error_solutions",
                "key": error_code,
                "value": solution,
            },
        )
        if verbose:
            print(f"    📁 DOMAIN MAPPING STORED: {domain}/{error_code} → {solution}")
    except Exception as e:
        logger.debug(f"Domain mapping storage failed (non-critical): {e}")

    return success


async def store_experience(
    mcp_client: MCPClientProtocol,
    task: str,
    success: bool,
    strategy: str,
    domain: str,
    error_code: str = "",
    solution: str = "",
    failed_options: str = "",
    condition_key: str = "",
) -> None:
    """
    Store a task experience for learning.

    Args:
        mcp_client: The MCP client
        task: The task that was executed
        success: Whether the task succeeded
        strategy: The strategy used
        domain: The domain name
        error_code: The error code encountered (for failures)
        solution: The working solution found (for successes after retry)
        failed_options: Comma-separated list of options that failed
        condition_key: The composite condition key for multi-condition scenarios
    """
    outcome = "success" if success else "failure"

    # Include actionable information in lessons.
    # Condition codes are embedded so _extract_factual_statement can produce
    # factual knowledge items that overlap with static KB entries for conflict detection.
    cond_suffix = f", Conditions: {condition_key}" if condition_key else ""
    if success and solution:
        lessons = f"Domain: {domain}, Strategy: {strategy}, Solution: {solution}{cond_suffix}"
    elif error_code:
        lessons = f"Domain: {domain}, Error: {error_code}, Failed: {failed_options}{cond_suffix}"
    else:
        lessons = f"Domain: {domain}, Strategy: {strategy}{cond_suffix}"

    await mcp_client.store_experience(
        task=task,
        outcome=outcome,
        strategy=strategy,
        lessons=lessons,
        domain=domain,
        error_code=error_code,
        solution=solution,
        failed_options=failed_options,
        condition_key=condition_key,
    )


def compute_usefulness_feedback(success: bool) -> float:
    """
    Compute memory usefulness feedback based on task outcome.

    Args:
        success: Whether the task succeeded

    Returns:
        Feedback value (positive for success, negative for failure)
    """
    return 0.5 if success else -0.3


async def update_memory_usefulness(
    mcp_client: MCPClientProtocol,
    success: bool,
) -> None:
    """
    Update memory usefulness based on task outcome.

    Args:
        mcp_client: The MCP client
        success: Whether the task succeeded
    """
    try:
        feedback = compute_usefulness_feedback(success)
        await mcp_client.update_memory_usefulness(
            feedback=feedback,
            task_succeeded=success,
        )
    except Exception:
        pass  # Don't fail on usefulness update issues


def should_trigger_consolidation(
    tasks_since_consolidation: int,
    consolidation_interval: int,
) -> bool:
    """
    Check if consolidation should be triggered.

    Args:
        tasks_since_consolidation: Counter for consolidation
        consolidation_interval: Interval for consolidation

    Returns:
        True if consolidation should be triggered
    """
    return tasks_since_consolidation >= consolidation_interval


def should_trigger_compass_evolution(
    tasks_since_compass: int,
    consecutive_failures: int,
    compass_evolution_interval: int,
    failure_threshold: int,
    enable_compass_optimization: bool,
) -> bool:
    """
    Check if COMPASS evolution should be triggered.

    Args:
        tasks_since_compass: Counter for COMPASS evolution
        consecutive_failures: Counter for failures
        compass_evolution_interval: Interval for evolution
        failure_threshold: Threshold for failures
        enable_compass_optimization: Whether optimization is enabled

    Returns:
        True if COMPASS evolution should be triggered
    """
    if not enable_compass_optimization:
        return False

    return (
        tasks_since_compass >= compass_evolution_interval
        or consecutive_failures >= failure_threshold
    )


async def trigger_consolidation(
    mcp_client: MCPClientProtocol,
) -> None:
    """
    Trigger memory consolidation.

    Args:
        mcp_client: The MCP client
    """
    await mcp_client.trigger_consolidation()


async def trigger_compass_evolution(
    mcp_client: MCPClientProtocol,
    failure_context: str,
) -> None:
    """
    Trigger COMPASS prompt evolution.

    Args:
        mcp_client: The MCP client
        failure_context: Context for failed tasks
    """
    await mcp_client.trigger_compass_evolution(failure_context)


def increment_counters(
    tasks_since_consolidation: int,
    tasks_since_compass: int,
) -> Dict[str, int]:
    """
    Increment task counters.

    Args:
        tasks_since_consolidation: Current consolidation counter
        tasks_since_compass: Current COMPASS counter

    Returns:
        Updated counters dict
    """
    return {
        "tasks_since_consolidation": tasks_since_consolidation + 1,
        "tasks_since_compass": tasks_since_compass + 1,
    }


async def store_experience_and_trigger_learning(
    mcp_client: MCPClientProtocol,
    task: str,
    success: bool,
    strategy: str,
    domain: str,
    tasks_since_consolidation: int,
    tasks_since_compass: int,
    consecutive_failures: int,
    config: AgentConfig,
    failure_context: str = "",
    verbose: bool = False,
    error_code: str = "",
    solution: str = "",
    failed_options: str = "",
    condition_key: str = "",
) -> Dict[str, int]:
    """
    Store experience and trigger learning events.

    This is a composition function that orchestrates the learning flow.
    For better testability, use the individual functions:
    - store_experience()
    - update_memory_usefulness()
    - should_trigger_consolidation()
    - trigger_consolidation()
    - should_trigger_compass_evolution()
    - trigger_compass_evolution()

    Args:
        mcp_client: The MCP client
        task: The task that was executed
        success: Whether the task succeeded
        strategy: The strategy used
        domain: The domain name
        tasks_since_consolidation: Counter for consolidation
        tasks_since_compass: Counter for COMPASS evolution
        consecutive_failures: Counter for failures
        config: Agent configuration
        failure_context: Context for failed tasks
        verbose: Whether to print debug info
        error_code: The error code encountered (for failures)
        solution: The working solution found (for successes after retry)
        failed_options: Comma-separated list of options that failed
        condition_key: The composite condition key for multi-condition scenarios

    Returns:
        Updated counters dict
    """
    # Store experience with enriched data
    await store_experience(
        mcp_client,
        task,
        success,
        strategy,
        domain,
        error_code=error_code,
        solution=solution,
        failed_options=failed_options,
        condition_key=condition_key,
    )

    # Update counters
    counters = increment_counters(tasks_since_consolidation, tasks_since_compass)
    tasks_since_consolidation = counters["tasks_since_consolidation"]
    tasks_since_compass = counters["tasks_since_compass"]

    # Update memory usefulness
    await update_memory_usefulness(mcp_client, success)

    # Check and trigger consolidation
    if should_trigger_consolidation(
        tasks_since_consolidation, config.consolidation_interval
    ):
        await trigger_consolidation(mcp_client)
        tasks_since_consolidation = 0

    # Check and trigger COMPASS evolution
    if should_trigger_compass_evolution(
        tasks_since_compass,
        consecutive_failures,
        config.compass_evolution_interval,
        config.failure_threshold,
        config.enable_compass_optimization,
    ):
        await trigger_compass_evolution(mcp_client, failure_context)
        tasks_since_compass = 0
        if verbose:
            print("    📝 COMPASS evolution triggered")

    return {
        "tasks_since_consolidation": tasks_since_consolidation,
        "tasks_since_compass": tasks_since_compass,
    }


# =============================================================================
# LLM REASONING HELPERS (Pure Functions for Testability)
# =============================================================================


def format_error_feedback(error_feedback: str) -> str:
    """
    Format error feedback for LLM prompt.

    Args:
        error_feedback: Raw error message

    Returns:
        Formatted error feedback string
    """
    if not error_feedback:
        return ""
    return f"Previous attempt failed: {error_feedback}"


def build_llm_reasoning_result(
    suggestion: Any,
) -> Optional[Dict[str, Any]]:
    """
    Build LLM reasoning result from parsed suggestion.

    Pure function - no side effects.

    Args:
        suggestion: Parsed LLMSuggestion or None

    Returns:
        Dict with suggested_solution, reasoning, confidence or None
    """
    if not suggestion:
        return None
    return {
        "suggested_solution": suggestion.suggested_solution,
        "reasoning": suggestion.reasoning,
        "confidence": suggestion.confidence,
    }


def update_llm_stats(
    calls: int,
    successes: int,
    failures: int,
    result: Optional[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Update LLM statistics based on result.

    Pure function - returns new stats instead of mutating.

    Args:
        calls: Current call count
        successes: Current success count
        failures: Current failure count
        result: The LLM result (or None if failed)

    Returns:
        Updated stats dict
    """
    new_calls = calls + 1
    if result:
        return {"calls": new_calls, "successes": successes + 1, "failures": failures}
    else:
        return {"calls": new_calls, "successes": successes, "failures": failures + 1}


async def call_llm_for_reasoning(
    model_client: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """
    Make an LLM API call and return the response text.

    This is the ONLY function that performs I/O. Everything else is pure.

    Args:
        model_client: The LLM client
        system_prompt: System message
        user_prompt: User message
        max_tokens: Max tokens for response
        temperature: Temperature for sampling

    Returns:
        Response text from LLM
    """
    try:
        from autogen_core.models import SystemMessage, UserMessage

        response = await model_client.create(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt, source="user"),
            ],
            extra_create_args={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    except ImportError:
        response = await model_client.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            extra_create_args={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

    # Extract text
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, str):
        return response
    else:
        return str(response)


async def perform_llm_reasoning(
    model_client: Any,
    system_prompt: str,
    task: str,
    parsed_task: Any,
    memories: str,
    learned_rules: str,
    forbidden_section: str,
    error_feedback: str,
    prompts: Any,
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    """
    Perform LLM reasoning and return structured result.

    This is a composition function that combines:
    - Pure prompt building
    - I/O for LLM call
    - Pure response parsing

    Easy to test by mocking only call_llm_for_reasoning.

    Args:
        model_client: The LLM client
        system_prompt: Current system prompt
        task: The task string
        parsed_task: Parsed task object
        memories: Retrieved memories
        learned_rules: Learned rules text
        forbidden_section: Forbidden options text
        error_feedback: Error from previous attempt
        prompts: Prompt templates
        max_tokens: Max tokens for LLM
        temperature: Temperature for LLM

    Returns:
        Dict with suggested_solution, reasoning, confidence or None
    """
    # Pure: Build prompt
    # NOTE: We do NOT pass available_options to maintain black swan authenticity.
    # PRECEPT learns through handle_error exploration, not upfront hints.
    formatted_feedback = format_error_feedback(error_feedback)
    reasoning_prompt = build_reasoning_prompt(
        task=task,
        parsed_task=parsed_task,
        memories=memories,
        learned_rules=learned_rules,
        forbidden_section=forbidden_section,
        error_feedback=formatted_feedback,
        prompts=prompts,
    )

    # I/O: Call LLM
    response_text = await call_llm_for_reasoning(
        model_client=model_client,
        system_prompt=system_prompt,
        user_prompt=reasoning_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Pure: Parse response
    result = parse_llm_response(response_text)

    # Pure: Build result
    return build_llm_reasoning_result(result)


# =============================================================================
# TASK RESULT HELPERS (Pure Functions)
# =============================================================================


def build_task_result(
    success: bool,
    task_steps: int,
    overhead_steps: int,
    duration: float,
    response: str,
    strategy: str,
    complexity: str,
    domain: str,
    first_try: bool = False,
    rule_learned: bool = False,
    learned_rule_key: str = "",
    learned_solution: str = "",
    learned_via: str = "",
) -> Dict[str, Any]:
    """
    Build a task result dictionary.

    Args:
        success: Whether the task succeeded
        task_steps: Number of task steps
        overhead_steps: Number of overhead steps
        duration: Duration in seconds
        response: Final response
        strategy: Strategy used
        complexity: Complexity analysis
        domain: Domain name
        first_try: Whether success was achieved without error recovery
        rule_learned: Whether this task created/recorded a learned rule
        learned_rule_key: Condition/error key associated with learned rule
        learned_solution: Solution stored for learned rule
        learned_via: Learning path ("first_try" or "pivot")

    Returns:
        Task result dictionary
    """
    return {
        "success": success,
        "steps": task_steps,
        "overhead": overhead_steps,
        "duration": duration,
        "response": response,
        "strategy": strategy,
        "complexity": complexity,
        "domain": domain,
        "first_try": first_try,
        "rule_learned": rule_learned,
        "learned_rule_key": learned_rule_key,
        "learned_solution": learned_solution,
        "learned_via": learned_via,
    }


def build_task_record(
    task: str,
    success: bool,
    steps: int,
    overhead: int,
    duration: float,
    strategy: str,
) -> Dict[str, Any]:
    """
    Build a task record for COMPASS scoring.

    Args:
        task: The task string
        success: Whether successful
        steps: Number of steps
        overhead: Overhead steps
        duration: Duration
        strategy: Strategy used

    Returns:
        Task record dictionary
    """
    return {
        "task": task,
        "success": success,
        "steps": steps,
        "overhead": overhead,
        "duration": duration,
        "strategy": strategy,
    }


def apply_procedure_hint(
    parsed_task: Any,
    procedure: str,
) -> bool:
    """
    Apply a procedure hint to a parsed task if available.

    Args:
        parsed_task: The parsed task object
        procedure: The procedure string

    Returns:
        True if procedure was applied, False otherwise
    """
    if procedure and "No procedure found" not in procedure:
        parsed_task.parameters["procedure_hint"] = procedure
        return True
    return False


def apply_llm_suggestion(
    parsed_task: Any,
    llm_suggestion: Optional[Dict[str, Any]],
) -> tuple:
    """
    Apply an LLM suggestion to a parsed task.

    Args:
        parsed_task: The parsed task object
        llm_suggestion: The LLM suggestion dict (or None)

    Returns:
        Tuple of (was_applied, strategy_used)
    """
    if not llm_suggestion:
        return False, ""

    suggested = llm_suggestion.get("suggested_solution")
    if suggested:
        parsed_task.parameters["preferred_solution"] = suggested
        reasoning = llm_suggestion.get("reasoning", "applied")[:30]
        strategy_used = f"LLM-Reasoned:{reasoning}"
        return True, strategy_used

    return False, ""


def update_failure_counter(consecutive_failures: int, success: bool) -> int:
    """
    Update the consecutive failures counter.

    Args:
        consecutive_failures: Current counter
        success: Whether the task succeeded

    Returns:
        Updated counter (reset to 0 on success, incremented on failure)
    """
    return 0 if success else consecutive_failures + 1


def format_failure_context(task: str, response: str, success: bool) -> str:
    """
    Format failure context for COMPASS evolution.

    Args:
        task: The task string
        response: The response
        success: Whether successful

    Returns:
        Formatted failure context (empty if success)
    """
    if success:
        return ""
    return f"{task} → {response}"


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================


def compute_success_rate(successful: int, total: int) -> float:
    """Compute success rate as a fraction."""
    if total == 0:
        return 0.0
    return successful / total


def compute_average(values: List[float]) -> float:
    """Compute average of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_agent_stats(
    total_tasks: int,
    successful_tasks: int,
    steps_per_task: List[int],
    domain: str,
    learning_events: List[str],
    pruning_stats: Dict[str, int],
    llm_stats: Dict[str, Any],
    prompt_stats: Dict[str, Any],
    compass_stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build comprehensive agent statistics.

    Args:
        total_tasks: Total tasks executed
        successful_tasks: Number of successful tasks
        steps_per_task: List of steps per task
        domain: Domain name
        learning_events: List of learning events
        pruning_stats: Pruning statistics
        llm_stats: LLM reasoning statistics
        prompt_stats: Prompt evolution statistics
        compass_stats: COMPASS statistics

    Returns:
        Comprehensive statistics dictionary
    """
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": compute_success_rate(successful_tasks, total_tasks),
        "avg_steps": compute_average([float(s) for s in steps_per_task]),
        "domain": domain,
        "learning_events": len(learning_events),
        "compass_stats": compass_stats,
        "prompt_generation": prompt_stats.get("prompt_generation", 0),
        "has_evolved_prompt": prompt_stats.get("has_evolved", False),
        "pruning_stats": pruning_stats,
        "dumb_retries_prevented": pruning_stats.get("dumb_retries_prevented", 0),
        "total_constraints": pruning_stats.get("total_constraints", 0),
        "llm_reasoning_calls": llm_stats.get("total_calls", 0),
        "llm_reasoning_successes": llm_stats.get("successes", 0),
        "llm_reasoning_failures": llm_stats.get("failures", 0),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Protocols
    "MCPClientProtocol",
    "DomainStrategyProtocol",
    "LLMClientProtocol",
    # Data classes
    "LLMSuggestion",
    "TaskResult",
    "ContextFetchResult",
    # Parsing functions
    "parse_llm_response",
    "parse_reflexion_response",
    # Rule filtering
    "filter_rules_by_relevance",
    # Prompt building functions
    "build_reasoning_prompt",
    "build_baseline_prompt",
    "build_reflexion_prompt",
    "build_full_reflexion_prompt",
    # Async helpers
    "parallel_fetch",
    "fetch_context",
    # LLM reasoning helpers (single responsibility, testable)
    "format_error_feedback",
    "build_llm_reasoning_result",
    "update_llm_stats",
    "call_llm_for_reasoning",
    "perform_llm_reasoning",  # Composition
    # Error handling functions (single responsibility)
    "record_error",
    "add_error_constraint",
    "record_error_and_add_constraint",  # Composition (deprecated)
    "record_successful_solution",
    # Rule failure / unlearning functions
    "report_rule_failure",
    "handle_rule_based_failure",
    # Experience and learning functions (single responsibility)
    "store_experience",
    "compute_usefulness_feedback",
    "update_memory_usefulness",
    "should_trigger_consolidation",
    "should_trigger_compass_evolution",
    "trigger_consolidation",
    "trigger_compass_evolution",
    "increment_counters",
    "store_experience_and_trigger_learning",  # Composition
    # Task result helpers (single responsibility)
    "build_task_result",
    "build_task_record",
    "apply_procedure_hint",
    "apply_llm_suggestion",
    "update_failure_counter",
    "format_failure_context",
    # Statistics functions
    "compute_success_rate",
    "compute_average",
    "build_agent_stats",
]
