"""
PRECEPT Baseline Agent Pure Functions Module.

Contains pure functions for baseline agent operations, enabling:
- Dependency injection for testability
- Separation of concerns
- Easier unit testing
- Functional composition

Usage:
    from precept.baseline_functions import (
        parse_baseline_llm_response,
        build_error_context,
        format_accumulated_reflections,
    )

    # Parse LLM response for baseline
    solution = parse_baseline_llm_response(response_text, valid_options)

    # Build error context
    context = build_error_context(failed_options, last_error)
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PromptTemplates

# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE PATHS - For saving/loading baseline memories across subprocesses
# ═══════════════════════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_REFLECTION_PERSIST_PATH = _PROJECT_ROOT / "data" / "full_reflexion_memory.json"
_EXPEL_PERSIST_PATH = _PROJECT_ROOT / "data" / "expel_insights.json"

# =============================================================================
# REFLECTION MEMORY (Shared across FullReflexionBaselineAgent instances)
# =============================================================================

# Global reflection memory storage
# Structure: {task_type: [list of reflection dicts]}
_reflection_memory: Dict[str, List[Dict[str, Any]]] = {}

# Vector store for reflection embeddings (for hybrid retrieval)
_reflection_vector_store: Optional[Any] = None
_reflection_id_counter: int = 0


def _get_reflection_vector_store() -> Optional[Any]:
    """Get or initialize the reflection vector store."""
    global _reflection_vector_store

    if _reflection_vector_store is not None:
        return _reflection_vector_store

    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        _reflection_vector_store = Chroma(
            collection_name="full_reflexion_reflections",
            embedding_function=embeddings,
            persist_directory="./data/chroma_full_reflexion",
        )
        return _reflection_vector_store
    except Exception as e:
        print(f"[Full Reflexion] Vector store init failed (using BM25 only): {e}")
        return None


def get_reflection_memory(task_type: str) -> List[Dict[str, Any]]:
    """Get accumulated reflections for a task type."""
    return _reflection_memory.get(task_type, [])


def add_reflection(
    task_type: str,
    reflection: Dict[str, Any],
    max_size: int = 20,
    enable_vector_store: bool = False,
) -> None:
    """
    Add a reflection to the memory buffer.

    Args:
        task_type: The task type for categorization
        reflection: The reflection dictionary
        max_size: Maximum number of reflections to keep
        enable_vector_store: If True, also add to vector store for hybrid retrieval
    """
    global _reflection_id_counter

    if task_type not in _reflection_memory:
        _reflection_memory[task_type] = []

    # Assign unique ID if not present
    if "id" not in reflection:
        reflection["id"] = f"ref_{_reflection_id_counter}"
        _reflection_id_counter += 1

    _reflection_memory[task_type].append(reflection)

    # Add to vector store if enabled
    if enable_vector_store:
        vector_store = _get_reflection_vector_store()
        if vector_store is not None:
            try:
                # Build document text for embedding
                doc_text = (
                    f"Task: {reflection.get('task', '')} "
                    f"Reflection: {reflection.get('reflection', '')} "
                    f"Lesson: {reflection.get('lesson', '')} "
                    f"Conditions: {' '.join(reflection.get('conditions', []))} "
                    f"Outcome: {reflection.get('outcome', '')}"
                )
                vector_store.add_texts(
                    texts=[doc_text],
                    metadatas=[
                        {
                            "id": reflection["id"],
                            "task_type": task_type,
                            "conditions": ",".join(reflection.get("conditions", [])),
                            "outcome": reflection.get("outcome", ""),
                        }
                    ],
                    ids=[reflection["id"]],
                )
            except Exception as e:
                print(f"[Full Reflexion] Failed to add to vector store: {e}")

    # Prune old reflections (keep most recent)
    if len(_reflection_memory[task_type]) > max_size:
        _reflection_memory[task_type] = _reflection_memory[task_type][-max_size:]


def clear_reflection_memory(task_type: Optional[str] = None) -> None:
    """Clear reflection memory (for testing/reset)."""
    global _reflection_memory, _reflection_vector_store
    if task_type:
        _reflection_memory[task_type] = []
    else:
        _reflection_memory.clear()
        # Also clear vector store
        _reflection_vector_store = None


def save_reflection_memory() -> int:
    """Save reflection memory to disk for cross-subprocess persistence.

    BUGFIX: Python global dicts don't survive across subprocesses. This function
    serializes Full Reflexion's accumulated reflections to JSON so they can be
    reloaded in a subsequent test subprocess (when --preserve-learned-rules is set).
    Without this, baselines are always memoryless at test time, making the
    comparison with PRECEPT (which persists rules via JSON) unfair.

    Returns:
        Number of reflections saved.
    """
    total = sum(len(v) for v in _reflection_memory.values())
    if total == 0:
        return 0

    try:
        _REFLECTION_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Convert to serializable format (strip non-JSON fields)
        serializable = {}
        for task_type, reflections in _reflection_memory.items():
            serializable[task_type] = []
            for ref in reflections:
                # Keep only JSON-serializable fields
                clean_ref = {}
                for k, v in ref.items():
                    try:
                        json.dumps(v)
                        clean_ref[k] = v
                    except (TypeError, ValueError):
                        clean_ref[k] = str(v)
                serializable[task_type].append(clean_ref)

        with open(_REFLECTION_PERSIST_PATH, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  💾 Full Reflexion: Saved {total} reflections to {_REFLECTION_PERSIST_PATH.name}")
        return total
    except Exception as e:
        print(f"  ⚠️ Full Reflexion: Failed to save reflections: {e}")
        return 0


def load_reflection_memory() -> int:
    """Load reflection memory from disk (for cross-subprocess persistence).

    Call at the start of the test subprocess when --preserve-learned-rules is set.

    Returns:
        Number of reflections loaded.
    """
    global _reflection_memory, _reflection_id_counter

    if not _REFLECTION_PERSIST_PATH.exists():
        print(f"  📂 Full Reflexion: No saved reflections found at {_REFLECTION_PERSIST_PATH.name}")
        return 0

    try:
        with open(_REFLECTION_PERSIST_PATH) as f:
            data = json.load(f)

        total = 0
        for task_type, reflections in data.items():
            if task_type not in _reflection_memory:
                _reflection_memory[task_type] = []
            _reflection_memory[task_type].extend(reflections)
            total += len(reflections)

        # Update counter to avoid ID collisions
        _reflection_id_counter = total
        print(f"  📂 Full Reflexion: Loaded {total} reflections from {_REFLECTION_PERSIST_PATH.name}")
        return total
    except Exception as e:
        print(f"  ⚠️ Full Reflexion: Failed to load reflections: {e}")
        return 0


def get_memory_stats() -> Dict[str, int]:
    """Get memory statistics."""
    return {
        task_type: len(reflections)
        for task_type, reflections in _reflection_memory.items()
    }


def retrieve_reflections_with_condition_filter(
    task_type: str,
    task_conditions: List[str],
    max_reflections: int = 10,
    task_description: str = "",
    hybrid_retrieval: bool = False,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    IMPROVED BASELINES: Filter First → Then Rank (BM25, Vector, or Hybrid RRF).

    This function implements a two-stage retrieval for Full Reflexion:
    1. FILTER: Pre-filter reflections by condition metadata (≥1 overlap)
    2. RANK: Within filtered set, rank by:
       - BM25 similarity only (default)
       - BM25 + Vector similarity with RRF fusion (when hybrid_retrieval=True)

    This gives Full Reflexion PRECEPT-like O(1) condition lookup while still
    leveraging similarity-based ranking within relevant candidates.

    Args:
        task_type: The task type to get reflections for
        task_conditions: Conditions in the current task
        max_reflections: Maximum number of reflections to retrieve
        task_description: Task text for BM25/semantic relevance scoring
        hybrid_retrieval: If True, use BM25 + Vector ensemble with RRF fusion
        bm25_weight: Weight for BM25 vs Vector (0-1), only used if hybrid_retrieval=True

    Returns:
        List of reflections, filtered by conditions then ranked by similarity
    """
    reflections = get_reflection_memory(task_type)

    if not reflections:
        return []

    if not task_conditions:
        # No conditions to filter by, return most recent
        return reflections[-max_reflections:]

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: FILTER by condition metadata (O(1)-like pre-filtering)
    # Keep only reflections with at least 1 matching condition
    # ═══════════════════════════════════════════════════════════════════════════
    task_conds_set = set(task_conditions)
    filtered_reflections = []

    for i, ref in enumerate(reflections):
        ref_conditions = set(ref.get("conditions", []))
        if not ref_conditions:
            continue

        # Calculate condition overlap
        is_exact = ref_conditions == task_conds_set
        is_superset = ref_conditions >= task_conds_set
        intersection = ref_conditions & task_conds_set
        overlap = len(intersection)

        if overlap == 0:
            continue  # Skip reflections with no matching conditions

        # Store with overlap info for potential tie-breaking
        filtered_reflections.append(
            {
                "ref": ref,
                "idx": i,
                "is_exact": is_exact,
                "is_superset": is_superset,
                "overlap": overlap,
            }
        )

    if not filtered_reflections:
        # No matching reflections found - return most recent as fallback
        return reflections[-max_reflections:]

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: RANK within filtered set using RRF (Reciprocal Rank Fusion)
    # - Without hybrid: BM25 only
    # - With hybrid: Ensemble retrieval (BM25 + Vector) using RRF fusion
    # ═══════════════════════════════════════════════════════════════════════════
    query = f"Task: {task_description} Conditions: {' '.join(task_conditions)}"
    rrf_k = 60  # RRF constant (standard value)

    # Build RRF scores from retrieval methods
    rrf_scores: Dict[int, float] = {}  # idx -> score

    if len(filtered_reflections) > 1:
        # ───────────────────────────────────────────────────────────────────────
        # Get BM25 rankings (always used)
        # ───────────────────────────────────────────────────────────────────────
        bm25_ranks: Dict[int, int] = {}

        if BM25_AVAILABLE and task_description:
            try:
                from rank_bm25 import BM25Okapi

                # Build corpus from filtered reflections
                corpus = []
                for item in filtered_reflections:
                    ref = item["ref"]
                    text = (
                        f"Task: {ref.get('task', '')} "
                        f"Reflection: {ref.get('reflection', '')} "
                        f"Lesson: {ref.get('lesson', '')} "
                        f"Conditions: {' '.join(ref.get('conditions', []))}"
                    )
                    corpus.append(re.findall(r"\b\w+\b", text.lower()))

                if corpus:
                    bm25 = BM25Okapi(corpus)
                    query_tokens = re.findall(r"\b\w+\b", query.lower())
                    scores = bm25.get_scores(query_tokens)

                    # Get BM25 rankings
                    ranked_indices = sorted(
                        range(len(scores)),
                        key=lambda i: scores[i],
                        reverse=True,
                    )

                    for rank, idx in enumerate(ranked_indices, start=1):
                        if scores[idx] > 0:
                            bm25_ranks[idx] = rank
            except Exception:
                pass

        # Add BM25 RRF scores
        bm25_rrf_weight = 1.0 if not hybrid_retrieval else bm25_weight
        for idx, rank in bm25_ranks.items():
            if idx not in rrf_scores:
                rrf_scores[idx] = 0
            rrf_scores[idx] += bm25_rrf_weight * (1 / (rrf_k + rank))

        # ───────────────────────────────────────────────────────────────────────
        # Get Vector similarity rankings (only when hybrid_retrieval=True)
        # ───────────────────────────────────────────────────────────────────────
        if hybrid_retrieval:
            vector_store = _get_reflection_vector_store()
            semantic_ranks: Dict[int, int] = {}

            if vector_store is not None:
                try:
                    # Get IDs of filtered reflections
                    filtered_ids = {
                        item["ref"].get("id", f"idx_{item['idx']}"): item["idx"]
                        for item in filtered_reflections
                    }

                    # Semantic search
                    search_results = vector_store.similarity_search(
                        query=query,
                        k=100,
                    )

                    rank = 1
                    for doc in search_results:
                        doc_id = doc.metadata.get("id", "")
                        if doc_id in filtered_ids:
                            idx = filtered_ids[doc_id]
                            semantic_ranks[idx] = rank
                            rank += 1
                except Exception:
                    pass

            # Add semantic RRF scores
            semantic_weight = 1 - bm25_weight
            for idx, rank in semantic_ranks.items():
                if idx not in rrf_scores:
                    rrf_scores[idx] = 0
                rrf_scores[idx] += semantic_weight * (1 / (rrf_k + rank))

        # Apply RRF scores to filtered reflections
        for i, item in enumerate(filtered_reflections):
            item["rrf_score"] = rrf_scores.get(i, 0)

        # Sort by: exact match first, then RRF score, then overlap
        def sort_key(item):
            exact_score = (
                1000 if item["is_exact"] else (100 if item["is_superset"] else 0)
            )
            rrf = item.get("rrf_score", 0)
            overlap = item["overlap"]
            # Higher exact_score is better, higher RRF is better, higher overlap is better
            return (-exact_score, -rrf, -overlap)

        filtered_reflections.sort(key=sort_key)

    # Return top matches
    return [item["ref"] for item in filtered_reflections[:max_reflections]]


# =============================================================================
# PURE FUNCTIONS FOR LLM RESPONSE PARSING
# =============================================================================


def extract_solution_from_response(response_text: str) -> Optional[str]:
    """
    Extract the SOLUTION field from an LLM response.

    This is a pure function that extracts the solution using regex.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Extracted solution string or None
    """
    solution_match = re.search(r"SOLUTION:\s*(\S+)", response_text, re.IGNORECASE)
    if solution_match:
        return solution_match.group(1).strip().lower()
    return None


def match_option(suggested: str, valid_options: List[str]) -> Optional[str]:
    """
    Match a suggested solution to valid options.

    This is a pure function that finds the best matching option.

    Args:
        suggested: The suggested solution string
        valid_options: List of valid options

    Returns:
        Matched option or None
    """
    for opt in valid_options:
        if opt.lower() == suggested:
            return opt
    return None


def find_option_in_text(response_text: str, valid_options: List[str]) -> Optional[str]:
    """
    Find any valid option mentioned in the response text.

    This is a pure function for fallback matching.

    Args:
        response_text: Raw text response from LLM
        valid_options: List of valid options

    Returns:
        First matched option or None
    """
    response_lower = response_text.lower()
    # Sort by length descending so longer options match before shorter substrings
    # (e.g., 'salesforce-backup' before 'salesforce')
    for opt in sorted(valid_options, key=len, reverse=True):
        if opt.lower() in response_lower:
            return opt
    return None


def parse_baseline_llm_response(
    response_text: str, valid_options: List[str]
) -> Optional[str]:
    """
    Parse LLM response to extract suggested solution for baseline agents.

    This is a composition function that uses:
    - extract_solution_from_response()
    - match_option()
    - find_option_in_text()

    Args:
        response_text: Raw text response from LLM
        valid_options: List of valid options

    Returns:
        Matched option or None
    """
    # Try to find SOLUTION: pattern
    suggested = extract_solution_from_response(response_text)
    if suggested:
        # Validate against available options
        matched = match_option(suggested, valid_options)
        if matched:
            return matched
        # If not found in options, return as-is (might still work)
        return suggested

    # Fallback: look for any option mentioned in the response
    return find_option_in_text(response_text, valid_options)


# =============================================================================
# PURE FUNCTIONS FOR CONTEXT BUILDING
# =============================================================================


def build_error_context(
    failed_options: List[str],
    last_error: Optional[str],
) -> str:
    """
    Build error context for LLM retry.

    Args:
        failed_options: List of options that have failed
        last_error: The last error message

    Returns:
        Formatted error context string
    """
    if not last_error or not failed_options:
        return ""

    return f"""
⚠️ PREVIOUS ATTEMPT FAILED:
- Tried options: {", ".join(failed_options)}
- Error received: {last_error}
- Please suggest a DIFFERENT option from the available list."""


def build_reflection_section(previous_attempts: List[Dict[str, str]]) -> str:
    """
    Build reflection section from previous attempts.

    Args:
        previous_attempts: List of {option, error, reflection} dicts

    Returns:
        Formatted reflection section string
    """
    if not previous_attempts:
        return ""

    lines = [
        "═══════════════════════════════════════════════════════════════════════════════",
        "PREVIOUS ATTEMPTS (Learn from these failures!):",
    ]

    for i, attempt in enumerate(previous_attempts, 1):
        lines.append(f"\n  Attempt {i}:")
        lines.append(f"    Option tried: {attempt['option']}")
        lines.append(f"    Error received: {attempt['error']}")
        if attempt.get("reflection"):
            lines.append(f"    Your reflection: {attempt['reflection']}")

    lines.append(
        "\n⚠️ You MUST reflect on these failures and choose a DIFFERENT option!"
    )
    lines.append(
        "═══════════════════════════════════════════════════════════════════════════════"
    )

    return "\n".join(lines)


def format_accumulated_reflections(
    task_type: str,
    max_display: int = 10,
    condition_aware: bool = False,  # Ablation: show conditions for PRECEPT-like behavior
    current_conditions: Optional[
        List[str]
    ] = None,  # Filter by current task's conditions
    condition_enhanced_retrieval: bool = False,  # Ablation: strict filtering by conditions
    hybrid_retrieval: bool = False,  # Ablation: BM25 + condition matching
    task_description: str = "",  # Task text for BM25 search
    improved_baselines: bool = False,  # IMPROVED: strict condition-based filtering
) -> str:
    """
    Format accumulated reflections for Full Reflexion prompt.

    Following the Reflexion paper (Shinn et al., 2023), reflections include:
    - The verbal reflection on what happened
    - The lesson learned
    - The TRAJECTORY: which options failed and which succeeded

    ABLATION MODE (condition_aware=True):
    When enabled, shows CONDITIONS for each reflection and prioritizes
    reflections that match the current task's conditions.

    ABLATION MODE (condition_enhanced_retrieval=True):
    When enabled, ONLY retrieves reflections that match the current conditions
    (or have high overlap). This effectively gives Full Reflexion "hash-like" lookup.

    ABLATION MODE (hybrid_retrieval=True):
    When enabled, uses BM25 keyword matching on reflection text + conditions
    combined with condition-based filtering. This helps match reflections
    that mention similar keywords (e.g., condition codes like "FIN-058").

    IMPROVED BASELINES (improved_baselines=True):
    When enabled, uses strict condition-based filtering with exact match priority.
    This gives Full Reflexion O(1)-like lookup similar to PRECEPT's hash lookup.
    Not faithful to the original paper but tests if structured matching helps.

    Args:
        task_type: The task type to get reflections for
        max_display: Maximum number of reflections to display
        condition_aware: If True, show condition codes and prioritize matching (ablation mode)
        current_conditions: Current task's conditions (for filtering in ablation mode)
        condition_enhanced_retrieval: If True, strictly filter by condition match (ablation mode)
        hybrid_retrieval: If True, use BM25 + condition matching (ablation mode)
        task_description: Task text for BM25 relevance scoring
        improved_baselines: If True, use strict condition-based filtering with exact match priority

    Returns:
        Formatted reflections string
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVED BASELINES: Use strict condition-based filtering
    # This gives O(1)-like lookup by prioritizing exact condition matches
    # When hybrid_retrieval is also True, uses BM25+Vector ensemble with RRF fusion
    # ═══════════════════════════════════════════════════════════════════════════
    if improved_baselines and current_conditions:
        display_reflections = retrieve_reflections_with_condition_filter(
            task_type=task_type,
            task_conditions=current_conditions,
            max_reflections=max_display,
            task_description=task_description,  # For BM25/semantic ranking
            hybrid_retrieval=hybrid_retrieval,  # Enable BM25+Vector ensemble with RRF
        )

        if not display_reflections:
            return "No previous reflections matching these specific conditions."

        # Format the retrieved reflections with condition display
        lines = []
        for i, ref in enumerate(display_reflections, 1):
            lines.append(f"\n📝 Episode {ref.get('episode', i)}:")

            # Show conditions with match indicator
            if ref.get("conditions"):
                conditions = ref.get("conditions", [])
                if conditions:
                    current_set = set(current_conditions)
                    ref_set = set(conditions)
                    if ref_set == current_set:
                        lines.append(
                            f"   🎯 CONDITIONS (EXACT MATCH!): {', '.join(conditions)}"
                        )
                    else:
                        overlap = len(ref_set & current_set)
                        lines.append(
                            f"   CONDITIONS ({overlap} overlap): {', '.join(conditions)}"
                        )

            lines.append(f"   Task: {ref.get('task', 'N/A')[:50]}...")
            lines.append(f"   Outcome: {ref.get('outcome', 'N/A')}")
            if ref.get("reflection"):
                lines.append(f"   Reflection: {ref.get('reflection')[:100]}...")
            if ref.get("lesson"):
                lines.append(f"   Lesson: {ref.get('lesson')[:100]}...")
            # Show trajectory (faithful to original paper)
            if ref.get("failed_options"):
                lines.append(
                    f"   ❌ Failed options: {', '.join(ref.get('failed_options', []))}"
                )
            if ref.get("successful_option"):
                lines.append(f"   ✅ Successful option: {ref.get('successful_option')}")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD MODE: Use existing retrieval logic
    # ═══════════════════════════════════════════════════════════════════════════
    reflections = get_reflection_memory(task_type)

    if not reflections:
        return "No previous reflections. This is your first episode of this task type."

    # Filter/Sort logic
    display_reflections = reflections

    # ═══════════════════════════════════════════════════════════════════════
    # HYBRID BM25 RETRIEVAL (when enabled)
    # Uses BM25 keyword matching on reflection text + conditions
    # ═══════════════════════════════════════════════════════════════════════
    bm25_scores: Dict[int, float] = {}
    if hybrid_retrieval and task_description and BM25_AVAILABLE:
        try:
            from rank_bm25 import BM25Okapi

            # Build BM25 corpus from reflections
            corpus = []
            for ref in reflections:
                # Include task, reflection, conditions in searchable text
                ref_text = f"{ref.get('task', '')} {ref.get('reflection', '')} {' '.join(ref.get('conditions', []))}"
                tokens = re.findall(r"\b\w+\b", ref_text.lower())
                corpus.append(tokens)

            if corpus:
                bm25 = BM25Okapi(corpus)
                # Query includes task description + conditions
                query_text = f"{task_description} {' '.join(current_conditions or [])}"
                query_tokens = re.findall(r"\b\w+\b", query_text.lower())
                scores = bm25.get_scores(query_tokens)

                for i, score in enumerate(scores):
                    bm25_scores[i] = score
        except Exception:
            pass  # Fall back to condition-only matching

    if (condition_aware or condition_enhanced_retrieval) and current_conditions:
        current_set = set(current_conditions)

        def condition_match_score(ref: Dict[str, Any]) -> int:
            ref_conditions = set(ref.get("conditions", []))
            if not ref_conditions:
                return 0
            # Exact match gets highest priority
            if ref_conditions == current_set:
                return 100
            # Partial overlap gets some priority
            return len(ref_conditions & current_set)

        # In enhanced retrieval mode, we filter out non-matching reflections
        # This simulates PRECEPT's "hash lookup" behavior
        if condition_enhanced_retrieval:
            display_reflections = [
                r
                for r in reflections
                if condition_match_score(r) > 0  # Must have at least some overlap
            ]
            # If no matches found, fall back to showing recent (or nothing)
            if not display_reflections:
                return "No previous reflections matching these specific conditions."

        # Combined scoring: condition match + BM25 (if hybrid)
        def combined_score(ref: Dict[str, Any], idx: int) -> tuple:
            cond_score = condition_match_score(ref)
            bm25_score = bm25_scores.get(idx, 0) if hybrid_retrieval else 0
            # Weight: 70% condition match, 30% BM25 (condition match more important)
            combined = cond_score * 0.7 + bm25_score * 0.3
            return (combined, ref.get("timestamp", 0))

        # Sort by combined score (highest first), then by recency
        indexed_reflections = list(enumerate(display_reflections))
        indexed_reflections.sort(
            key=lambda x: combined_score(x[1], x[0]),
            reverse=True,
        )
        display_reflections = [r for _, r in indexed_reflections]

    elif hybrid_retrieval and bm25_scores:
        # Pure BM25 sorting (when conditions not provided)
        indexed_reflections = list(enumerate(reflections))
        indexed_reflections.sort(
            key=lambda x: bm25_scores.get(x[0], 0),
            reverse=True,
        )
        display_reflections = [r for _, r in indexed_reflections]

    lines = []
    for i, ref in enumerate(display_reflections[:max_display], 1):
        lines.append(f"\n📝 Episode {ref.get('episode', i)}:")

        # ═══════════════════════════════════════════════════════════════════════
        # ABLATION MODE: Show conditions if condition_aware=True
        # This gives Reflexion PRECEPT-like condition→solution mappings
        # ═══════════════════════════════════════════════════════════════════════
        if (condition_aware or condition_enhanced_retrieval) and ref.get("conditions"):
            conditions = ref.get("conditions", [])
            if conditions:
                # Highlight if conditions match current task
                if current_conditions and set(conditions) == set(current_conditions):
                    lines.append(f"   🎯 CONDITIONS (MATCH!): {', '.join(conditions)}")
                else:
                    lines.append(f"   CONDITIONS: {', '.join(conditions)}")

        lines.append(f"   Task: {ref.get('task', 'N/A')[:50]}...")
        lines.append(f"   Outcome: {ref.get('outcome', 'N/A')}")
        if ref.get("reflection"):
            lines.append(f"   Reflection: {ref.get('reflection')[:100]}...")
        if ref.get("lesson"):
            lines.append(f"   Lesson: {ref.get('lesson')[:100]}...")
        # Show trajectory (faithful to original paper)
        if ref.get("failed_options"):
            lines.append(
                f"   ❌ Failed options: {', '.join(ref.get('failed_options', []))}"
            )
        if ref.get("successful_option"):
            lines.append(f"   ✅ Successful option: {ref.get('successful_option')}")

    return "\n".join(lines)


def build_current_episode_context(current_attempts: List[Dict[str, str]]) -> str:
    """
    Build current episode context for Full Reflexion.

    Args:
        current_attempts: List of attempts in current episode

    Returns:
        Formatted context string
    """
    if not current_attempts:
        return ""

    lines = [
        "\n═══════════════════════════════════════════════════════════════════════════════",
        "CURRENT EPISODE ATTEMPTS:",
    ]

    for i, attempt in enumerate(current_attempts, 1):
        lines.append(f"\n  Attempt {i}:")
        lines.append(f"    Option: {attempt['option']}")
        lines.append(f"    Error: {attempt['error']}")

    lines.append("\n⚠️ Choose a DIFFERENT option!")
    lines.append(
        "═══════════════════════════════════════════════════════════════════════════════"
    )

    return "\n".join(lines)


# =============================================================================
# PURE FUNCTIONS FOR PROMPT BUILDING
# =============================================================================


def build_baseline_llm_prompt(
    task: str,
    parsed_task: Any,
    options: List[str],  # Kept for API compatibility but NOT used in prompt
    memories: str,
    error_context: str = "",
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """
    Build the prompt for baseline LLM reasoning.

    FAIR COMPARISON: Options are NOT included in the prompt.
    The agent must learn through trial-and-error, just like PRECEPT.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        options: Available options (NOT passed to prompt for fair comparison)
        memories: Retrieved memories
        error_context: Error context from previous attempt
        prompts: Optional prompt templates

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    # NOTE: options parameter intentionally NOT passed to prompt
    # This ensures fair comparison with PRECEPT which also doesn't get options
    return prompts.baseline_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=getattr(parsed_task, "task_type", "general"),
        memories=memories or "No relevant memories found.",
        error_context=error_context,
    )


def build_reflexion_llm_prompt(
    task: str,
    parsed_task: Any,
    options: List[str],  # Kept for API compatibility but NOT used in prompt
    memories: str,
    reflection_section: str = "",
    prompts: Optional[PromptTemplates] = None,
) -> str:
    """
    Build the prompt for Reflexion-style reasoning.

    FAIR COMPARISON: Options are NOT included in the prompt.
    The agent must learn through trial-and-error, just like PRECEPT.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        options: Available options (NOT passed to prompt for fair comparison)
        memories: Retrieved memories
        reflection_section: Previous attempts and reflections
        prompts: Optional prompt templates

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    # NOTE: options parameter intentionally NOT passed to prompt
    # This ensures fair comparison with PRECEPT which also doesn't get options
    return prompts.reflexion_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=getattr(parsed_task, "task_type", "general"),
        memories=memories or "No relevant memories found.",
        reflection_section=reflection_section,
    )


def build_full_reflexion_llm_prompt(
    task: str,
    parsed_task: Any,
    options: List[str],  # Used only in improved baselines mode
    task_type: str,
    accumulated_reflections: str,
    current_episode_context: str = "",
    prompts: Optional[PromptTemplates] = None,
    conditions: Optional[List[str]] = None,
    include_options_conditions: bool = False,  # Improved baselines: show options + conditions
) -> str:
    """
    Build the prompt for Full Reflexion with cross-episode memory.

    FAIR COMPARISON: Options are NOT included in the prompt.
    The agent must learn through trial-and-error, just like PRECEPT.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        options: Available options (only passed when include_options_conditions=True)
        task_type: The task type identifier
        accumulated_reflections: Formatted reflections from previous episodes
        current_episode_context: Current episode attempts
        prompts: Optional prompt templates
        conditions: Conditions in the current task (optional)
        include_options_conditions: If True, show options + conditions (improved baselines)

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        prompts = PromptTemplates()

    prompt = prompts.full_reflexion_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=task_type,
        accumulated_reflections=accumulated_reflections,
        current_episode_context=current_episode_context,
    )

    if include_options_conditions:
        conditions_str = ", ".join(conditions or []) if conditions else "None"
        options_str = ", ".join(options) if options else "None"
        prompt += f"""

═══════════════════════════════════════════════════════════════════════════════
IMPROVED BASELINE CONTEXT:
═══════════════════════════════════════════════════════════════════════════════
CONDITIONS (use these to match the right reflection):
{conditions_str}

AVAILABLE OPTIONS (you MUST choose from this list):
{options_str}

Choose the single best option based on the conditions + reflections above.
"""

    return prompt


# =============================================================================
# STATISTICS FUNCTIONS - Pure Computations
# =============================================================================


def compute_success_rate(successful: int, total: int) -> float:
    """
    Compute success rate as a fraction.

    Args:
        successful: Number of successful outcomes
        total: Total number of outcomes

    Returns:
        Success rate (0.0 to 1.0)
    """
    if total == 0:
        return 0.0
    return successful / total


def compute_average_steps(steps_per_task: List[int]) -> float:
    """
    Compute average steps per task.

    Args:
        steps_per_task: List of step counts

    Returns:
        Average steps (0.0 if empty)
    """
    if not steps_per_task:
        return 0.0
    return sum(steps_per_task) / len(steps_per_task)


def compute_per_task_rate(count: int, total_tasks: int) -> float:
    """
    Compute a per-task rate.

    Args:
        count: The count to divide
        total_tasks: Total number of tasks

    Returns:
        Rate per task (0.0 if no tasks)
    """
    if total_tasks == 0:
        return 0.0
    return count / total_tasks


def compute_llm_accuracy(suggestions_followed: int, total_calls: int) -> float:
    """
    Compute LLM suggestion accuracy.

    Args:
        suggestions_followed: Number of suggestions that led to success
        total_calls: Total LLM calls made

    Returns:
        Accuracy rate (0.0 to 1.0)
    """
    if total_calls == 0:
        return 0.0
    return suggestions_followed / total_calls


def build_core_stats(
    total_tasks: int,
    successful_tasks: int,
    steps_per_task: List[int],
    llm_calls: int,
    llm_suggestions_followed: int,
    llm_suggestions_failed: int,
    domain: str,
    baseline_type: str,
) -> Dict[str, Any]:
    """
    Build core statistics common to all baseline types.

    Args:
        total_tasks: Total tasks executed
        successful_tasks: Number of successful tasks
        steps_per_task: List of steps per task
        llm_calls: Total LLM calls
        llm_suggestions_followed: Suggestions that led to success
        llm_suggestions_failed: Suggestions that failed
        domain: Domain name
        baseline_type: Type of baseline

    Returns:
        Core statistics dictionary
    """
    return {
        "total_tasks": total_tasks,
        "successful_tasks": successful_tasks,
        "success_rate": compute_success_rate(successful_tasks, total_tasks),
        "avg_steps": compute_average_steps(steps_per_task),
        "llm_calls": llm_calls,
        "llm_calls_per_task": compute_per_task_rate(llm_calls, total_tasks),
        "llm_suggestions_followed": llm_suggestions_followed,
        "llm_suggestions_failed": llm_suggestions_failed,
        "llm_accuracy": compute_llm_accuracy(llm_suggestions_followed, llm_calls),
        "domain": domain,
        "baseline_type": baseline_type,
    }


def add_reflexion_stats(
    stats: Dict[str, Any],
    reflections_generated: int,
    total_tasks: int,
) -> Dict[str, Any]:
    """
    Add reflexion-specific statistics to a stats dict.

    Args:
        stats: The base statistics dictionary
        reflections_generated: Number of reflections generated
        total_tasks: Total tasks executed

    Returns:
        Updated statistics dictionary
    """
    stats["reflections_generated"] = reflections_generated
    stats["reflections_per_task"] = compute_per_task_rate(
        reflections_generated, total_tasks
    )
    return stats


def add_full_reflexion_stats(
    stats: Dict[str, Any],
    reflections_reused: int,
    total_episodes: int,
) -> Dict[str, Any]:
    """
    Add full reflexion-specific statistics to a stats dict.

    Args:
        stats: The base statistics dictionary
        reflections_reused: Number of reflections reused
        total_episodes: Total episodes

    Returns:
        Updated statistics dictionary
    """
    stats["reflections_reused"] = reflections_reused
    stats["total_episodes"] = total_episodes
    stats["memory_stats"] = get_memory_stats()
    return stats


def build_baseline_stats(
    total_tasks: int,
    successful_tasks: int,
    steps_per_task: List[int],
    llm_calls: int,
    llm_suggestions_followed: int,
    llm_suggestions_failed: int,
    domain: str,
    baseline_type: str,
    reflections_generated: int = 0,
    reflections_reused: int = 0,
    total_episodes: int = 0,
) -> Dict[str, Any]:
    """
    Build comprehensive baseline agent statistics.

    This is a composition function that uses:
    - build_core_stats()
    - add_reflexion_stats()
    - add_full_reflexion_stats()

    Args:
        total_tasks: Total tasks executed
        successful_tasks: Number of successful tasks
        steps_per_task: List of steps per task
        llm_calls: Total LLM calls
        llm_suggestions_followed: Suggestions that led to success
        llm_suggestions_failed: Suggestions that failed
        domain: Domain name
        baseline_type: Type of baseline (adapted_react, adapted_reflexion, full_reflexion)
        reflections_generated: Number of reflections generated
        reflections_reused: Number of reflections reused
        total_episodes: Total episodes (for full reflexion)

    Returns:
        Comprehensive statistics dictionary
    """
    # Build core stats
    stats = build_core_stats(
        total_tasks=total_tasks,
        successful_tasks=successful_tasks,
        steps_per_task=steps_per_task,
        llm_calls=llm_calls,
        llm_suggestions_followed=llm_suggestions_followed,
        llm_suggestions_failed=llm_suggestions_failed,
        domain=domain,
        baseline_type=baseline_type,
    )

    # Add reflexion-specific stats
    if baseline_type in ("adapted_reflexion", "full_reflexion"):
        stats = add_reflexion_stats(stats, reflections_generated, total_tasks)

    # Add full reflexion-specific stats
    if baseline_type == "full_reflexion":
        stats = add_full_reflexion_stats(stats, reflections_reused, total_episodes)

    return stats


def create_reflection_record(
    episode: int,
    task: str,
    success: bool,
    reflection: Optional[str],
    lesson: Optional[str],
    failed_options: List[str],
    successful_option: Optional[str],
    attempts: int,
    conditions: Optional[List[str]] = None,  # Ablation: store conditions
) -> Dict[str, Any]:
    """
    Create a reflection record for cross-episode memory.

    Args:
        episode: Episode number
        task: The task string
        success: Whether the task succeeded
        reflection: The reflection text
        lesson: The lesson learned
        failed_options: Options that failed
        successful_option: The option that worked
        attempts: Number of attempts
        conditions: Optional list of condition codes (for ablation studies)

    Returns:
        Reflection record dictionary
    """
    return {
        "episode": episode,
        "task": task,
        "outcome": "success" if success else "failure",
        "reflection": reflection,
        "lesson": lesson,
        "failed_options": failed_options,
        "successful_option": successful_option,
        "attempts": attempts,
        "conditions": conditions or [],  # Store conditions for ablation
        "timestamp": time.time(),
    }


# =============================================================================
# ExpeL (Experiential Learning) FUNCTIONS - Zhao et al., 2023
# =============================================================================
#
# Faithful implementation based on: "ExpeL: LLM Agents Are Experiential Learners"
# https://arxiv.org/abs/2308.10144
#
# Key components from the paper:
# 1. Experience Collection: Gather trajectories from training tasks
# 2. Insight Extraction: Extract generalizable insights using LLM
# 3. Insight Storage: Store insights with task context for VECTOR RETRIEVAL
# 4. Retrieval: Find relevant insights via SEMANTIC SIMILARITY (embeddings)
#
# IMPORTANT: ExpeL uses its own ISOLATED ChromaDB collection (chroma_expel/)
# to avoid any data contamination with PRECEPT's vector stores.
#
# =============================================================================

from pathlib import Path

# Global insight storage for ExpeL (in-memory backup)
# Structure: List of {
#   insight: str,           - The generalizable insight text
#   conditions: List[str],  - Conditions this insight applies to
#   solution: str,          - Recommended solution (for success)
#   avoid: List[str],       - Options to avoid (for failure)
#   confidence: str,        - high/medium/low
#   type: str,              - 'success' or 'failure'
#   task: str,              - Original task for similarity matching
#   trajectory: Dict,       - Full trajectory info (attempts, options tried)
#   timestamp: float        - When this insight was extracted
#   id: str                 - Unique ID for vector store reference
# }
_expel_insight_store: List[Dict[str, Any]] = []

# ExpeL's own vector store (ISOLATED from PRECEPT)
_expel_vector_store = None
_expel_insight_counter = 0


def _get_expel_vector_store():
    """
    Get or initialize ExpeL's isolated vector store.

    Uses a SEPARATE ChromaDB collection at data/chroma_expel/ to ensure
    complete isolation from PRECEPT's vector stores.
    """
    global _expel_vector_store

    if _expel_vector_store is not None:
        return _expel_vector_store

    try:
        from langchain_chroma import Chroma
        from langchain_openai import OpenAIEmbeddings

        # ExpeL's ISOLATED persist directory - separate from PRECEPT
        expel_persist_dir = Path("data/chroma_expel")
        expel_persist_dir.mkdir(parents=True, exist_ok=True)

        import os

        embedding_model = os.getenv("PRECEPT_EMBEDDING_MODEL", "text-embedding-3-small")
        embeddings = OpenAIEmbeddings(model=embedding_model)

        _expel_vector_store = Chroma(
            collection_name="expel_insights",  # Unique collection name
            embedding_function=embeddings,
            persist_directory=str(expel_persist_dir),
        )

        return _expel_vector_store
    except ImportError:
        # Fall back to in-memory only if ChromaDB not available
        return None
    except Exception as e:
        # Continue with in-memory fallback
        print(f"[ExpeL] Vector store init failed (using in-memory): {e}")
        return None


def get_expel_insights() -> List[Dict[str, Any]]:
    """Get all stored ExpeL insights."""
    return _expel_insight_store.copy()


def add_expel_insight(
    insight: Dict[str, Any],
    condition_enhanced: bool = False,  # Ablation: include conditions in embedding
    improved_baselines: bool = False,  # IMPROVED: store conditions as structured metadata
) -> None:
    """
    Add an insight to the ExpeL store with vector indexing.

    Following the ExpeL paper, insights are extracted from task trajectories
    and stored for future retrieval during inference using SEMANTIC SIMILARITY.

    Args:
        insight: Dictionary containing:
            - insight: The insight text (generalizable pattern)
            - conditions: List of conditions this applies to
            - solution: Recommended solution (for success insights)
            - avoid: Options to avoid (for failure insights)
            - confidence: Confidence level (high/medium/low)
            - type: 'success' or 'failure'
            - task: Original task description (for similarity retrieval)
            - trajectory: Execution trajectory info
        condition_enhanced: If True, include condition codes in the vector embedding text.
                            This tests if vector search can distinguish conditions (ablation).
        improved_baselines: If True, store conditions as structured metadata for filtering.
                            This enables O(1)-like lookup using ChromaDB's 'where' clause.
    """
    global _expel_insight_counter

    insight["timestamp"] = time.time()
    _expel_insight_counter += 1
    insight["id"] = f"expel_insight_{_expel_insight_counter}"

    # Add to in-memory store (always)
    _expel_insight_store.append(insight)

    # Add to vector store for semantic similarity retrieval
    vector_store = _get_expel_vector_store()
    if vector_store is not None:
        try:
            # Create searchable text combining task and insight
            task_text = insight.get("task", "")
            insight_text = insight.get("insight", "")

            # ═══════════════════════════════════════════════════════════════════
            # ABLATION: Condition-Enhanced Vector Search
            # If enabled, we embed the condition codes into the vector.
            # This tests if embeddings can learn to distinguish tasks by condition key.
            # ═══════════════════════════════════════════════════════════════════
            condition_text = ""
            if condition_enhanced and insight.get("conditions"):
                condition_text = (
                    f" Conditions: {' '.join(insight.get('conditions', []))}"
                )

            searchable_text = f"""Task: {task_text}{condition_text}
Insight: {insight_text}
Type: {insight.get("type", "unknown")}"""

            # ═══════════════════════════════════════════════════════════════════
            # IMPROVED BASELINES: Store conditions as structured metadata
            # When enabled, we store conditions in metadata so they can be used
            # for pre-filtering with ChromaDB's 'where' clause.
            # This gives O(1)-like lookup while still using semantic similarity.
            # ═══════════════════════════════════════════════════════════════════
            metadata = {
                "id": insight["id"],
                "type": insight.get("type", "unknown"),
                "confidence": insight.get("confidence", "low"),
            }

            if improved_baselines and insight.get("conditions"):
                # Store conditions as a comma-separated string (ChromaDB metadata)
                # ChromaDB doesn't support list metadata, so we join them
                conditions_list = insight.get("conditions", [])
                metadata["conditions_str"] = ",".join(sorted(conditions_list))
                # Also store individual conditions for $contains queries
                # ChromaDB supports string matching, so we create a searchable format
                metadata["conditions_count"] = len(conditions_list)
                # Store solution for direct lookup (IMPROVED mode only)
                if insight.get("solution"):
                    metadata["solution"] = insight.get("solution")
                # Store the first few conditions as separate fields for filtering
                for i, cond in enumerate(conditions_list[:5]):  # Max 5 conditions
                    metadata[f"cond_{i}"] = cond

            vector_store.add_texts(
                texts=[searchable_text],
                metadatas=[metadata],
                ids=[insight["id"]],
            )
        except Exception as e:
            # Continue with in-memory backup if vector store fails
            print(f"[ExpeL] Vector store add failed: {e}")


def clear_expel_insights() -> None:
    """Clear all ExpeL insights (for testing/reset between experiments)."""
    global _expel_insight_store, _expel_vector_store, _expel_insight_counter

    _expel_insight_store.clear()
    _expel_insight_counter = 0

    # Clear vector store
    if _expel_vector_store is not None:
        try:
            _expel_vector_store.delete_collection()
        except Exception:
            pass
        _expel_vector_store = None

    # Also clean up persist directory to ensure fresh start
    try:
        import shutil

        expel_persist_dir = Path("data/chroma_expel")
        if expel_persist_dir.exists():
            shutil.rmtree(expel_persist_dir)
    except Exception:
        pass


def save_expel_insights() -> int:
    """Save ExpeL insights to disk for cross-subprocess persistence.

    BUGFIX: Python global lists don't survive across subprocesses. This function
    serializes ExpeL's accumulated insights to JSON so they can be reloaded in a
    subsequent test subprocess (when --preserve-learned-rules is set).
    Without this, ExpeL insights extracted during training are completely lost
    at test time, making the comparison with PRECEPT unfair.

    Returns:
        Number of insights saved.
    """
    if not _expel_insight_store:
        return 0

    try:
        _EXPEL_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for insight in _expel_insight_store:
            clean = {}
            for k, v in insight.items():
                try:
                    json.dumps(v)
                    clean[k] = v
                except (TypeError, ValueError):
                    clean[k] = str(v)
            serializable.append(clean)

        with open(_EXPEL_PERSIST_PATH, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  💾 ExpeL: Saved {len(serializable)} insights to {_EXPEL_PERSIST_PATH.name}")
        return len(serializable)
    except Exception as e:
        print(f"  ⚠️ ExpeL: Failed to save insights: {e}")
        return 0


def load_expel_insights() -> int:
    """Load ExpeL insights from disk (for cross-subprocess persistence).

    Call at the start of the test subprocess when --preserve-learned-rules is set.

    Returns:
        Number of insights loaded.
    """
    global _expel_insight_store, _expel_insight_counter

    if not _EXPEL_PERSIST_PATH.exists():
        print(f"  📂 ExpeL: No saved insights found at {_EXPEL_PERSIST_PATH.name}")
        return 0

    try:
        with open(_EXPEL_PERSIST_PATH) as f:
            data = json.load(f)

        _expel_insight_store.extend(data)
        _expel_insight_counter = len(_expel_insight_store)
        print(f"  📂 ExpeL: Loaded {len(data)} insights from {_EXPEL_PERSIST_PATH.name}")
        return len(data)
    except Exception as e:
        print(f"  ⚠️ ExpeL: Failed to load insights: {e}")
        return 0


def get_expel_stats() -> Dict[str, Any]:
    """Get ExpeL insight statistics."""
    success_count = sum(1 for i in _expel_insight_store if i.get("type") == "success")
    failure_count = sum(1 for i in _expel_insight_store if i.get("type") == "failure")
    high_conf = sum(1 for i in _expel_insight_store if i.get("confidence") == "high")

    vector_store = _get_expel_vector_store()
    vector_count = 0
    if vector_store is not None:
        try:
            vector_count = vector_store._collection.count()
        except Exception:
            pass

    return {
        "total_insights": len(_expel_insight_store),
        "success_insights": success_count,
        "failure_insights": failure_count,
        "high_confidence_insights": high_conf,
        "vector_store_count": vector_count,
        "vector_store_enabled": vector_store is not None,
    }


def extract_conditions_from_task(task: str) -> List[str]:
    """
    Extract condition codes from a task string.

    Args:
        task: Task string like "Book shipment... [Conditions: R-482 + C-HZMT]"

    Returns:
        List of condition codes like ["R-482", "C-HZMT"]
    """
    # Match patterns like [Conditions: R-482 + C-HZMT + P-220]
    match = re.search(r"\[Conditions:\s*([^\]]+)\]", task)
    if match:
        conditions_str = match.group(1)
        # Split by + and clean up
        conditions = [c.strip() for c in conditions_str.split("+")]
        return [c for c in conditions if c]  # Filter empty strings
    return []


def build_expel_insight_extraction_prompt(
    task: str,
    success: bool,
    attempts: int,
    failed_options: List[str],
    successful_option: Optional[str],
    conditions: List[str],
    errors: Optional[List[str]] = None,
    prompts: Optional["PromptTemplates"] = None,
    improved_baselines: bool = False,
) -> str:
    """
    Build prompt for ExpeL insight extraction.

    Args:
        task: The task string
        success: Whether the task succeeded
        attempts: Number of attempts
        failed_options: Options that failed
        successful_option: The option that worked (if success)
        conditions: List of condition codes
        errors: List of error messages (for failures)
        prompts: Prompt templates
        improved_baselines: If True, use PRECEPT-like specific condition→solution prompts.
                           This is NOT faithful to ExpeL paper but allows fair comparison.

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        from .config import PromptTemplates

        prompts = PromptTemplates()

    conditions_str = ", ".join(conditions) if conditions else "None detected"
    failed_options_str = ", ".join(failed_options) if failed_options else "None"

    if success:
        if improved_baselines:
            # IMPROVED MODE: Store SPECIFIC condition→solution mappings
            return prompts.expel_insight_extraction_success_improved.format(
                task=task,
                attempts=attempts,
                successful_option=successful_option or "unknown",
                failed_options=failed_options_str,
                conditions=conditions_str,
            )
        else:
            # DEFAULT MODE: Store GENERAL insights (faithful to ExpeL paper)
            return prompts.expel_insight_extraction_success.format(
                task=task,
                attempts=attempts,
                successful_option=successful_option or "unknown",
                failed_options=failed_options_str,
                conditions=conditions_str,
            )
    else:
        if improved_baselines:
            # IMPROVED MODE: Store SPECIFIC options to avoid
            return prompts.expel_insight_extraction_failure_improved.format(
                task=task,
                attempts=attempts,
                failed_options=failed_options_str,
                errors=", ".join(errors) if errors else "None",
                conditions=conditions_str,
            )
        else:
            # DEFAULT MODE: Store GENERAL failure insights (faithful to ExpeL paper)
            return prompts.expel_insight_extraction_failure.format(
                task=task,
                attempts=attempts,
                failed_options=failed_options_str,
                errors=", ".join(errors) if errors else "None",
                conditions=conditions_str,
            )


def parse_expel_insight_response(
    response_text: str,
    is_success: bool,
) -> Dict[str, Any]:
    """
    Parse LLM response for ExpeL insight extraction.

    Following the ExpeL paper, we extract structured insights that can be
    retrieved and applied to similar future tasks.

    Args:
        response_text: Raw LLM response
        is_success: Whether this was a success or failure insight

    Returns:
        Parsed insight dictionary with:
            - insight: The generalizable pattern/rule
            - conditions: List of condition codes this applies to
            - solution: Recommended solution (success) or None
            - avoid: Options to avoid (failure) or empty list
            - confidence: high/medium/low
            - type: 'success' or 'failure'
    """
    result = {
        "insight": None,
        "conditions": [],
        "solution": None,
        "avoid": [],
        "confidence": "low",
        "type": "success" if is_success else "failure",
    }

    # Extract INSIGHT - the core generalizable pattern
    insight_match = re.search(
        r"INSIGHT:\s*(.+?)(?=\n[A-Z_]+:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if insight_match:
        result["insight"] = insight_match.group(1).strip()

    # Extract CONDITIONS_COVERED - which conditions this insight applies to
    conditions_match = re.search(
        r"CONDITIONS_COVERED:\s*(.+?)(?=\n[A-Z_]+:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if conditions_match:
        conditions_str = conditions_match.group(1).strip()
        # Parse conditions - match patterns like R-482, C-HZMT, P-220
        conditions = re.findall(r"[A-Z]+-\d+|[A-Z]+-[A-Z]+", conditions_str)
        result["conditions"] = conditions

    # Extract SOLUTION (for success insights) - the working solution
    solution_match = re.search(r"SOLUTION:\s*(\S+)", response_text, re.IGNORECASE)
    if solution_match:
        solution = solution_match.group(1).strip().lower()
        # Clean up any trailing punctuation
        solution = re.sub(r"[,.\s]+$", "", solution)
        result["solution"] = solution

    # Extract AVOID (for failure insights) - options that don't work
    avoid_match = re.search(
        r"AVOID:\s*(.+?)(?=\n[A-Z_]+:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if avoid_match:
        avoid_str = avoid_match.group(1).strip()
        # Parse options to avoid - split by comma, space, or 'and'
        avoid_options = [
            opt.strip().lower()
            for opt in re.split(r"[,\s]+|and", avoid_str)
            if opt.strip() and opt.strip().lower() not in ["", "and", "or"]
        ]
        result["avoid"] = avoid_options

    # Extract CONFIDENCE - how generalizable this insight is
    conf_match = re.search(
        r"CONFIDENCE:\s*(high|medium|low)",
        response_text,
        re.IGNORECASE,
    )
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    return result


def retrieve_expel_insights_by_task(
    task: str,
    task_conditions: List[str],
    top_k: int = 10,  # Match Full Reflexion's max_display=10 for fairness
    condition_enhanced: bool = False,  # Ablation: include conditions in search query
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant insights using VECTOR SIMILARITY on task descriptions.

    Following the ExpeL paper (Zhao et al., 2023), retrieval finds insights from
    similar past experiences using SEMANTIC SIMILARITY on task embeddings.

    Args:
        task: The current task description (for semantic similarity)
        task_conditions: Conditions in the current task (for condition boost)
        top_k: Number of insights to retrieve
        condition_enhanced: If True, include condition codes in search query (ablation)

    Returns:
        List of relevant insights sorted by semantic similarity
    """
    if not _expel_insight_store:
        return []

    # Try vector similarity first (primary method per ExpeL paper)
    vector_store = _get_expel_vector_store()
    if vector_store is not None:
        try:
            # Semantic similarity search
            search_query = f"Task: {task}"

            # ═══════════════════════════════════════════════════════════════════
            # ABLATION: Condition-Enhanced Vector Search
            # If enabled, include condition codes in the query embedding.
            # ═══════════════════════════════════════════════════════════════════
            if condition_enhanced and task_conditions:
                search_query += f" Conditions: {' '.join(task_conditions)}"

            docs = vector_store.similarity_search(search_query, k=top_k)

            if docs:
                # Map vector results back to full insight objects
                retrieved_ids = set()
                results = []

                for doc in docs:
                    insight_id = doc.metadata.get("id", "")
                    if insight_id and insight_id not in retrieved_ids:
                        retrieved_ids.add(insight_id)
                        # Find the full insight in our store
                        for insight in _expel_insight_store:
                            if insight.get("id") == insight_id:
                                results.append(insight)
                                break

                if results:
                    return results[:top_k]
        except Exception as e:
            # Fall back to condition-based retrieval
            print(f"[ExpeL] Vector search failed, using fallback: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FAIR COMPARISON: Do NOT fallback to condition-based retrieval
    # ═══════════════════════════════════════════════════════════════════════════
    # The fallback `retrieve_expel_insights_by_conditions` uses Jaccard similarity
    # on condition codes, which effectively gives ExpeL O(1)-like exact matching.
    # This would be an unfair advantage over PRECEPT's O(1) hash lookup.
    #
    # Per ExpeL paper: retrieval should be SEMANTIC SIMILARITY ONLY.
    # If vector store is unavailable or returns empty, return empty (no insights).
    # The agent must explore without prior knowledge - fair cold start.
    # ═══════════════════════════════════════════════════════════════════════════
    return []  # No fallback to condition matching - pure semantic only


def retrieve_expel_insights_by_conditions(
    task_conditions: List[str],
    top_k: int = 10,  # Match Full Reflexion's max_display=10 for fairness
    prioritize_exact_match: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant insights based on condition similarity (FALLBACK method).

    This is the FALLBACK retrieval method when vector store is unavailable.
    Uses Jaccard similarity on condition sets.

    The retrieval strategy:
    1. EXACT MATCH: Prioritize insights where conditions match exactly
    2. SUBSET MATCH: Include insights that cover a subset of current conditions
    3. JACCARD SIMILARITY: Score by overall condition overlap

    Args:
        task_conditions: Conditions in the current task
        top_k: Number of insights to retrieve
        prioritize_exact_match: If True, exact condition matches get priority

    Returns:
        List of relevant insights sorted by relevance
    """
    if not _expel_insight_store or not task_conditions:
        return []

    task_conditions_set = set(task_conditions)
    scored_insights = []

    for insight in _expel_insight_store:
        insight_conditions = set(insight.get("conditions", []))
        if not insight_conditions:
            continue

        # Calculate various similarity scores
        intersection = task_conditions_set & insight_conditions
        union = task_conditions_set | insight_conditions

        # Jaccard similarity (standard)
        jaccard = len(intersection) / len(union) if union else 0

        # Check for exact match
        is_exact_match = task_conditions_set == insight_conditions

        # Check if insight conditions are subset of task conditions
        is_applicable = insight_conditions <= task_conditions_set

        # Check if task conditions are subset of insight conditions
        is_subset = task_conditions_set <= insight_conditions

        # Compute final score with priority weighting
        if prioritize_exact_match:
            if is_exact_match:
                score = 2.0 + jaccard
            elif is_applicable:
                score = 1.5 + jaccard
            elif is_subset:
                score = 1.0 + jaccard
            else:
                score = jaccard
        else:
            score = jaccard

        if score > 0:
            scored_insights.append((score, is_exact_match, insight))

    # Sort by score descending, then by exact match
    scored_insights.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [insight for _, _, insight in scored_insights[:top_k]]


def retrieve_expel_insights_with_metadata_filter(
    task: str,
    task_conditions: List[str],
    top_k: int = 10,
    hybrid_retrieval: bool = False,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    IMPROVED BASELINES: Filter First → Then Rank (Vector, BM25, or Hybrid).

    This function implements a two-stage retrieval:
    1. FILTER: Pre-filter insights by condition metadata (≥1 overlap)
    2. RANK: Within filtered set, rank by:
       - Vector similarity only (default)
       - BM25 + Vector similarity hybrid (when hybrid_retrieval=True)

    This gives baselines PRECEPT-like O(1) condition lookup while still
    leveraging similarity-based ranking within relevant candidates.

    Args:
        task: The current task description (for semantic/BM25 similarity)
        task_conditions: Conditions in the current task (for filtering)
        top_k: Number of insights to retrieve
        hybrid_retrieval: If True, use BM25 + vector similarity for ranking
        bm25_weight: Weight for BM25 vs semantic (0-1), only used if hybrid_retrieval=True

    Returns:
        List of relevant insights, filtered by conditions then ranked by similarity
    """
    if not _expel_insight_store:
        return []

    if not task_conditions:
        # No conditions to filter by, fall back to semantic-only
        return retrieve_expel_insights_by_task(
            task=task,
            task_conditions=[],
            top_k=top_k,
            condition_enhanced=False,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: FILTER by condition metadata (O(1)-like pre-filtering)
    # Keep only insights with at least 1 matching condition
    # ═══════════════════════════════════════════════════════════════════════════
    task_conds_set = set(task_conditions)
    sorted_task_conds = ",".join(sorted(task_conditions))
    filtered_insights = []

    for insight in _expel_insight_store:
        insight_conditions = insight.get("conditions", [])
        if not insight_conditions:
            continue

        insight_conds_set = set(insight_conditions)
        sorted_insight_conds = ",".join(sorted(insight_conditions))

        # Calculate condition overlap
        is_exact = sorted_insight_conds == sorted_task_conds
        intersection = task_conds_set & insight_conds_set
        overlap = len(intersection)

        if overlap == 0:
            continue  # Skip insights with no matching conditions

        # Store with overlap info for potential tie-breaking
        filtered_insights.append(
            {
                "insight": insight,
                "is_exact": is_exact,
                "overlap": overlap,
                "id": insight.get("id", ""),
            }
        )

    if not filtered_insights:
        # No matching insights found - fall back to semantic search
        return retrieve_expel_insights_by_task(
            task=task,
            task_conditions=task_conditions,
            top_k=top_k,
            condition_enhanced=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: RANK within filtered set using RRF (Reciprocal Rank Fusion)
    # - Without hybrid: Vector similarity only
    # - With hybrid: Ensemble retrieval (BM25 + Vector) using LangChain or fallback
    # ═══════════════════════════════════════════════════════════════════════════
    query = f"Task: {task} Conditions: {' '.join(task_conditions)}"
    filtered_ids = {item["id"] for item in filtered_insights}
    rrf_k = 60  # RRF constant (standard value)

    # Build RRF scores from both retrieval methods
    rrf_scores: Dict[str, float] = {}

    if len(filtered_insights) > 1:
        # ───────────────────────────────────────────────────────────────────────
        # Get Vector similarity rankings
        # ───────────────────────────────────────────────────────────────────────
        vector_store = _get_expel_vector_store()
        semantic_ranks: Dict[str, int] = {}

        if vector_store is not None:
            try:
                search_results = vector_store.similarity_search(
                    query=query,
                    k=min(len(_expel_insight_store), 100),
                )
                rank = 1
                for doc in search_results:
                    doc_id = doc.metadata.get("id", "")
                    if doc_id in filtered_ids:
                        semantic_ranks[doc_id] = rank
                        rank += 1
            except Exception:
                pass

        # Add semantic RRF scores (always used)
        semantic_weight = 1.0 if not hybrid_retrieval else (1 - bm25_weight)
        for doc_id, rank in semantic_ranks.items():
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += semantic_weight * (1 / (rrf_k + rank))

        # ───────────────────────────────────────────────────────────────────────
        # Get BM25 rankings (only when hybrid_retrieval=True)
        # Uses same approach as retrieve_expel_insights_hybrid
        # ───────────────────────────────────────────────────────────────────────
        if hybrid_retrieval and BM25_AVAILABLE:
            try:
                from rank_bm25 import BM25Okapi

                # Build corpus from filtered insights only
                corpus = []
                id_list = []
                for item in filtered_insights:
                    insight = item["insight"]
                    # Include task, insight, conditions for BM25 matching
                    text = (
                        f"Task: {insight.get('task', '')} "
                        f"Insight: {insight.get('insight', '')} "
                        f"Conditions: {' '.join(insight.get('conditions', []))} "
                        f"Solution: {insight.get('solution', '')}"
                    )
                    corpus.append(re.findall(r"\b\w+\b", text.lower()))
                    id_list.append(item["id"])

                if corpus:
                    bm25 = BM25Okapi(corpus)
                    query_tokens = re.findall(r"\b\w+\b", query.lower())
                    scores = bm25.get_scores(query_tokens)

                    # Get BM25 rankings
                    ranked_indices = sorted(
                        range(len(scores)),
                        key=lambda i: scores[i],
                        reverse=True,
                    )

                    # Add BM25 RRF scores
                    for rank, idx in enumerate(ranked_indices, start=1):
                        if scores[idx] > 0:  # Only count if score > 0
                            doc_id = id_list[idx]
                            if doc_id not in rrf_scores:
                                rrf_scores[doc_id] = 0
                            rrf_scores[doc_id] += bm25_weight * (1 / (rrf_k + rank))
            except Exception:
                pass

        # Apply RRF scores to filtered insights
        for item in filtered_insights:
            doc_id = item["id"]
            item["rrf_score"] = rrf_scores.get(doc_id, 0)

        # Sort by: exact match first, then RRF score, then overlap
        def sort_key(item):
            exact_score = 1000 if item["is_exact"] else 0
            rrf = item.get("rrf_score", 0)
            overlap_score = item["overlap"]
            # Higher exact_score is better, higher RRF is better, higher overlap is better
            return (-exact_score, -rrf, -overlap_score)

        filtered_insights.sort(key=sort_key)

    # Return top-k insights
    return [item["insight"] for item in filtered_insights[:top_k]]


def format_expel_insights_for_prompt(
    insights: List[Dict[str, Any]],
    max_insights: int = 10,  # Match Full Reflexion's max_display=10 for fairness
    condition_aware: bool = False,  # Ablation: show conditions for PRECEPT-like behavior
) -> str:
    """
    Format retrieved insights for inclusion in the task prompt.

    Following the ExpeL paper (Zhao et al., 2023), insights include:
    - The generalizable pattern/insight text
    - The SOLUTION that worked (for success insights)
    - Options to AVOID (for failure insights)

    ABLATION MODE (condition_aware=True):
    When enabled, also shows the CONDITIONS that the insight applies to.
    This gives ExpeL PRECEPT-like condition information for ablation studies.

    Args:
        insights: List of insight dictionaries
        max_insights: Maximum number to include (default 10 for fairness with Full Reflexion)
        condition_aware: If True, show condition codes (ablation mode)

    Returns:
        Formatted string for prompt
    """
    if not insights:
        return "No relevant insights from past experience. This is your first encounter with this type of task."

    lines = []
    for i, insight in enumerate(insights[:max_insights], 1):
        insight_type = insight.get("type", "unknown")
        confidence = insight.get("confidence", "unknown")
        emoji = "✅" if insight_type == "success" else "❌"
        conf_emoji = (
            "🔴" if confidence == "high" else "🟡" if confidence == "medium" else "⚪"
        )

        lines.append(
            f"\n{emoji} Insight {i} ({insight_type.upper()}) {conf_emoji} [{confidence} confidence]:"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # ABLATION MODE: Show conditions if condition_aware=True
        # This gives ExpeL PRECEPT-like condition→solution mappings
        # ═══════════════════════════════════════════════════════════════════════
        if condition_aware and insight.get("conditions"):
            conditions = insight.get("conditions", [])
            if conditions:
                lines.append(f"   CONDITIONS: {', '.join(conditions)}")

        # Show the insight pattern
        lines.append(f"   PATTERN: {insight.get('insight', 'N/A')}")

        # For success insights, show the solution that worked
        if insight_type == "success" and insight.get("solution"):
            lines.append(f"   → SOLUTION: {insight.get('solution')}")

        # For failure insights, show what to avoid
        if insight_type == "failure" and insight.get("avoid"):
            lines.append(f"   → AVOID: {', '.join(insight.get('avoid', []))}")

    return "\n".join(lines)


def build_expel_task_prompt(
    task: str,
    parsed_task: Any,
    options: List[str],  # Used only in improved baselines mode
    conditions: List[str],  # Used only in improved baselines mode
    insights: List[Dict[str, Any]],
    current_episode_context: str = "",
    prompts: Optional["PromptTemplates"] = None,
    condition_aware: bool = False,  # Ablation: show conditions in insights
    include_options_conditions: bool = False,  # Improved baselines: show options + conditions
) -> str:
    """
    Build the prompt for ExpeL task execution.

    FAIR COMPARISON (default):
    - Options are NOT included in the prompt.
    - Conditions are NOT included in the prompt (removed for Black Swan fairness).
    The agent must learn through trial-and-error, just like PRECEPT.

    ABLATION MODE (condition_aware=True):
    - Shows CONDITIONS in insights for PRECEPT-like behavior comparison.

    Args:
        task: The raw task string
        parsed_task: The parsed task object
        options: Available options (only passed when include_options_conditions=True)
        conditions: Conditions in this task (only passed when include_options_conditions=True)
        insights: Retrieved relevant insights
        current_episode_context: Context from current episode attempts
        prompts: Prompt templates
        condition_aware: If True, show conditions in insights (ablation mode)
        include_options_conditions: If True, show options + conditions (improved baselines)

    Returns:
        Formatted prompt string
    """
    if prompts is None:
        from .config import PromptTemplates

        prompts = PromptTemplates()

    insights_str = format_expel_insights_for_prompt(
        insights, condition_aware=condition_aware
    )

    prompt = prompts.expel_task_prompt.format(
        task=task,
        action=getattr(parsed_task, "action", "unknown"),
        entity=getattr(parsed_task, "entity", "unknown"),
        source=getattr(parsed_task, "source", None) or "N/A",
        target=getattr(parsed_task, "target", None) or "N/A",
        task_type=getattr(parsed_task, "task_type", "general"),
        insights=insights_str,
        current_episode_context=current_episode_context,
    )

    if include_options_conditions:
        conditions_str = ", ".join(conditions) if conditions else "None"
        options_str = ", ".join(options) if options else "None"
        prompt += f"""

═══════════════════════════════════════════════════════════════════════════════
IMPROVED BASELINE CONTEXT:
═══════════════════════════════════════════════════════════════════════════════
CONDITIONS (use these to match the right insight):
{conditions_str}

AVAILABLE OPTIONS (you MUST choose from this list):
{options_str}

Choose the single best option based on the conditions + insights above.
"""

    return prompt


def parse_expel_task_response(
    response_text: str,
    valid_options: List[str],
) -> Dict[str, Optional[str]]:
    """
    Parse LLM response for ExpeL task execution.

    Args:
        response_text: Raw LLM response
        valid_options: List of valid options

    Returns:
        Dictionary with solution, insight_applied, reasoning, confidence
    """
    result = {
        "solution": None,
        "insight_applied": None,
        "reasoning": None,
        "confidence": "low",
    }

    # Extract INSIGHT_APPLIED
    insight_match = re.search(
        r"INSIGHT_APPLIED:\s*(.+?)(?=\n[A-Z_]+:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if insight_match:
        result["insight_applied"] = insight_match.group(1).strip()

    # Extract SOLUTION
    solution_match = re.search(r"SOLUTION:\s*(\S+)", response_text, re.IGNORECASE)
    if solution_match:
        suggested = solution_match.group(1).strip().lower()
        # Match to valid options
        for opt in valid_options:
            if opt.lower() == suggested:
                result["solution"] = opt
                break
        if not result["solution"]:
            # Fallback: look for any option in response
            # Sort by length descending so longer options match before shorter substrings
            response_lower = response_text.lower()
            for opt in sorted(valid_options, key=len, reverse=True):
                if opt.lower() in response_lower:
                    result["solution"] = opt
                    break

    # Extract REASONING
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)", response_text, re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract CONFIDENCE
    conf_match = re.search(
        r"CONFIDENCE:\s*(high|medium|low)", response_text, re.IGNORECASE
    )
    if conf_match:
        result["confidence"] = conf_match.group(1).lower()

    return result


def add_expel_stats(
    stats: Dict[str, Any],
    insights_extracted: int,
    insights_retrieved: int,
    insights_applied: int,
    total_tasks: int,
) -> Dict[str, Any]:
    """
    Add ExpeL-specific statistics to a stats dict.

    Args:
        stats: The base statistics dictionary
        insights_extracted: Number of insights extracted
        insights_retrieved: Number of insights retrieved
        insights_applied: Number of insights successfully applied
        total_tasks: Total tasks executed

    Returns:
        Updated statistics dictionary
    """
    stats["insights_extracted"] = insights_extracted
    stats["insights_retrieved"] = insights_retrieved
    stats["insights_applied"] = insights_applied
    stats["insight_application_rate"] = (
        insights_applied / total_tasks if total_tasks > 0 else 0.0
    )
    stats["insight_store_stats"] = get_expel_stats()
    return stats


# =============================================================================
# HYBRID BM25 + SEMANTIC RETRIEVAL (Using LangChain)
# =============================================================================
#
# This module provides hybrid retrieval combining:
# 1. BM25 (lexical/keyword matching) - Good for exact terms, condition codes
# 2. Semantic Search (embedding similarity) - Good for conceptual matching
#
# Uses LangChain's BM25Retriever and EnsembleRetriever for production-quality
# hybrid search with Reciprocal Rank Fusion (RRF).
# =============================================================================

# Check for LangChain hybrid retrieval dependencies
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document

    LANGCHAIN_BM25_AVAILABLE = True
except ImportError:
    LANGCHAIN_BM25_AVAILABLE = False

# Fallback to raw rank_bm25 if LangChain not available
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def create_hybrid_retriever(
    documents: List[str],
    document_ids: List[str],
    vector_store: Any,
    bm25_weight: float = 0.4,
) -> Optional[Any]:
    """
    Create a LangChain EnsembleRetriever combining BM25 and semantic search.

    Args:
        documents: List of document texts
        document_ids: List of document IDs (must match documents)
        vector_store: LangChain vector store for semantic search
        bm25_weight: Weight for BM25 (0-1), semantic gets (1-bm25_weight)

    Returns:
        EnsembleRetriever or None if dependencies unavailable
    """
    if not LANGCHAIN_BM25_AVAILABLE:
        return None

    if not documents:
        return None

    try:
        # Create LangChain Documents with metadata
        lc_docs = [
            Document(page_content=doc, metadata={"id": doc_id})
            for doc, doc_id in zip(documents, document_ids)
        ]

        # Create BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(lc_docs)
        bm25_retriever.k = 10  # Number of results

        # Create semantic retriever from vector store
        semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # Create ensemble with RRF fusion
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[bm25_weight, 1 - bm25_weight],
        )

        return ensemble
    except Exception as e:
        print(f"[Hybrid] Failed to create ensemble retriever: {e}")
        return None


class HybridRetrieverFallback:
    """
    Fallback hybrid retriever using raw rank_bm25 (when LangChain unavailable).

    Uses Reciprocal Rank Fusion (RRF) to merge rankings from both methods.
    """

    def __init__(
        self,
        documents: List[str],
        document_ids: Optional[List[str]] = None,
        rrf_k: int = 60,
    ):
        self.documents = documents
        self.document_ids = document_ids or [str(i) for i in range(len(documents))]
        self.rrf_k = rrf_k

        if BM25_AVAILABLE and documents:
            tokenized_docs = [self._tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def bm25_search(self, query: str, top_k: int = 10) -> List[tuple]:
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self.document_ids[i], scores[i]) for i in ranked_indices if scores[i] > 0
        ]

    def hybrid_rerank(
        self,
        bm25_results: List[tuple],
        semantic_results: List[tuple],
        top_k: int = 10,
        bm25_weight: float = 0.5,
    ) -> List[str]:
        rrf_scores = {}

        for rank, (doc_id, _) in enumerate(bm25_results, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += bm25_weight * (1 / (self.rrf_k + rank))

        semantic_weight = 1 - bm25_weight
        for rank, (doc_id, _) in enumerate(semantic_results, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += semantic_weight * (1 / (self.rrf_k + rank))

        ranked_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        return ranked_ids[:top_k]


# Global hybrid retriever for ExpeL insights
_expel_hybrid_retriever: Optional[HybridRetrieverFallback] = None
_expel_langchain_ensemble: Optional[Any] = None


def _get_expel_hybrid_retriever(use_langchain: bool = True) -> Optional[Any]:
    """Get or create hybrid retriever for ExpeL insights."""
    global _expel_hybrid_retriever, _expel_langchain_ensemble

    if not _expel_insight_store:
        return None

    # Prepare documents
    documents = []
    doc_ids = []
    for insight in _expel_insight_store:
        task_text = insight.get("task", "")
        insight_text = insight.get("insight", "")
        conditions = insight.get("conditions", [])

        doc_text = f"Task: {task_text} Insight: {insight_text} Conditions: {' '.join(conditions)}"
        documents.append(doc_text)
        doc_ids.append(insight.get("id", str(len(documents))))

    # Try LangChain ensemble first
    if use_langchain and LANGCHAIN_BM25_AVAILABLE:
        vector_store = _get_expel_vector_store()
        if vector_store and (
            _expel_langchain_ensemble is None
            or len(_expel_insight_store) != len(documents)
        ):
            _expel_langchain_ensemble = create_hybrid_retriever(
                documents, doc_ids, vector_store
            )
        if _expel_langchain_ensemble:
            return _expel_langchain_ensemble

    # Fallback to raw BM25
    if BM25_AVAILABLE:
        if _expel_hybrid_retriever is None or len(_expel_insight_store) != len(
            _expel_hybrid_retriever.documents
        ):
            _expel_hybrid_retriever = HybridRetrieverFallback(documents, doc_ids)
        return _expel_hybrid_retriever

    return None


def retrieve_expel_insights_hybrid(
    task: str,
    task_conditions: List[str],
    top_k: int = 10,
    bm25_weight: float = 0.4,
    condition_enhanced: bool = False,  # Combined mode: include conditions in semantic query
) -> List[Dict[str, Any]]:
    """
    Retrieve ExpeL insights using HYBRID BM25 + Semantic search.

    Uses LangChain's EnsembleRetriever if available, falls back to raw BM25.

    When both --hybrid-retrieval and --condition-enhanced-retrieval are active,
    this function combines both approaches:
    - BM25: Always includes conditions in keyword search
    - Semantic: Includes conditions in embedding query when condition_enhanced=True

    Args:
        task: The current task description
        task_conditions: Conditions in the current task
        top_k: Number of insights to retrieve
        bm25_weight: Weight for BM25 vs semantic (0-1)
        condition_enhanced: If True, include conditions in semantic search query too

    Returns:
        List of relevant insights
    """
    if not _expel_insight_store:
        return []

    # BM25 query always includes conditions (keyword matching benefits from them)
    query_with_conditions = f"Task: {task} Conditions: {' '.join(task_conditions)}"

    # Semantic query: include conditions only if condition_enhanced is True
    semantic_query = f"Task: {task}"
    if condition_enhanced and task_conditions:
        semantic_query += f" Conditions: {' '.join(task_conditions)}"

    retriever = _get_expel_hybrid_retriever()

    # Try LangChain EnsembleRetriever
    if LANGCHAIN_BM25_AVAILABLE and hasattr(retriever, "invoke"):
        try:
            docs = retriever.invoke(query_with_conditions)
            retrieved_ids = [doc.metadata.get("id") for doc in docs[:top_k]]
            id_to_insight = {
                insight.get("id"): insight for insight in _expel_insight_store
            }
            return [
                id_to_insight[doc_id]
                for doc_id in retrieved_ids
                if doc_id in id_to_insight
            ]
        except Exception as e:
            print(f"[ExpeL Hybrid] LangChain retrieval failed: {e}")

    # Fallback to manual hybrid
    if isinstance(retriever, HybridRetrieverFallback):
        bm25_results = retriever.bm25_search(query_with_conditions, top_k=top_k * 2)

        semantic_results = []
        vector_store = _get_expel_vector_store()
        if vector_store:
            try:
                # Use semantic_query which may or may not include conditions
                docs = vector_store.similarity_search(semantic_query, k=top_k * 2)
                for i, doc in enumerate(docs):
                    insight_id = doc.metadata.get("id", "")
                    if insight_id:
                        semantic_results.append((insight_id, 1.0 / (i + 1)))
            except Exception:
                pass

        if bm25_results or semantic_results:
            ranked_ids = retriever.hybrid_rerank(
                bm25_results, semantic_results, top_k=top_k, bm25_weight=bm25_weight
            )
            id_to_insight = {
                insight.get("id"): insight for insight in _expel_insight_store
            }
            return [
                id_to_insight[doc_id]
                for doc_id in ranked_ids
                if doc_id in id_to_insight
            ]

    return []


def clear_expel_hybrid_retriever() -> None:
    """Clear the hybrid retriever (call when insights are cleared)."""
    global _expel_hybrid_retriever, _expel_langchain_ensemble
    _expel_hybrid_retriever = None
    _expel_langchain_ensemble = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Reflection memory functions
    "get_reflection_memory",
    "add_reflection",
    "clear_reflection_memory",
    "get_memory_stats",
    "retrieve_reflections_with_condition_filter",  # IMPROVED: strict condition filtering
    # Response parsing (single responsibility)
    "extract_solution_from_response",
    "match_option",
    "find_option_in_text",
    "parse_baseline_llm_response",  # Composition
    # Context building
    "build_error_context",
    "build_reflection_section",
    "format_accumulated_reflections",
    "build_current_episode_context",
    # Prompt building
    "build_baseline_llm_prompt",
    "build_reflexion_llm_prompt",
    "build_full_reflexion_llm_prompt",
    # Statistics (single responsibility)
    "compute_success_rate",
    "compute_average_steps",
    "compute_per_task_rate",
    "compute_llm_accuracy",
    "build_core_stats",
    "add_reflexion_stats",
    "add_full_reflexion_stats",
    "build_baseline_stats",  # Composition
    "create_reflection_record",
    # ExpeL (Experiential Learning) functions
    "get_expel_insights",
    "add_expel_insight",
    "clear_expel_insights",
    "get_expel_stats",
    "extract_conditions_from_task",
    "build_expel_insight_extraction_prompt",
    "parse_expel_insight_response",
    "retrieve_expel_insights_by_task",  # Primary: Vector similarity (ExpeL paper)
    "retrieve_expel_insights_by_conditions",  # Fallback: Condition-based
    "retrieve_expel_insights_with_metadata_filter",  # IMPROVED: metadata filtering
    "format_expel_insights_for_prompt",
    "build_expel_task_prompt",
    "parse_expel_task_response",
    "add_expel_stats",
    # Hybrid BM25 + Semantic retrieval (LangChain-based)
    "create_hybrid_retriever",
    "HybridRetrieverFallback",
    "retrieve_expel_insights_hybrid",
    "clear_expel_hybrid_retriever",
    "BM25_AVAILABLE",
    "LANGCHAIN_BM25_AVAILABLE",
]
