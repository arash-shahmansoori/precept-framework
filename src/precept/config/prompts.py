"""
Prompt Templates for PRECEPT.

Centralized prompt templates for all agents (PRECEPT and baselines).

Usage:
    from precept.config.prompts import PromptTemplates

    prompts = PromptTemplates()
    reasoning_prompt = prompts.reasoning_prompt.format(...)
"""

from dataclasses import dataclass


@dataclass
class PromptTemplates:
    """Centralized prompt templates."""

    reasoning_prompt: str = """Based on your learned rules and domain knowledge, analyze this task:

TASK: {task}

PARSED CONTEXT:
- Action: {action}
- Entity: {entity}
- Source/Origin: {source}
- Target/Destination: {target}
- Task Type: {task_type}
- Additional Parameters: {parameters}

{procedure_section}
═══════════════════════════════════════════════════════════════════════════════
🔴 LEARNED RULES (Format: CONDITION_KEY → SOLUTION)
═══════════════════════════════════════════════════════════════════════════════
Rules are stored as "condition_key → solution" mappings learned from experience.
When the current task's conditions match a rule's condition_key:
  - Apply the associated SOLUTION immediately (no guessing needed)
  - Condition keys may be single codes (E-101) or composites (A+B+C)
  - These rules generalize across different task instances with the same conditions

APPLY THESE RULES PROACTIVELY (check for matching condition patterns):
{learned_rules}

═══════════════════════════════════════════════════════════════════════════════
🟡 SIMILAR PAST EXPERIENCES (patterns, not definitive)
═══════════════════════════════════════════════════════════════════════════════
{memories}

═══════════════════════════════════════════════════════════════════════════════
DECISION STRATEGY:
1. CHECK RULES FIRST: If Source/Origin matches a known problematic entity,
   look for a rule with that error code and use its solution IMMEDIATELY
2. IF no matching rule: Use memory patterns as hints
3. IF no matches at all: Respond "EXPLORE"
═══════════════════════════════════════════════════════════════════════════════

{forbidden_section}
{error_feedback_section}
⚠️ CRITICAL: Any option in the FORBIDDEN list has probability 0.0 - DO NOT suggest it.
If all options are forbidden, respond with "EXHAUSTED".

Suggest the solution from LEARNED RULES if applicable, otherwise EXPLORE.

Respond EXACTLY in this format:
SOLUTION: <solution_from_rule OR EXPLORE OR EXHAUSTED>
REASONING: <one sentence: which rule applied or why exploring>
CONFIDENCE: <high if from rule, medium from memory, low if exploring>"""

    # =========================================================================
    # FAIR COMPARISON: No valid options given upfront (like PRECEPT)
    # All agents must learn through trial-and-error interaction with MCP server
    # =========================================================================

    baseline_prompt: str = """You are an AI assistant helping with domain-specific tasks.
Analyze this task and suggest the best solution through reasoning.

TASK: {task}

CONTEXT:
- Action: {action}
- Entity: {entity}
- Source/Origin: {source}
- Target/Destination: {target}
- Task Type: {task_type}

RETRIEVED MEMORIES (may or may not be relevant):
{memories}

{error_context}

Based on the task, context, and any patterns from memory, suggest a solution.
You must reason about what option might work - no list of valid options is provided.

⚠️ Error messages are intentionally vague and do NOT reveal which option works.
⚠️ Learn through trial and error by remembering what worked/failed before.

Respond in this format:
SOLUTION: <your_suggested_option>
REASONING: <brief explanation based on task context and memories>
CONFIDENCE: <high/medium/low>"""

    reflexion_prompt: str = """You are an AI assistant helping with domain-specific tasks.
You learn from failures by reflecting on what went wrong.

TASK: {task}

CONTEXT:
- Action: {action}
- Entity: {entity}
- Source/Origin: {source}
- Target/Destination: {target}
- Task Type: {task_type}

RETRIEVED MEMORIES (may or may not be relevant):
{memories}

{reflection_section}

═══════════════════════════════════════════════════════════════════════════════
INSTRUCTIONS:
1. If there was a previous failure, REFLECT on why it failed
2. Based on your reflection, suggest a DIFFERENT option
3. Use your reflections to guide your next guess

⚠️ Error codes are intentionally vague (e.g., CFN-881, K8S-101) and do NOT hint at solutions.
⚠️ No list of valid options is provided - you must learn through trial and error.
═══════════════════════════════════════════════════════════════════════════════

Respond in this EXACT format:
REFLECTION: <What went wrong? Why did this option fail? What pattern do you notice?>
LESSON: <What should I do differently this time?>
SOLUTION: <your_suggested_option based on reflection>
REASONING: <Why this option should work based on your reflection>
CONFIDENCE: <high/medium/low>"""

    full_reflexion_prompt: str = """You are a reflective AI agent that learns from experience.
You have access to reflections from PREVIOUS EPISODES of similar tasks.

TASK: {task}

CONTEXT:
- Action: {action}
- Entity: {entity}
- Source/Origin: {source}
- Target/Destination: {target}
- Task Type: {task_type}

═══════════════════════════════════════════════════════════════════════════════
ACCUMULATED REFLECTIONS (From previous episodes - USE THESE!):
{accumulated_reflections}
═══════════════════════════════════════════════════════════════════════════════

{current_episode_context}

INSTRUCTIONS:
1. Review accumulated reflections from previous episodes
2. Apply lessons learned to avoid repeating mistakes
3. If this is a retry, reflect on what went wrong
4. Suggest the best option based on your reflections and accumulated knowledge

⚠️ Error codes are intentionally vague and do NOT hint at solutions.
⚠️ No list of valid options is provided - learn through trial and error using your reflections.

Respond in this EXACT format:
REFLECTION: <What patterns do you notice? What should you avoid?>
LESSON: <Key insight for this and future episodes>
SOLUTION: <your_suggested_option based on reflections>
REASONING: <Why this option, based on reflections>
CONFIDENCE: <high/medium/low>"""

    # =========================================================================
    # ExpeL (Experiential Learning) Prompts - Zhao et al., 2023
    # =========================================================================
    #
    # Based on: "ExpeL: LLM Agents Are Experiential Learners"
    # https://arxiv.org/abs/2308.10144
    #
    # Key principles from the paper:
    # 1. Learn from both successes and failures
    # 2. Extract GENERALIZABLE insights (not task-specific rules)
    # 3. Use natural language to capture nuanced patterns
    # 4. Retrieve relevant past experiences for new tasks
    # =========================================================================

    # =========================================================================
    # ExpeL Insight Extraction - NATURAL LANGUAGE ONLY (Fair Comparison)
    # =========================================================================
    # Per ExpeL paper: insights should be GENERALIZABLE patterns, NOT explicit
    # condition-to-solution mappings. In Black Swan scenarios, there are no
    # learnable patterns - solutions depend on arbitrary hashes. ExpeL should
    # learn general strategies, not memorize specific answers.
    # =========================================================================

    expel_insight_extraction_success: str = """You are an AI that extracts generalizable insights from successful task executions.

TASK COMPLETED SUCCESSFULLY:
{task}

EXECUTION TRAJECTORY:
- Total attempts made: {attempts}
- Final outcome: success after {attempts} attempt(s)
- Some options didn't work before finding the right one: {failed_options}

═══════════════════════════════════════════════════════════════════════════════
YOUR GOAL: Extract a GENERAL observation about this success
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: You must extract a GENERAL insight that captures the NATURE of the task,
NOT a specific rule mapping conditions to solutions. The insight should describe:
- What TYPE of task this was (trading, compliance, order processing, etc.)
- What GENERAL APPROACH might work for similar tasks
- What you OBSERVED about the process

⚠️ DO NOT include specific condition codes in your insight!
⚠️ DO NOT reveal the specific solution that worked!
⚠️ Insights should be GENERAL observations, not memorized answers.

EXAMPLE GOOD INSIGHTS:
✓ "Order processing tasks with multiple compliance requirements need careful exploration"
✓ "When initial attempts fail, systematically trying different options can lead to success"
✓ "Complex trading scenarios may require non-standard order types"

EXAMPLE BAD INSIGHTS:
✗ "When CPL-REG+FIN-062 is present, use order_type_b" (Too specific - memorization!)
✗ "Use limit orders when you see compliance codes" (Reveals solution type!)

Respond in this EXACT format:
INSIGHT: <A general observation about the task type and approach>
CONFIDENCE: <high if clear pattern, medium if some uncertainty, low if unclear>"""

    expel_insight_extraction_failure: str = """You are an AI that extracts generalizable insights from failed task executions.

TASK FAILED (exhausted all attempts):
{task}

EXECUTION TRAJECTORY:
- Total attempts made: {attempts}
- Tried multiple options but none succeeded
- Error messages received: {errors}

═══════════════════════════════════════════════════════════════════════════════
YOUR GOAL: Extract a GENERAL observation about why this task was difficult
═══════════════════════════════════════════════════════════════════════════════

IMPORTANT: You must extract a GENERAL insight about task difficulty,
NOT a specific list of options to avoid for specific conditions.

⚠️ DO NOT include specific condition codes in your insight!
⚠️ DO NOT list specific options to avoid!
⚠️ Insights should be GENERAL observations about task difficulty.

EXAMPLE GOOD INSIGHTS:
✓ "Some complex scenarios may require exploring all available options"
✓ "Tasks with cryptic error codes may not have learnable patterns"
✓ "Certain combinations of requirements make standard approaches fail"

EXAMPLE BAD INSIGHTS:
✗ "When CPL-REG+FIN-062 is present, avoid order_type_a" (Too specific!)
✗ "Avoid limit orders when you see compliance codes" (Reveals answer!)

Respond in this EXACT format:
INSIGHT: <A general observation about why this task was difficult>
AVOID: <Options that definitely don't work for these conditions>
CONFIDENCE: <high/medium/low>"""

    # =========================================================================
    # ExpeL Task Execution Prompt - FAIR COMPARISON (No Condition Codes)
    # =========================================================================
    # Per ExpeL paper: agent uses SEMANTIC SIMILARITY to find relevant insights.
    # In true Black Swan CSPs, there are no learnable patterns - solutions are
    # arbitrary. ExpeL must rely on general insights, not condition matching.
    # =========================================================================

    expel_task_prompt: str = """You are an AI agent that learns from past experiences.
You have access to GENERAL insights extracted from previous task executions.

CURRENT TASK: {task}

TASK CONTEXT:
- Action: {action}
- Entity: {entity}
- Source: {source}
- Destination: {target}
- Task Type: {task_type}

═══════════════════════════════════════════════════════════════════════════════
GENERAL INSIGHTS FROM PAST EXPERIENCE:
{insights}
═══════════════════════════════════════════════════════════════════════════════

{current_episode_context}

═══════════════════════════════════════════════════════════════════════════════
HOW TO USE INSIGHTS:
═══════════════════════════════════════════════════════════════════════════════
The insights above are GENERAL observations about similar tasks.
They describe patterns and approaches, NOT specific solutions.

Use these insights to guide your REASONING about what approach might work.
You must explore and learn through trial and error.

⚠️ No list of valid options is provided - you must discover them.
⚠️ Insights are general patterns, not memorized answers.
⚠️ Each task may have a unique solution that requires exploration.

Respond in this EXACT format:
INSIGHT_APPLIED: <Which insight number helped your reasoning, or "None">
SOLUTION: <your_suggested_option based on reasoning>
REASONING: <Why this option? Explain your logic>
CONFIDENCE: <high/medium/low based on uncertainty>"""

    # =========================================================================
    # IMPROVED BASELINES: ExpeL with PRECEPT-like Condition→Solution Storage
    # =========================================================================
    # These prompts are used when --improved-baselines flag is set.
    # They allow ExpeL to store SPECIFIC condition→solution mappings,
    # giving it PRECEPT-like O(1) lookup capability for fair comparison.
    #
    # WARNING: This is NOT faithful to the ExpeL paper. It's an ablation
    # study to understand if the difference is in representation vs application.
    # =========================================================================

    expel_insight_extraction_success_improved: str = """You are an AI that extracts SPECIFIC, ACTIONABLE insights from successful task executions.

TASK COMPLETED SUCCESSFULLY:
{task}

EXECUTION TRAJECTORY:
- Total attempts made: {attempts}
- SUCCESSFUL SOLUTION: {successful_option}
- Failed options tried before success: {failed_options}
- CONDITIONS PRESENT: {conditions}

═══════════════════════════════════════════════════════════════════════════════
YOUR GOAL: Extract a SPECIFIC rule that maps CONDITIONS to SOLUTION
═══════════════════════════════════════════════════════════════════════════════

You MUST extract:
1. The EXACT condition codes that were present (e.g., C-COLD, H-903, LA-550)
2. The EXACT solution that worked (e.g., antwerp, hamburg)
3. A brief explanation of why this mapping works

This is for a machine learning system that needs EXACT matches.
Be SPECIFIC and PRECISE - vague insights are useless here.

EXAMPLE GOOD INSIGHTS:
✓ "When conditions C-COLD, H-903, LA-550 are present, use 'antwerp' as the solution"
✓ "For condition combination R-482 + T-NGHT + C-HZMT, select 'hamburg'"

EXAMPLE BAD INSIGHTS:
✗ "Try different options when shipping perishables" (Too vague!)
✗ "Explore systematically" (No actionable information!)

Respond in this EXACT format:
INSIGHT: When conditions {conditions} are present, the correct solution is {successful_option}
CONDITIONS_COVERED: {conditions}
SOLUTION: {successful_option}
CONFIDENCE: high"""

    expel_insight_extraction_failure_improved: str = """You are an AI that extracts SPECIFIC insights from failed task executions.

TASK FAILED (exhausted all attempts):
{task}

EXECUTION TRAJECTORY:
- Total attempts made: {attempts}
- ALL OPTIONS THAT FAILED: {failed_options}
- Error messages received: {errors}
- CONDITIONS PRESENT: {conditions}

═══════════════════════════════════════════════════════════════════════════════
YOUR GOAL: Record SPECIFIC options to AVOID for these CONDITIONS
═══════════════════════════════════════════════════════════════════════════════

You MUST record:
1. The EXACT condition codes that were present
2. The EXACT options that FAILED and should be avoided
3. This helps the system avoid repeating mistakes

EXAMPLE GOOD INSIGHTS:
✓ "For conditions C-COLD, H-903, LA-550: AVOID hamburg, rotterdam, los_angeles"
✓ "When R-482 + T-NGHT present, do NOT use: singapore, shanghai"

Respond in this EXACT format:
INSIGHT: For conditions {conditions}, the following options failed: {failed_options}
CONDITIONS_COVERED: {conditions}
AVOID: {failed_options}
CONFIDENCE: high"""


__all__ = ["PromptTemplates"]
