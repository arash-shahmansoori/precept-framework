"""
Memory Consolidation for GemEvo Framework.

Implements the "Compiler" phase that "bakes" frequent memories into system prompts.

Key capabilities:
1. Frequency Analysis: Identify frequently used strategies and lessons
2. Rule Extraction: Convert memories into abstract rules
3. Prompt Mutation: Add new rules to system prompts
4. Memory Pruning: Archive consolidated memories

This bridges the gap between Evo-Memory (runtime) and COMPASS (compilation).
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .memory_store import Experience, MemoryStore
from .llm_clients import precept_llm_client


class ConsolidationType(Enum):
    """Types of consolidation rules."""
    
    STRATEGY = "strategy"  # General approach/strategy
    CONSTRAINT = "constraint"  # Hard constraint/rule
    PREFERENCE = "preference"  # Soft preference
    PROCEDURE = "procedure"  # Step-by-step procedure
    WARNING = "warning"  # Things to avoid


@dataclass
class ConsolidatedRule:
    """A rule extracted from frequent memories."""
    
    id: str
    rule_type: ConsolidationType
    rule_text: str
    source_memory_ids: List[str]
    confidence: float
    occurrence_count: int
    domain: str
    created_at: float = field(default_factory=time.time)
    
    def to_prompt_instruction(self) -> str:
        """Convert rule to prompt instruction format."""
        prefix_map = {
            ConsolidationType.STRATEGY: "STRATEGY",
            ConsolidationType.CONSTRAINT: "CRITICAL RULE",
            ConsolidationType.PREFERENCE: "PREFERENCE",
            ConsolidationType.PROCEDURE: "PROCEDURE",
            ConsolidationType.WARNING: "WARNING",
        }
        prefix = prefix_map.get(self.rule_type, "INSTRUCTION")
        return f"[{prefix}]: {self.rule_text}"


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    
    new_rules: List[ConsolidatedRule]
    updated_rules: List[ConsolidatedRule]
    pruned_memory_ids: List[str]
    prompt_additions: List[str]
    stats: Dict[str, Any]


class FrequencyAnalysis(BaseModel):
    """Analysis of memory frequency patterns (internal use, not for LLM structured output)."""
    
    strategy_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Frequent strategy patterns")
    lesson_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Frequent lessons")
    skill_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Frequently demonstrated skills")
    domain_distribution: Dict[str, int] = Field(default_factory=dict, description="Memory counts per domain")
    consolidation_candidates: List[str] = Field(default_factory=list, description="Items ready for consolidation")
    
    model_config = {"extra": "allow"}  # Allow extra fields for flexibility


class RuleExtraction(BaseModel):
    """Extracted rules from memory patterns."""
    
    rules: List[str] = Field(description="List of extracted rule texts")
    rule_types: List[str] = Field(description="Types for each rule (strategy/constraint/warning/procedure)")
    confidence_scores: List[float] = Field(description="Confidence scores (0-1) for each rule")
    rationales: List[str] = Field(description="Rationale explaining each rule")


class PromptMutationSuggestion(BaseModel):
    """Suggestion for mutating a prompt with new rules."""
    
    original_section: str = Field(default="", description="Original prompt section")
    mutated_section: str = Field(default="", description="Mutated section with new rules")
    rules_added: List[str] = Field(default_factory=list, description="Rules added to this section")
    insertion_point: str = Field(default="end", description="Where to insert the rules")
    rationale: str = Field(default="", description="Why these rules fit here")


class FrequencyAnalyzer:
    """
    Analyzes memory store for consolidation candidates.
    
    Identifies patterns that appear frequently enough to be "baked in".
    """
    
    def __init__(
        self,
        min_strategy_count: int = 5,
        min_lesson_count: int = 3,
        min_success_rate: float = 0.7,
    ):
        self.min_strategy_count = min_strategy_count
        self.min_lesson_count = min_lesson_count
        self.min_success_rate = min_success_rate
    
    def analyze(self, memory_store: MemoryStore) -> FrequencyAnalysis:
        """
        Analyze memory store for consolidation candidates.
        """
        # Get frequent strategies
        strategy_patterns = self._analyze_strategies(memory_store)
        
        # Get frequent lessons
        lesson_patterns = self._analyze_lessons(memory_store)
        
        # Get frequent skills
        skill_patterns = self._analyze_skills(memory_store)
        
        # Get domain distribution
        domain_distribution = self._get_domain_distribution(memory_store)
        
        # Identify consolidation candidates
        candidates = self._identify_candidates(
            strategy_patterns, lesson_patterns, skill_patterns
        )
        
        return FrequencyAnalysis(
            strategy_patterns=strategy_patterns,
            lesson_patterns=lesson_patterns,
            skill_patterns=skill_patterns,
            domain_distribution=domain_distribution,
            consolidation_candidates=candidates,
        )
    
    def _analyze_strategies(
        self, memory_store: MemoryStore
    ) -> List[Dict[str, Any]]:
        """Analyze strategy usage patterns."""
        strategies = memory_store.get_frequent_strategies(
            min_count=self.min_strategy_count
        )
        
        patterns = []
        for strategy, count, avg_correctness in strategies:
            if avg_correctness >= self.min_success_rate:
                patterns.append({
                    "strategy": strategy,
                    "count": count,
                    "success_rate": avg_correctness,
                    "ready_for_consolidation": count >= self.min_strategy_count,
                })
        
        return patterns
    
    def _analyze_lessons(
        self, memory_store: MemoryStore
    ) -> List[Dict[str, Any]]:
        """Analyze lesson frequency patterns."""
        lessons = memory_store.get_frequent_lessons(
            min_count=self.min_lesson_count
        )
        
        patterns = []
        for lesson, count in lessons:
            patterns.append({
                "lesson": lesson,
                "count": count,
                "ready_for_consolidation": count >= self.min_lesson_count,
            })
        
        return patterns
    
    def _analyze_skills(
        self, memory_store: MemoryStore
    ) -> List[Dict[str, Any]]:
        """Analyze skill demonstration patterns."""
        skill_counts: Dict[str, Tuple[int, float]] = {}  # skill -> (count, total_correctness)
        
        for exp in memory_store.episodic_memory.experiences:
            for skill in exp.skills_demonstrated:
                if skill not in skill_counts:
                    skill_counts[skill] = (0, 0.0)
                count, total = skill_counts[skill]
                skill_counts[skill] = (count + 1, total + exp.correctness)
        
        patterns = []
        for skill, (count, total) in skill_counts.items():
            avg_correctness = total / count if count > 0 else 0
            patterns.append({
                "skill": skill,
                "count": count,
                "avg_correctness": avg_correctness,
            })
        
        return sorted(patterns, key=lambda x: x["count"], reverse=True)
    
    def _get_domain_distribution(
        self, memory_store: MemoryStore
    ) -> Dict[str, int]:
        """Get distribution of memories across domains."""
        return {
            domain: len(ids)
            for domain, ids in memory_store.episodic_memory.domain_index.items()
        }
    
    def _identify_candidates(
        self,
        strategy_patterns: List[Dict[str, Any]],
        lesson_patterns: List[Dict[str, Any]],
        skill_patterns: List[Dict[str, Any]],
    ) -> List[str]:
        """Identify items ready for consolidation."""
        candidates = []
        
        for pattern in strategy_patterns:
            if pattern.get("ready_for_consolidation"):
                candidates.append(f"strategy:{pattern['strategy']}")
        
        for pattern in lesson_patterns:
            if pattern.get("ready_for_consolidation"):
                candidates.append(f"lesson:{pattern['lesson']}")
        
        return candidates


class MemoryConsolidator:
    """
    Main consolidator that "bakes" memories into prompts.
    
    This is the key component that bridges Evo-Memory and COMPASS.
    """
    
    def __init__(
        self,
        memory_store: MemoryStore,
        llm_client: Optional[Callable] = None,
        frequency_analyzer: Optional[FrequencyAnalyzer] = None,
    ):
        self.memory_store = memory_store
        # Use actual OpenAI LLM client by default - NO MOCKS
        self.llm_client = llm_client or precept_llm_client
        self.frequency_analyzer = frequency_analyzer or FrequencyAnalyzer()
        
        # Consolidated rules storage
        self.consolidated_rules: Dict[str, ConsolidatedRule] = {}
        self.consolidation_history: List[Dict[str, Any]] = []
        
        # System prompts
        self.extraction_system_prompt = self._get_extraction_system_prompt()
        self.mutation_system_prompt = self._get_mutation_system_prompt()
    
    async def consolidate(
        self,
        current_prompts: Dict[str, str],
        domain_filter: Optional[str] = None,
        force_consolidation: bool = False,
    ) -> ConsolidationResult:
        """
        Run the consolidation process.
        
        1. Analyze memory frequency patterns
        2. Extract rules from frequent patterns
        3. Mutate prompts with new rules
        4. Prune consolidated memories
        
        Args:
            current_prompts: Current system prompts to mutate
            domain_filter: Only consolidate memories from this domain
            force_consolidation: Force consolidation even if thresholds not met
        
        Returns:
            ConsolidationResult with new rules and prompt mutations
        """
        # Step 1: Analyze frequency patterns
        analysis = self.frequency_analyzer.analyze(self.memory_store)
        
        if not analysis.consolidation_candidates and not force_consolidation:
            return ConsolidationResult(
                new_rules=[],
                updated_rules=[],
                pruned_memory_ids=[],
                prompt_additions=[],
                stats={"status": "no_candidates", "analysis": analysis.dict()},
            )
        
        # Step 2: Extract rules from patterns
        # If force_consolidation and no candidates, extract from all memories directly
        if force_consolidation and not analysis.consolidation_candidates:
            new_rules = await self._extract_rules_from_all_memories(domain_filter)
        else:
            new_rules = await self._extract_rules(analysis, domain_filter)
        
        # Step 3: Mutate prompts with new rules
        prompt_additions = await self._mutate_prompts(
            current_prompts, new_rules
        )
        
        # Step 4: Identify memories to prune (those now consolidated)
        consolidated_items = set()
        for rule in new_rules:
            consolidated_items.add(rule.rule_text)
        
        pruned_count = self.memory_store.prune_consolidated(consolidated_items)
        
        # Track consolidation
        result = ConsolidationResult(
            new_rules=new_rules,
            updated_rules=[],
            pruned_memory_ids=[],  # Could track specific IDs if needed
            prompt_additions=prompt_additions,
            stats={
                "patterns_analyzed": len(analysis.consolidation_candidates),
                "rules_extracted": len(new_rules),
                "memories_pruned": pruned_count,
                "domains_covered": list(analysis.domain_distribution.keys()),
            },
        )
        
        self.consolidation_history.append({
            "timestamp": time.time(),
            "result": result.stats,
        })
        
        return result
    
    async def _extract_rules(
        self,
        analysis: FrequencyAnalysis,
        domain_filter: Optional[str],
    ) -> List[ConsolidatedRule]:
        """Extract abstract rules from frequency patterns."""
        # Build context for rule extraction
        extraction_prompt = self._create_extraction_prompt(analysis, domain_filter)
        
        try:
            extraction = await self.llm_client(
                system_prompt=self.extraction_system_prompt,
                user_prompt=extraction_prompt,
                response_model=RuleExtraction,
            )
            
            rules = []
            for i, rule_text in enumerate(extraction.rules):
                rule_type = self._parse_rule_type(
                    extraction.rule_types[i] if i < len(extraction.rule_types) else "strategy"
                )
                confidence = (
                    extraction.confidence_scores[i]
                    if i < len(extraction.confidence_scores)
                    else 0.8
                )
                
                rule = ConsolidatedRule(
                    id=self._generate_rule_id(rule_text),
                    rule_type=rule_type,
                    rule_text=rule_text,
                    source_memory_ids=[],  # Could track if needed
                    confidence=confidence,
                    occurrence_count=5,
                    domain=domain_filter or "general",
                )
                rules.append(rule)
                self.consolidated_rules[rule.id] = rule
            
            return rules
        
        except Exception as e:
            print(f"Rule extraction failed: {e}")
            return []
    
    async def _extract_rules_from_all_memories(
        self,
        domain_filter: Optional[str],
    ) -> List[ConsolidatedRule]:
        """
        Extract rules directly from all memories (used when force_consolidation=True).
        
        This bypasses frequency analysis and lets the LLM find patterns directly.
        """
        print("  → Using direct memory extraction (force mode)")
        
        # Get all memories
        all_memories = list(self.memory_store.episodic_memory.experiences)
        
        if domain_filter:
            all_memories = [m for m in all_memories if m.domain == domain_filter]
        
        if not all_memories:
            return []
        
        # Create a summary of all memories for the LLM
        memory_summaries = []
        for mem in all_memories[:20]:  # Limit to avoid token overflow
            summary = f"""
Task: {mem.task_description}
Strategy: {mem.strategy_used}
Outcome: {mem.outcome}
Lessons: {'; '.join(mem.lessons_learned) if mem.lessons_learned else 'None'}
Domain: {mem.domain}
"""
            memory_summaries.append(summary)
        
        extraction_prompt = f"""
MEMORY ANALYSIS FOR CONSOLIDATION
=================================

Analyze the following {len(memory_summaries)} memories and extract generalizable rules:

{''.join(memory_summaries)}

INSTRUCTIONS
============
Based on these experiences, extract abstract, reusable rules that should be "baked into" 
the agent's core instructions. Look for:

1. Common successful strategies that appear across multiple tasks
2. Lessons that could prevent future failures
3. Domain-specific best practices
4. Warnings about things to avoid

Even if patterns aren't exactly repeated, identify generalizable insights that could
improve future performance. Create at least 2-3 rules if any useful patterns exist.
"""
        
        try:
            extraction = await self.llm_client(
                system_prompt=self.extraction_system_prompt,
                user_prompt=extraction_prompt,
                response_model=RuleExtraction,
            )
            
            rules = []
            for i, rule_text in enumerate(extraction.rules):
                rule_type = self._parse_rule_type(
                    extraction.rule_types[i] if i < len(extraction.rule_types) else "strategy"
                )
                confidence = (
                    extraction.confidence_scores[i]
                    if i < len(extraction.confidence_scores)
                    else 0.8
                )
                
                rule = ConsolidatedRule(
                    id=self._generate_rule_id(rule_text),
                    rule_type=rule_type,
                    rule_text=rule_text,
                    source_memory_ids=[m.id for m in all_memories[:5]],
                    confidence=confidence,
                    occurrence_count=len(all_memories),
                    domain=domain_filter or "general",
                )
                rules.append(rule)
                self.consolidated_rules[rule.id] = rule
            
            return rules
        
        except Exception as e:
            print(f"Rule extraction from all memories failed: {e}")
            return []
    
    async def _mutate_prompts(
        self,
        current_prompts: Dict[str, str],
        new_rules: List[ConsolidatedRule],
    ) -> List[str]:
        """
        Mutate prompts to include new consolidated rules.
        
        This is where we "bake" the memories into the system prompts.
        """
        if not new_rules:
            return []
        
        prompt_additions = []
        
        # Group rules by type
        rules_by_type: Dict[ConsolidationType, List[ConsolidatedRule]] = {}
        for rule in new_rules:
            if rule.rule_type not in rules_by_type:
                rules_by_type[rule.rule_type] = []
            rules_by_type[rule.rule_type].append(rule)
        
        # Create mutation suggestions for main system prompt
        if "system" in current_prompts or "main" in current_prompts:
            target_key = "system" if "system" in current_prompts else "main"
            mutation_prompt = self._create_mutation_prompt(
                current_prompts[target_key],
                new_rules,
            )
            
            try:
                suggestion = await self.llm_client(
                    system_prompt=self.mutation_system_prompt,
                    user_prompt=mutation_prompt,
                    response_model=PromptMutationSuggestion,
                )
                
                prompt_additions.extend(suggestion.rules_added)
            except Exception as e:
                print(f"Prompt mutation failed: {e}")
                # Fallback: Generate simple rule additions
                for rule in new_rules:
                    prompt_additions.append(rule.to_prompt_instruction())
        else:
            
            # No system prompt to mutate, just return rule instructions
            for rule in new_rules:
                prompt_additions.append(rule.to_prompt_instruction())
        
        return prompt_additions
    
    def get_all_rules_as_prompt_section(self) -> str:
        """
        Get all consolidated rules formatted as a prompt section.
        
        This can be added to any system prompt to include learned rules.
        """
        if not self.consolidated_rules:
            return ""
        
        lines = [
            "=================================================",
            "CONSOLIDATED INSTRUCTIONS (Learned from Experience)",
            "=================================================",
            "",
        ]
        
        # Group by type
        rules_by_type: Dict[ConsolidationType, List[ConsolidatedRule]] = {}
        for rule in self.consolidated_rules.values():
            if rule.rule_type not in rules_by_type:
                rules_by_type[rule.rule_type] = []
            rules_by_type[rule.rule_type].append(rule)
        
        # Add critical constraints first
        if ConsolidationType.CONSTRAINT in rules_by_type:
            lines.append("CRITICAL RULES:")
            for rule in rules_by_type[ConsolidationType.CONSTRAINT]:
                lines.append(f"  • {rule.rule_text}")
            lines.append("")
        
        # Add warnings
        if ConsolidationType.WARNING in rules_by_type:
            lines.append("WARNINGS (Things to Avoid):")
            for rule in rules_by_type[ConsolidationType.WARNING]:
                lines.append(f"  ⚠ {rule.rule_text}")
            lines.append("")
        
        # Add strategies
        if ConsolidationType.STRATEGY in rules_by_type:
            lines.append("PROVEN STRATEGIES:")
            for rule in rules_by_type[ConsolidationType.STRATEGY]:
                lines.append(f"  → {rule.rule_text}")
            lines.append("")
        
        # Add procedures
        if ConsolidationType.PROCEDURE in rules_by_type:
            lines.append("STANDARD PROCEDURES:")
            for rule in rules_by_type[ConsolidationType.PROCEDURE]:
                lines.append(f"  1. {rule.rule_text}")
            lines.append("")
        
        # Add preferences
        if ConsolidationType.PREFERENCE in rules_by_type:
            lines.append("PREFERENCES:")
            for rule in rules_by_type[ConsolidationType.PREFERENCE]:
                lines.append(f"  - {rule.rule_text}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_extraction_prompt(
        self,
        analysis: FrequencyAnalysis,
        domain_filter: Optional[str],
    ) -> str:
        """Create prompt for rule extraction."""
        return f"""
MEMORY FREQUENCY ANALYSIS
=========================

**Strategy Patterns (Most Frequent):**
{self._format_patterns(analysis.strategy_patterns[:10])}

**Lesson Patterns (Most Frequent):**
{self._format_patterns(analysis.lesson_patterns[:10])}

**Skill Patterns (Most Demonstrated):**
{self._format_patterns(analysis.skill_patterns[:10])}

**Domain Distribution:**
{analysis.domain_distribution}

**Consolidation Candidates:**
{analysis.consolidation_candidates}

{f"**Domain Filter:** Only consider patterns from domain: {domain_filter}" if domain_filter else ""}

INSTRUCTIONS
============
Extract abstract, generalizable rules from these patterns:

1. For each frequent strategy with high success rate, create a STRATEGY rule
2. For each recurring lesson, create a CONSTRAINT or WARNING rule
3. For demonstrated skills, create PROCEDURE rules if applicable
4. Focus on rules that are:
   - Generalizable across similar tasks
   - Actionable and specific
   - Not too obvious or trivial

Output rules that can be "baked into" system prompts to improve future performance.
"""
    
    def _create_mutation_prompt(
        self,
        current_prompt: str,
        new_rules: List[ConsolidatedRule],
    ) -> str:
        """Create prompt for suggesting prompt mutations."""
        rules_text = "\n".join([
            f"- [{rule.rule_type.value.upper()}] {rule.rule_text} (confidence: {rule.confidence:.2f})"
            for rule in new_rules
        ])
        
        return f"""
CURRENT SYSTEM PROMPT
=====================
{current_prompt}

NEW RULES TO INTEGRATE
======================
{rules_text}

INSTRUCTIONS
============
Suggest how to integrate these learned rules into the system prompt:

1. Identify the best insertion point for each rule
2. Ensure rules don't conflict with existing instructions
3. Maintain the prompt's coherence and flow
4. Group related rules together
5. Use appropriate formatting (bullets, headers, etc.)

The goal is to "bake in" these learned behaviors so the agent follows them instinctively.
"""
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for prompt."""
        if not patterns:
            return "None identified"
        
        lines = []
        for p in patterns:
            parts = []
            for key, value in p.items():
                if key not in ["ready_for_consolidation"]:
                    parts.append(f"{key}: {value}")
            lines.append(f"  • {', '.join(parts)}")
        return "\n".join(lines)
    
    def _parse_rule_type(self, type_str: str) -> ConsolidationType:
        """Parse rule type from string."""
        type_map = {
            "strategy": ConsolidationType.STRATEGY,
            "constraint": ConsolidationType.CONSTRAINT,
            "preference": ConsolidationType.PREFERENCE,
            "procedure": ConsolidationType.PROCEDURE,
            "warning": ConsolidationType.WARNING,
        }
        return type_map.get(type_str.lower(), ConsolidationType.STRATEGY)
    
    def _generate_rule_id(self, rule_text: str) -> str:
        """Generate unique ID for a rule."""
        return hashlib.md5(rule_text.encode()).hexdigest()[:10]
    
    def _get_extraction_system_prompt(self) -> str:
        """System prompt for rule extraction."""
        return """You are an expert at extracting generalizable rules from patterns of behavior.

Your task is to analyze memory frequency patterns and extract abstract rules that can be "baked into" system prompts.

Focus on:
- Rules that appear frequently across multiple experiences
- Rules with high success correlation
- Rules that are specific enough to be actionable
- Rules that generalize well to similar situations

Avoid:
- Overly specific rules tied to single instances
- Obvious or trivial rules
- Rules that contradict each other

Be concise and precise. Each rule should be a clear, actionable instruction."""
    
    def _get_mutation_system_prompt(self) -> str:
        """System prompt for prompt mutation."""
        return """You are an expert prompt engineer specializing in system prompt optimization.

Your task is to integrate newly learned rules into existing system prompts.

Key principles:
- Preserve the original prompt's structure and intent
- Insert rules at logical locations
- Ensure new rules don't conflict with existing instructions
- Use clear, consistent formatting
- Group related rules together
- Make rules feel like natural parts of the prompt

The goal is seamless integration where learned behaviors become instinctive instructions."""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        return {
            "total_rules": len(self.consolidated_rules),
            "rules_by_type": {
                t.value: sum(1 for r in self.consolidated_rules.values() if r.rule_type == t)
                for t in ConsolidationType
            },
            "consolidation_runs": len(self.consolidation_history),
            "domains_covered": list(set(
                r.domain for r in self.consolidated_rules.values()
            )),
        }

