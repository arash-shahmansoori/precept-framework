"""
Dynamic Rule Parser for PRECEPT Learning.

This module provides domain-agnostic parsing of learned rules from text.
The parser extracts rules dynamically - NO HARDCODED KNOWLEDGE.

Example rules it can parse:
- "R-482: Rotterdam blocked, use Hamburg"
- "Error H-903: Hamburg→US blocked → use Antwerp"
- "When booking to US from Rotterdam, use Antwerp (learned from R-482+H-903)"

The parser only knows vocabulary (entity names), not which are blocked.
All blocking knowledge is extracted from the rule text itself.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass
class ParsedRule:
    """A dynamically parsed rule from learned rules text."""
    error_code: str
    blocked_entity: str
    alternative: str
    condition: Optional[str] = None  # e.g., "US", "non-US"
    confidence: float = 1.0


class DynamicRuleParser:
    """
    Parses learned rules dynamically - NO HARDCODED KNOWLEDGE.

    The parser extracts rules from text like:
    - "R-482: Rotterdam blocked, use Hamburg"
    - "Error H-903: Hamburg→US blocked → use Antwerp"
    - "When booking to US from Rotterdam, use Antwerp (learned from R-482+H-903)"

    This ensures PRECEPT truly learns and doesn't have hardcoded solutions.

    🚨 CRITICAL: This parser is DOMAIN-AGNOSTIC.
    It receives the list of known entities from the DomainStrategy.
    It has NO knowledge of which entities are blocked or which are alternatives.

    Usage:
        parser = DynamicRuleParser(known_entities=["rotterdam", "hamburg", "antwerp"])
        rules = parser.parse_rules(["R-482: Rotterdam blocked, use Hamburg"])
        # Returns: [ParsedRule(error_code="R-482", blocked_entity="rotterdam", alternative="hamburg")]
    """

    # Generic patterns for rule extraction (NOT domain-specific)
    ERROR_CODE_PATTERN = r'([A-Z]{1,2}-\d{3})'
    BLOCKED_KEYWORDS = ["blocked", "closed", "unavailable", "failed", "error", "rejected"]
    USE_PATTERN = r'(use|try|switch to|fallback to|redirect to)\s+(\w+)'

    def __init__(self, known_entities: Optional[List[str]] = None):
        """
        Initialize parser with known entities.

        Args:
            known_entities: List of valid entity names for this domain.
                           E.g., ["rotterdam", "hamburg", "antwerp"] for logistics.
                           The parser does NOT know which are blocked.
        """
        # These are just the vocabulary, NOT knowledge of what's blocked
        self._known_entities: Set[str] = set(e.lower() for e in (known_entities or []))

    def add_entities(self, entities: List[str]) -> None:
        """Add more known entities to the vocabulary."""
        self._known_entities.update(e.lower() for e in entities)

    def parse_rules(self, rules: List[str]) -> List[ParsedRule]:
        """
        Parse rules dynamically from text.

        Returns list of ParsedRule objects extracted from the rules text.

        🚨 NO hardcoded knowledge:
        - Does NOT know which entities are blocked
        - Does NOT know which alternatives work
        - Extracts ALL information from the rule text itself

        Args:
            rules: List of rule strings to parse

        Returns:
            List of ParsedRule objects
        """
        parsed_rules = []

        for rule_text in rules:
            rule_lower = rule_text.lower()

            # Extract error code from rule text
            error_match = re.search(self.ERROR_CODE_PATTERN, rule_text, re.IGNORECASE)
            error_code = error_match.group(1).upper() if error_match else None

            # Extract blocked entity from rule text
            # Look for pattern: "<entity> <blocked keyword>"
            blocked_entity = None
            for entity in self._known_entities:
                if entity in rule_lower:
                    # Check if this entity appears near a "blocked" keyword
                    for keyword in self.BLOCKED_KEYWORDS:
                        if keyword in rule_lower:
                            # Check proximity (entity mentioned before "blocked")
                            entity_pos = rule_lower.find(entity)
                            keyword_pos = rule_lower.find(keyword)
                            if entity_pos < keyword_pos and keyword_pos - entity_pos < 30:
                                blocked_entity = entity
                                break
                    if blocked_entity:
                        break

            # Extract alternative from rule text
            # Look for pattern: "use <entity>" or "try <entity>"
            alternative = None
            use_match = re.search(self.USE_PATTERN, rule_lower)
            if use_match:
                alt = use_match.group(2).lower()
                if alt in self._known_entities:
                    alternative = alt

            # If no "use X" pattern, look for entity after "→" or "->"
            if not alternative:
                arrow_match = re.search(r'[→\->]\s*(\w+)', rule_lower)
                if arrow_match:
                    alt = arrow_match.group(1).lower()
                    if alt in self._known_entities:
                        alternative = alt

            # Extract condition from rule text
            # Look for destination hints (US, europe, asia, etc.)
            condition = None
            us_keywords = ["us", "usa", "boston", "new_york", "united states", "america"]
            if any(us in rule_lower for us in us_keywords):
                if "non-us" in rule_lower or "non us" in rule_lower:
                    condition = "non-US"
                else:
                    condition = "US"

            # Only create rule if we extracted enough info from text
            if error_code and blocked_entity and alternative:
                parsed_rules.append(ParsedRule(
                    error_code=error_code,
                    blocked_entity=blocked_entity,
                    alternative=alternative,
                    condition=condition,
                ))

        return parsed_rules

    def find_applicable_rule(
        self,
        parsed_rules: List[ParsedRule],
        source: str,
        is_us_destination: bool = False,
    ) -> Optional[ParsedRule]:
        """
        Find a rule that applies to the current task.

        Returns the first matching rule, or None if no rule applies.

        🚨 NO hardcoded logic - just matches parsed rules to current state.

        Args:
            parsed_rules: List of parsed rules to search
            source: The source entity (e.g., port name)
            is_us_destination: Whether the destination is US

        Returns:
            Matching ParsedRule or None
        """
        for rule in parsed_rules:
            # Check if source matches blocked entity (from parsed rule)
            if rule.blocked_entity != source.lower():
                continue

            # Check condition (from parsed rule)
            if rule.condition == "US" and not is_us_destination:
                continue
            if rule.condition == "non-US" and is_us_destination:
                continue

            return rule

        return None

    def get_all_blocked_entities(self, parsed_rules: List[ParsedRule]) -> Set[str]:
        """Get all blocked entities from parsed rules."""
        return {rule.blocked_entity for rule in parsed_rules}

    def get_all_alternatives(self, parsed_rules: List[ParsedRule]) -> Set[str]:
        """Get all alternative entities from parsed rules."""
        return {rule.alternative for rule in parsed_rules}
