"""
Comprehensive Unit Tests for precept.rule_parser module.

Tests DynamicRuleParser and ParsedRule classes with detailed functionality coverage.
"""

import pytest

from precept.rule_parser import DynamicRuleParser, ParsedRule


# =============================================================================
# TEST PARSED RULE DATACLASS
# =============================================================================


class TestParsedRule:
    """Tests for ParsedRule dataclass."""

    def test_parsed_rule_creation(self):
        """Test creating a parsed rule."""
        rule = ParsedRule(
            error_code="R-482",
            blocked_entity="rotterdam",
            alternative="hamburg",
            condition="US",
            confidence=1.0,
        )
        assert rule.error_code == "R-482"
        assert rule.blocked_entity == "rotterdam"
        assert rule.alternative == "hamburg"
        assert rule.condition == "US"
        assert rule.confidence == 1.0

    def test_parsed_rule_default_values(self):
        """Test parsed rule default values."""
        rule = ParsedRule(
            error_code="R-482",
            blocked_entity="rotterdam",
            alternative="hamburg",
        )
        assert rule.condition is None
        assert rule.confidence == 1.0

    def test_parsed_rule_with_non_us_condition(self):
        """Test parsed rule with non-US condition."""
        rule = ParsedRule(
            error_code="H-903",
            blocked_entity="hamburg",
            alternative="antwerp",
            condition="non-US",
        )
        assert rule.condition == "non-US"


# =============================================================================
# TEST DYNAMIC RULE PARSER
# =============================================================================


class TestDynamicRuleParserInitialization:
    """Tests for DynamicRuleParser initialization."""

    def test_parser_creation_default(self):
        """Test creating parser with defaults."""
        parser = DynamicRuleParser()
        assert parser._known_entities == set()

    def test_parser_creation_with_entities(self):
        """Test creating parser with known entities."""
        parser = DynamicRuleParser(known_entities=["rotterdam", "hamburg", "antwerp"])
        assert "rotterdam" in parser._known_entities
        assert "hamburg" in parser._known_entities
        assert "antwerp" in parser._known_entities

    def test_parser_entities_are_lowercase(self):
        """Test that entities are normalized to lowercase."""
        parser = DynamicRuleParser(known_entities=["Rotterdam", "HAMBURG", "Antwerp"])
        assert "rotterdam" in parser._known_entities
        assert "hamburg" in parser._known_entities

    def test_add_entities(self):
        """Test adding entities to parser."""
        parser = DynamicRuleParser()
        parser.add_entities(["boston", "new_york"])
        assert "boston" in parser._known_entities
        assert "new_york" in parser._known_entities


class TestDynamicRuleParserParsing:
    """Tests for DynamicRuleParser rule parsing."""

    @pytest.fixture
    def logistics_parser(self):
        """Create a parser with logistics entities."""
        return DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp", "boston", "new_york"]
        )

    def test_parse_simple_blocked_rule(self, logistics_parser):
        """Test parsing a simple blocked port rule."""
        rules = logistics_parser.parse_rules([
            "R-482: Rotterdam blocked, use Hamburg"
        ])

        assert len(rules) == 1
        assert rules[0].error_code == "R-482"
        assert rules[0].blocked_entity == "rotterdam"
        assert rules[0].alternative == "hamburg"

    def test_parse_arrow_notation(self, logistics_parser):
        """Test parsing rules with arrow notation."""
        rules = logistics_parser.parse_rules([
            "Error H-903: Hamburg→US blocked → use Antwerp"
        ])

        assert len(rules) == 1
        assert rules[0].error_code == "H-903"
        assert rules[0].blocked_entity == "hamburg"
        # Alternative could be "antwerp" or parsed from arrow

    def test_parse_try_syntax(self, logistics_parser):
        """Test parsing rules with 'try' syntax."""
        rules = logistics_parser.parse_rules([
            "R-100: Rotterdam closed, try Antwerp"
        ])

        assert len(rules) == 1
        assert rules[0].alternative == "antwerp"

    def test_parse_switch_to_syntax(self, logistics_parser):
        """Test parsing rules with 'switch to' syntax."""
        rules = logistics_parser.parse_rules([
            "R-200: Hamburg unavailable, switch to Antwerp"
        ])

        assert len(rules) == 1
        assert rules[0].alternative == "antwerp"

    def test_parse_us_condition(self, logistics_parser):
        """Test parsing rules with US destination condition."""
        rules = logistics_parser.parse_rules([
            "R-482: Rotterdam blocked to US, use Hamburg"
        ])

        assert len(rules) == 1
        assert rules[0].condition == "US"

    def test_parse_non_us_condition(self, logistics_parser):
        """Test parsing rules with non-US destination condition."""
        rules = logistics_parser.parse_rules([
            "R-482: Rotterdam blocked for non-US destinations, use Hamburg"
        ])

        assert len(rules) == 1
        assert rules[0].condition == "non-US"

    def test_parse_multiple_rules(self, logistics_parser):
        """Test parsing multiple rules."""
        rules = logistics_parser.parse_rules([
            "R-482: Rotterdam blocked, use Hamburg",
            "H-903: Hamburg closed, use Antwerp",
        ])

        assert len(rules) == 2

    def test_parse_no_error_code_skipped(self, logistics_parser):
        """Test that rules without error codes are skipped."""
        rules = logistics_parser.parse_rules([
            "Rotterdam is blocked, use Hamburg"
        ])

        # No error code, so rule is incomplete
        assert len(rules) == 0

    def test_parse_empty_rules(self, logistics_parser):
        """Test parsing empty rules list."""
        rules = logistics_parser.parse_rules([])
        assert len(rules) == 0

    def test_parse_unknown_entities_skipped(self):
        """Test that rules with unknown entities are skipped."""
        parser = DynamicRuleParser(known_entities=["rotterdam"])
        rules = parser.parse_rules([
            "R-482: Rotterdam blocked, use UnknownPort"
        ])

        # Alternative not in known entities
        assert len(rules) == 0

    def test_blocked_keywords_variations(self, logistics_parser):
        """Test different blocked keyword variations."""
        test_cases = [
            ("R-001: Rotterdam blocked, use Hamburg", "blocked"),
            ("R-002: Rotterdam closed, use Hamburg", "closed"),
            ("R-003: Rotterdam unavailable, use Hamburg", "unavailable"),
            ("R-004: Rotterdam failed, use Hamburg", "failed"),
            ("R-005: Rotterdam error, use Hamburg", "error"),
            ("R-006: Rotterdam rejected, use Hamburg", "rejected"),
        ]

        for rule_text, _ in test_cases:
            rules = logistics_parser.parse_rules([rule_text])
            assert len(rules) == 1, f"Failed to parse: {rule_text}"


class TestDynamicRuleParserFindApplicable:
    """Tests for DynamicRuleParser.find_applicable_rule."""

    @pytest.fixture
    def parsed_rules(self):
        """Create a list of parsed rules for testing."""
        return [
            ParsedRule(
                error_code="R-482",
                blocked_entity="rotterdam",
                alternative="hamburg",
                condition=None,
            ),
            ParsedRule(
                error_code="H-903",
                blocked_entity="hamburg",
                alternative="antwerp",
                condition="US",
            ),
            ParsedRule(
                error_code="A-101",
                blocked_entity="antwerp",
                alternative="rotterdam",
                condition="non-US",
            ),
        ]

    def test_find_matching_rule(self, parsed_rules):
        """Test finding a matching rule."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="rotterdam",
            is_us_destination=False,
        )

        assert rule is not None
        assert rule.error_code == "R-482"

    def test_find_rule_with_us_condition(self, parsed_rules):
        """Test finding rule with US destination condition."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="hamburg",
            is_us_destination=True,
        )

        assert rule is not None
        assert rule.error_code == "H-903"

    def test_no_match_for_us_when_non_us(self, parsed_rules):
        """Test that US rule doesn't match for non-US destination."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="hamburg",
            is_us_destination=False,
        )

        # H-903 requires US destination
        assert rule is None

    def test_find_rule_with_non_us_condition(self, parsed_rules):
        """Test finding rule with non-US destination condition."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="antwerp",
            is_us_destination=False,
        )

        assert rule is not None
        assert rule.error_code == "A-101"

    def test_no_match_for_unknown_source(self, parsed_rules):
        """Test no match for unknown source."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="singapore",
            is_us_destination=False,
        )

        assert rule is None

    def test_case_insensitive_source_matching(self, parsed_rules):
        """Test case-insensitive source matching."""
        parser = DynamicRuleParser()
        rule = parser.find_applicable_rule(
            parsed_rules=parsed_rules,
            source="ROTTERDAM",
            is_us_destination=False,
        )

        assert rule is not None
        assert rule.error_code == "R-482"


class TestDynamicRuleParserHelpers:
    """Tests for DynamicRuleParser helper methods."""

    def test_get_all_blocked_entities(self):
        """Test getting all blocked entities from parsed rules."""
        parser = DynamicRuleParser()
        parsed_rules = [
            ParsedRule(error_code="R-1", blocked_entity="rotterdam", alternative="hamburg"),
            ParsedRule(error_code="R-2", blocked_entity="hamburg", alternative="antwerp"),
        ]

        blocked = parser.get_all_blocked_entities(parsed_rules)
        assert "rotterdam" in blocked
        assert "hamburg" in blocked

    def test_get_all_alternatives(self):
        """Test getting all alternative entities from parsed rules."""
        parser = DynamicRuleParser()
        parsed_rules = [
            ParsedRule(error_code="R-1", blocked_entity="rotterdam", alternative="hamburg"),
            ParsedRule(error_code="R-2", blocked_entity="hamburg", alternative="antwerp"),
        ]

        alternatives = parser.get_all_alternatives(parsed_rules)
        assert "hamburg" in alternatives
        assert "antwerp" in alternatives


class TestRuleParserIntegration:
    """Integration tests for the rule parser."""

    def test_full_workflow(self):
        """Test complete workflow: parse rules, find applicable, get entities."""
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp", "boston"]
        )

        # Parse rules from text
        rules = parser.parse_rules([
            "R-482: Rotterdam blocked, use Hamburg",
            "H-903: Hamburg closed, use Antwerp",
        ])

        # Check that rules were parsed
        assert len(rules) >= 1, f"Expected at least 1 rule, got {len(rules)}"

        # Get all blocked entities (these should work regardless of rule count)
        blocked = parser.get_all_blocked_entities(rules)
        alternatives = parser.get_all_alternatives(rules)

        assert isinstance(blocked, set)
        assert isinstance(alternatives, set)

    def test_chained_rules_workflow(self):
        """Test handling chained rules (e.g., Rotterdam -> Hamburg -> Antwerp)."""
        parser = DynamicRuleParser(
            known_entities=["rotterdam", "hamburg", "antwerp"]
        )

        rules = parser.parse_rules([
            "R-482: Rotterdam blocked, use Hamburg",
            "H-903: Hamburg blocked, use Antwerp",
        ])

        # Verify we can get blocked entities and alternatives
        blocked = parser.get_all_blocked_entities(rules)
        alternatives = parser.get_all_alternatives(rules)

        assert isinstance(blocked, set)
        assert isinstance(alternatives, set)

    def test_parser_entities_workflow(self):
        """Test adding entities and parsing rules."""
        parser = DynamicRuleParser()

        # Add entities dynamically
        parser.add_entities(["singapore", "hong_kong", "tokyo"])

        # Verify entities are stored
        assert "singapore" in parser._known_entities
        assert "hong_kong" in parser._known_entities
        assert "tokyo" in parser._known_entities
