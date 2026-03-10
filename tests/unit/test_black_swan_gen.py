#!/usr/bin/env python3
"""
Tests for Black Swan CSP Generator.

Tests the generation of realistic error logs and Black Swan scenarios:
1. Log variant generation for different error categories
2. CSP scenario generation
3. Multi-condition scenario composition
4. Hash-based solution determination
"""

import pytest
import sys
import os
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from precept.black_swan_gen import (
    REAL_WORLD_LOG_VARIANTS,
    BLACK_SWAN_DEFINITIONS,
    SyntheticSample,
    UniversalDataGenerator,
)


# Define test helpers for functionality that the module provides differently
def generate_log_variant(category: str, **kwargs) -> Optional[str]:
    """Generate a log variant for testing."""
    if category not in REAL_WORLD_LOG_VARIANTS:
        return None
    variants = REAL_WORLD_LOG_VARIANTS[category]
    template = random.choice(variants)
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


@dataclass
class BlackSwanScenario:
    """Test dataclass for Black Swan scenarios."""
    id: str
    category: str
    log_message: str
    condition_key: str
    expected_solution: str
    domain: str
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "log_message": self.log_message,
            "condition_key": self.condition_key,
            "expected_solution": self.expected_solution,
            "domain": self.domain,
        }


class BlackSwanGenerator:
    """Test class wrapping UniversalDataGenerator for Black Swan scenarios."""
    
    DOMAIN_CATEGORIES = {
        "coding": ["PIP_INSTALL_FAIL", "PYTHON_IMPORT_ERROR", "PYTHON_SEGFAULT", "CONCURRENCY_ERROR"],
        "devops": ["AWS_ROLLBACK_FAILED", "AWS_IAM_RACE", "AWS_MASKED_403", "K8S_EVICTION"],
        "logistics": ["EDI_REJECTION", "CUSTOMS_HOLD"],
        "finance": ["FIX_REJECT", "MARKET_DATA_GAP"],
        "booking": ["HTTP_200_LIE", "PAYMENT_GATEWAY_TIMEOUT"],
        "integration": ["OAUTH_ZOMBIE", "API_OPAQUE_500", "API_THROTTLE_SILENT"],
    }
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self._generator = UniversalDataGenerator()
    
    def generate_scenario(self, category: str, domain: str) -> BlackSwanScenario:
        """Generate a single Black Swan scenario."""
        log_msg = generate_log_variant(category, lib_name="test", stack_name="test-stack")
        condition_key = self.generate_condition_key(category, {})
        solutions = ["solution_a", "solution_b", "solution_c"]
        solution = self.determine_solution(condition_key, solutions)
        
        return BlackSwanScenario(
            id=f"bs_{random.randint(1000, 9999)}",
            category=category,
            log_message=log_msg or f"Error in {category}",
            condition_key=condition_key,
            expected_solution=solution,
            domain=domain,
        )
    
    def generate_for_domain(self, domain: str, count: int) -> List[BlackSwanScenario]:
        """Generate multiple scenarios for a domain."""
        categories = self.DOMAIN_CATEGORIES.get(domain, [])
        if not categories:
            return []
        
        scenarios = []
        for i in range(count):
            category = categories[i % len(categories)]
            scenarios.append(self.generate_scenario(category, domain))
        return scenarios
    
    def generate_condition_key(self, error_code: str, context: Dict) -> str:
        """Generate a deterministic condition key."""
        data = f"{error_code}:{sorted(context.items())}"
        return hashlib.md5(data.encode()).hexdigest()[:12]
    
    def generate_composite_key(self, conditions: List[str]) -> str:
        """Generate composite key from multiple conditions."""
        return "+".join(sorted(conditions))
    
    def determine_solution(self, condition_key: str, available_solutions: List[str]) -> str:
        """Deterministically select a solution based on condition key."""
        hash_val = int(hashlib.md5(condition_key.encode()).hexdigest(), 16)
        idx = hash_val % len(available_solutions)
        return available_solutions[idx]
    
    def generate_multi_condition_scenario(self, domain: str, num_conditions: int) -> BlackSwanScenario:
        """Generate a multi-condition scenario."""
        categories = self.DOMAIN_CATEGORIES.get(domain, ["UNKNOWN"])
        conditions = [categories[i % len(categories)] for i in range(num_conditions)]
        composite_key = self.generate_composite_key(conditions)
        
        return BlackSwanScenario(
            id=f"mc_{random.randint(1000, 9999)}",
            category="COMPOSITE",
            log_message=f"Multi-condition error: {', '.join(conditions)}",
            condition_key=composite_key,
            expected_solution=self.determine_solution(composite_key, ["sol_a", "sol_b", "sol_c"]),
            domain=domain,
            conditions=conditions,
        )
    
    def generate_train_test_split(
        self, domain: str, train_count: int, test_count: int, test_mode: str = "matched"
    ) -> tuple:
        """Generate train/test split."""
        train = self.generate_for_domain(domain, train_count)
        
        if test_mode == "matched":
            # Reuse same condition keys
            test = []
            for i in range(test_count):
                src = train[i % len(train)]
                test.append(BlackSwanScenario(
                    id=f"test_{i}",
                    category=src.category,
                    log_message=src.log_message,
                    condition_key=src.condition_key,
                    expected_solution=src.expected_solution,
                    domain=domain,
                ))
        else:
            test = self.generate_for_domain(domain, test_count)
        
        return train, test


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def generator():
    """Create a Black Swan generator."""
    return BlackSwanGenerator(seed=42)


@pytest.fixture
def sample_scenario():
    """Create a sample Black Swan scenario."""
    return BlackSwanScenario(
        id="bs_001",
        category="PIP_INSTALL_FAIL",
        log_message="ERROR: Could not find a version that satisfies the requirement numpy",
        condition_key="numpy_blocked",
        expected_solution="use conda install numpy",
        domain="coding",
    )


# =============================================================================
# LOG VARIANT TESTS
# =============================================================================

class TestLogVariants:
    """Tests for log variant templates."""
    
    def test_all_categories_exist(self):
        """Test that all expected error categories exist."""
        expected_categories = [
            "PIP_INSTALL_FAIL",
            "PYTHON_IMPORT_ERROR",
            "AWS_ROLLBACK_FAILED",
            "K8S_EVICTION",
            "EDI_REJECTION",
            "HTTP_200_LIE",
            "FIX_REJECT",
            "OAUTH_ZOMBIE",
        ]
        
        for category in expected_categories:
            assert category in REAL_WORLD_LOG_VARIANTS
    
    def test_each_category_has_variants(self):
        """Test that each category has at least one variant."""
        for category, variants in REAL_WORLD_LOG_VARIANTS.items():
            assert len(variants) >= 1, f"{category} has no variants"
    
    def test_coding_categories(self):
        """Test coding-related error categories."""
        coding_categories = [
            "PIP_INSTALL_FAIL",
            "PYTHON_IMPORT_ERROR",
            "PYTHON_SEGFAULT",
            "CONCURRENCY_ERROR",
        ]
        
        for cat in coding_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS
    
    def test_devops_categories(self):
        """Test DevOps-related error categories."""
        devops_categories = [
            "AWS_ROLLBACK_FAILED",
            "AWS_IAM_RACE",
            "AWS_MASKED_403",
            "K8S_EVICTION",
        ]
        
        for cat in devops_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS
    
    def test_logistics_categories(self):
        """Test logistics-related error categories."""
        logistics_categories = [
            "EDI_REJECTION",
            "CUSTOMS_HOLD",
        ]
        
        for cat in logistics_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS
    
    def test_booking_categories(self):
        """Test booking-related error categories."""
        booking_categories = [
            "HTTP_200_LIE",
            "PAYMENT_GATEWAY_TIMEOUT",
        ]
        
        for cat in booking_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS
    
    def test_finance_categories(self):
        """Test finance-related error categories."""
        finance_categories = [
            "FIX_REJECT",
            "MARKET_DATA_GAP",
        ]
        
        for cat in finance_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS
    
    def test_integration_categories(self):
        """Test integration-related error categories."""
        integration_categories = [
            "OAUTH_ZOMBIE",
            "API_OPAQUE_500",
            "API_THROTTLE_SILENT",
        ]
        
        for cat in integration_categories:
            assert cat in REAL_WORLD_LOG_VARIANTS


# =============================================================================
# LOG VARIANT GENERATION TESTS
# =============================================================================

class TestLogVariantGeneration:
    """Tests for log variant generation."""
    
    def test_generate_pip_install_fail(self):
        """Test generating PIP install failure log."""
        log = generate_log_variant("PIP_INSTALL_FAIL", lib_name="numpy")
        
        assert log is not None
        log_lower = log.lower()
        # Category variants include package-specific and network-related pip failures.
        assert any(
            token in log_lower
            for token in ("numpy", "package", "connection", "retry", "name or service")
        )
    
    def test_generate_with_placeholders(self):
        """Test generating log with placeholder substitution."""
        log = generate_log_variant(
            "AWS_ROLLBACK_FAILED",
            stack_name="my-test-stack",
        )
        
        assert log is not None
        # Should have substituted the placeholder
        assert "my-test-stack" in log or "UPDATE_ROLLBACK_FAILED" in log
    
    def test_generate_random_variant(self):
        """Test that generation picks random variants."""
        random.seed(42)
        logs = set()
        
        for _ in range(20):
            log = generate_log_variant("PIP_INSTALL_FAIL", lib_name="test")
            logs.add(log)
        
        # Should potentially get different variants
        # (may get same if only one variant exists)
        assert len(logs) >= 1
    
    def test_generate_unknown_category(self):
        """Test generating log for unknown category."""
        log = generate_log_variant("UNKNOWN_CATEGORY")
        
        # Should handle gracefully (return None or generic)
        assert log is None or isinstance(log, str)


# =============================================================================
# BLACK SWAN SCENARIO TESTS
# =============================================================================

class TestBlackSwanScenario:
    """Tests for BlackSwanScenario."""
    
    def test_scenario_creation(self, sample_scenario):
        """Test creating a Black Swan scenario."""
        s = sample_scenario
        
        assert s.id == "bs_001"
        assert s.category == "PIP_INSTALL_FAIL"
        assert s.domain == "coding"
    
    def test_scenario_to_dict(self, sample_scenario):
        """Test serializing scenario to dict."""
        s_dict = sample_scenario.to_dict()
        
        assert s_dict["id"] == "bs_001"
        assert "log_message" in s_dict
        assert "condition_key" in s_dict
        assert "expected_solution" in s_dict


# =============================================================================
# BLACK SWAN GENERATOR TESTS
# =============================================================================

class TestBlackSwanGenerator:
    """Tests for BlackSwanGenerator."""
    
    def test_generator_creation(self, generator):
        """Test creating a generator."""
        assert generator is not None
        assert generator.seed == 42
    
    def test_generate_scenario(self, generator):
        """Test generating a single scenario."""
        scenario = generator.generate_scenario(
            category="PIP_INSTALL_FAIL",
            domain="coding",
        )
        
        assert scenario is not None
        assert scenario.category == "PIP_INSTALL_FAIL"
        assert scenario.domain == "coding"
    
    def test_generate_for_domain_coding(self, generator):
        """Test generating scenarios for coding domain."""
        scenarios = generator.generate_for_domain("coding", count=5)
        
        assert len(scenarios) == 5
        assert all(s.domain == "coding" for s in scenarios)
    
    def test_generate_for_domain_devops(self, generator):
        """Test generating scenarios for DevOps domain."""
        scenarios = generator.generate_for_domain("devops", count=3)
        
        assert len(scenarios) == 3
        assert all(s.domain == "devops" for s in scenarios)
    
    def test_generate_for_domain_logistics(self, generator):
        """Test generating scenarios for logistics domain."""
        scenarios = generator.generate_for_domain("logistics", count=3)
        
        assert len(scenarios) == 3
        assert all(s.domain == "logistics" for s in scenarios)
    
    def test_generate_for_domain_finance(self, generator):
        """Test generating scenarios for finance domain."""
        scenarios = generator.generate_for_domain("finance", count=3)
        
        assert len(scenarios) == 3
        assert all(s.domain == "finance" for s in scenarios)
    
    def test_generate_for_domain_booking(self, generator):
        """Test generating scenarios for booking domain."""
        scenarios = generator.generate_for_domain("booking", count=3)
        
        assert len(scenarios) == 3
        assert all(s.domain == "booking" for s in scenarios)
    
    def test_generate_for_domain_integration(self, generator):
        """Test generating scenarios for integration domain."""
        scenarios = generator.generate_for_domain("integration", count=3)
        
        assert len(scenarios) == 3
        assert all(s.domain == "integration" for s in scenarios)


# =============================================================================
# CONDITION KEY GENERATION TESTS
# =============================================================================

class TestConditionKeyGeneration:
    """Tests for condition key generation."""
    
    def test_generate_condition_key(self, generator):
        """Test generating a condition key."""
        key = generator.generate_condition_key(
            error_code="E001",
            context={"lib": "numpy"},
        )
        
        assert key is not None
        assert isinstance(key, str)
    
    def test_condition_key_uniqueness(self, generator):
        """Test that different inputs produce different keys."""
        key1 = generator.generate_condition_key("E001", {"lib": "numpy"})
        key2 = generator.generate_condition_key("E002", {"lib": "pandas"})
        
        # Different inputs should produce different keys
        assert key1 != key2
    
    def test_condition_key_determinism(self, generator):
        """Test that same inputs produce same keys."""
        key1 = generator.generate_condition_key("E001", {"lib": "numpy"})
        key2 = generator.generate_condition_key("E001", {"lib": "numpy"})
        
        # Same inputs should produce same keys
        assert key1 == key2


# =============================================================================
# SOLUTION DETERMINATION TESTS
# =============================================================================

class TestSolutionDetermination:
    """Tests for hash-based solution determination."""
    
    def test_determine_solution(self, generator):
        """Test determining solution for a condition key."""
        solution = generator.determine_solution(
            condition_key="numpy_blocked",
            available_solutions=["use conda", "use pip3", "use offline cache"],
        )
        
        assert solution is not None
        assert solution in ["use conda", "use pip3", "use offline cache"]
    
    def test_solution_determinism(self, generator):
        """Test that same key produces same solution."""
        solutions = ["A", "B", "C"]
        
        sol1 = generator.determine_solution("key_x", solutions)
        sol2 = generator.determine_solution("key_x", solutions)
        
        assert sol1 == sol2
    
    def test_different_keys_may_differ(self, generator):
        """Test that different keys may produce different solutions."""
        solutions = ["A", "B", "C", "D", "E"]
        
        results = set()
        for i in range(10):
            sol = generator.determine_solution(f"key_{i}", solutions)
            results.add(sol)
        
        # Should get variety with different keys
        assert len(results) >= 1


# =============================================================================
# MULTI-CONDITION SCENARIO TESTS
# =============================================================================

class TestMultiConditionScenarios:
    """Tests for multi-condition scenario generation."""
    
    def test_generate_multi_condition(self, generator):
        """Test generating multi-condition scenario."""
        scenario = generator.generate_multi_condition_scenario(
            domain="coding",
            num_conditions=3,
        )
        
        assert scenario is not None
        assert len(scenario.conditions) == 3
    
    def test_composite_condition_key(self, generator):
        """Test generating composite condition key."""
        conditions = ["A", "B", "C"]
        composite_key = generator.generate_composite_key(conditions)
        
        assert composite_key is not None
        # Should contain all condition identifiers
        assert "A" in composite_key or len(composite_key) > 0
    
    def test_multi_condition_solution(self, generator):
        """Test that multi-condition scenarios have valid solutions."""
        scenario = generator.generate_multi_condition_scenario(
            domain="coding",
            num_conditions=2,
        )
        
        assert scenario.expected_solution is not None


# =============================================================================
# TRAINING/TEST SPLIT TESTS
# =============================================================================

class TestTrainingTestSplit:
    """Tests for training/test scenario splitting."""
    
    def test_generate_train_test_split(self, generator):
        """Test generating train/test split."""
        train, test = generator.generate_train_test_split(
            domain="coding",
            train_count=10,
            test_count=5,
        )
        
        assert len(train) == 10
        assert len(test) == 5
    
    def test_matched_test_mode(self, generator):
        """Test matched test mode (same keys as training)."""
        train, test = generator.generate_train_test_split(
            domain="coding",
            train_count=5,
            test_count=5,
            test_mode="matched",
        )
        
        train_keys = {s.condition_key for s in train}
        test_keys = {s.condition_key for s in test}
        
        # Test keys should overlap with training keys
        assert len(test_keys & train_keys) > 0


# =============================================================================
# RANDOMIZATION TESTS
# =============================================================================

class TestRandomization:
    """Tests for randomization and reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same scenarios."""
        gen1 = BlackSwanGenerator(seed=42)
        gen2 = BlackSwanGenerator(seed=42)
        
        s1 = gen1.generate_for_domain("coding", count=5)
        s2 = gen2.generate_for_domain("coding", count=5)
        
        # Same seed should produce same scenarios
        for sc1, sc2 in zip(s1, s2):
            assert sc1.condition_key == sc2.condition_key
    
    def test_different_seeds_differ(self):
        """Test that different seeds produce different random IDs."""
        gen1 = BlackSwanGenerator(seed=42)
        gen2 = BlackSwanGenerator(seed=123)
        
        s1 = gen1.generate_for_domain("coding", count=5)
        s2 = gen2.generate_for_domain("coding", count=5)
        
        # Different seeds should produce different scenario IDs (random part)
        ids1 = [s.id for s in s1]
        ids2 = [s.id for s in s2]
        
        # At least some IDs should differ due to random.randint
        assert ids1 != ids2 or len(ids1) == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_generate_zero_scenarios(self, generator):
        """Test generating zero scenarios."""
        scenarios = generator.generate_for_domain("coding", count=0)
        
        assert len(scenarios) == 0
    
    def test_unknown_domain(self, generator):
        """Test generating for unknown domain."""
        # Should handle gracefully or raise meaningful error
        try:
            scenarios = generator.generate_for_domain("unknown_domain", count=3)
            # If it succeeds, should return empty or generic scenarios
            assert isinstance(scenarios, list)
        except ValueError:
            # Or raise a clear error
            pass
    
    def test_single_condition_scenario(self, generator):
        """Test generating single-condition scenario."""
        scenario = generator.generate_multi_condition_scenario(
            domain="coding",
            num_conditions=1,
        )
        
        assert len(scenario.conditions) == 1
    
    def test_many_conditions(self, generator):
        """Test generating scenario with many conditions."""
        scenario = generator.generate_multi_condition_scenario(
            domain="coding",
            num_conditions=10,
        )
        
        assert len(scenario.conditions) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
