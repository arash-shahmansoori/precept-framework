"""
Integration Scenario Generator for PRECEPT Testing.

This module generates API/integration black swan scenarios using template-based variations.
Supports configurable num_samples and train_ratio for flexible train/test splits.

Configuration is imported from precept.config.integration - single source of truth.

Usage:
    from precept.scenario_generators import IntegrationScenarioGenerator

    generator = IntegrationScenarioGenerator(num_samples=20, train_ratio=0.6)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import IntegrationConfig
from ..config.multi_condition import (
    IntegrationConditions,
    MultiConditionConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.integration
IntegrationScenarioConfig = IntegrationConfig


class IntegrationScenarioGenerator:
    """
    Generate API/integration black swan scenarios using template-based variations.

    COHERENCE GUARANTEE: Each scenario maintains semantic consistency:
    - OAuth sources use correct error codes and recovery actions
    - Gateway endpoints have appropriate failure modes
    - Webhook issues have proper solutions

    Usage:
        generator = IntegrationScenarioGenerator(num_samples=20, train_ratio=0.6)
        scenarios = generator.generate_all()
    """

    def __init__(self, num_samples: int = 20, train_ratio: float = 0.6):
        """
        Initialize the generator.

        Args:
            num_samples: TOTAL number of scenarios (train + test combined)
            train_ratio: Ratio of training samples (0.0 to 1.0)
        """
        self.num_samples = num_samples
        self.train_ratio = max(0.1, min(0.9, train_ratio))
        self.generator = UniversalDataGenerator(num_samples=num_samples)
        self.integration_traps = BLACK_SWAN_DEFINITIONS.get("Integration", {})
        self.config = IntegrationScenarioConfig

    def _build_scenario(
        self,
        task: str,
        expected: str,
        black_swan_type: str,
        precept_lesson: str,
        phase: str = None,
        tests_learning: str = None,
    ) -> Dict:
        """Build a scenario dictionary with consistent structure."""
        scenario = {
            "task": task,
            "expected": expected,
            "black_swan_type": black_swan_type,
            "precept_lesson": precept_lesson,
        }
        if phase:
            scenario["phase"] = phase
        if tests_learning:
            scenario["tests_learning"] = tests_learning
        return scenario

    def generate_from_universal_generator(self, num_samples: int = 5) -> List[Dict]:
        """Generate scenarios using UniversalDataGenerator."""
        scenarios = []

        for trap_name, trap_def in self.integration_traps.items():
            sample = self.generator.generate_sample(
                category="Integration",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"Integration/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios[:num_samples]

    def generate_oauth_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate OAuth/authentication scenarios with template-based variations.

        COHERENCE GUARANTEE: Each source always uses:
        - Its own error_code (OAUTH-ZOMBIE-401, THROTTLE-429, etc.)
        - Its own recovery_action (re-authenticate, exponential-backoff, etc.)
        - Its own lesson text
        """
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training
        num_training = (
            num_training if num_training is not None else max(2, total_training // 2)
        )
        num_test = num_test if num_test is not None else max(1, total_test // 2)

        # Build all possible COHERENT combinations
        all_training_combos = []
        all_test_combos = []

        for source, source_info in self.config.OAUTH_SOURCES.items():
            for template in self.config.OAUTH_TRAINING_TEMPLATES:
                all_training_combos.append((source, source_info, template))

            for data_op in self.config.DATA_OPERATIONS:
                for template in self.config.OAUTH_TEST_TEMPLATES:
                    all_test_combos.append((source, source_info, data_op, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for source, source_info, template in sampled_training:
            task = template.format(
                source=source.replace("_", " ").title(),
                data_type=source_info["data_type"],
            )
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{source_info['error_code']} → {source_info['failure_reason']}",
                    black_swan_type="Integration/Auth_Zombie",
                    precept_lesson=source_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for source, source_info, data_op, template in sampled_test:
            task = template.format(
                source=source.replace("_", " ").title(),
                data_op=data_op,
            )
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {source_info['recovery_action']} (1 step)",
                    black_swan_type="Integration/Auth_Zombie",
                    precept_lesson=f"PRECEPT applies {source_info['recovery_action']} (learned)",
                    phase="test",
                    tests_learning=source,
                )
            )

        return training + test_variations

    def generate_gateway_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate gateway/proxy scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Build combinations
        all_training = []
        all_test = []

        for endpoint, endpoint_info in self.config.GATEWAY_ENDPOINTS.items():
            for template in self.config.GATEWAY_TRAINING_TEMPLATES:
                all_training.append((endpoint, endpoint_info, template))
            for template in self.config.GATEWAY_TEST_TEMPLATES:
                all_test.append((endpoint, endpoint_info, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for endpoint, endpoint_info, template in sampled_training:
            task = template.format(endpoint=endpoint.replace("-", " ").title())
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{endpoint_info['error_code']} → {endpoint_info['failure_reason']}",
                    black_swan_type="Integration/Gateway_Masking",
                    precept_lesson=endpoint_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for endpoint, endpoint_info, template in sampled_test:
            task = template.format(endpoint=endpoint.replace("-", " ").title())
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {endpoint_info['recovery_action']} (1 step)",
                    black_swan_type="Integration/Gateway_Masking",
                    precept_lesson=f"PRECEPT applies {endpoint_info['recovery_action']} (learned)",
                    phase="test",
                    tests_learning=endpoint,
                )
            )

        return training + test_variations

    def generate_webhook_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate webhook/event scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Use OAuth sources that have webhooks
        webhook_sources = ["stripe", "salesforce", "hubspot"]

        # Build combinations
        all_training = []
        all_test = []

        for issue, issue_info in self.config.WEBHOOK_ISSUES.items():
            for source in webhook_sources:
                for template in self.config.WEBHOOK_TRAINING_TEMPLATES:
                    all_training.append((issue, issue_info, source, template))
                for template in self.config.WEBHOOK_TEST_TEMPLATES:
                    all_test.append((issue, issue_info, source, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for issue, issue_info, source, template in sampled_training:
            task = template.format(source=source.replace("_", " ").title())
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['issue_description']}",
                    black_swan_type="Integration/Webhook_Replay",
                    precept_lesson=issue_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for issue, issue_info, source, template in sampled_test:
            task = template.format(source=source.replace("_", " ").title())
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {issue_info['solution']} (1 step)",
                    black_swan_type="Integration/Webhook_Replay",
                    precept_lesson=f"PRECEPT applies {issue_info['solution']} (learned)",
                    phase="test",
                    tests_learning=issue,
                )
            )

        return training + test_variations

    def generate_fleet_learning_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 1,
    ) -> List[Dict]:
        """
        Generate scenarios that explicitly test CROSS-ENTITY TRANSFER.

        Args:
            num_conditions: Number of conditions per scenario (for interface consistency)

        This is the "Fleet Learning" pattern (Enterprise Integrations):
        - Training: Integration X + Error Y → learns Rule Z
        - Testing: Integration K (DIFFERENT!) + SAME Error Y → applies Rule Z

        GUARANTEED PATTERN:
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - Salesforce OAuth (INT-401) for CRM sync → learns "INT-401 → re-authenticate"

        Testing (DIFFERENT integrations, SAME authentication issues):
          - Salesforce OAuth (INT-401) for Marketing sync → applies "INT-401 → re-auth" ✓
          - Salesforce OAuth (INT-401) for Support sync → applies "INT-401 → re-auth" ✓
        ═══════════════════════════════════════════════════════════════════════════

        Key insight: Rules are learned by ERROR CODE (condition), not by
        specific integration purpose (entity). This enables cross-entity transfer.
        SOC analyst's discovery benefits all integration flows.
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Get OAuth sources and integration types
        oauth_sources = self.config.OAUTH_SOURCES
        integration_types = [
            "CRM_sync",
            "Marketing_automation",
            "Support_tickets",
            "Analytics",
            "Customer_data",
            "Revenue_reporting",
        ]

        # For each OAuth source, create:
        # - Training: One integration type that learns the rule
        # - Testing: Different integration types that apply the rule

        for source, source_info in oauth_sources.items():
            error_code = source_info["error_code"]
            recovery = source_info["recovery_action"]
            data_type = source_info.get("data_type", "data")

            if len(integration_types) < 2:
                continue

            # Shuffle integration types
            type_list = integration_types.copy()
            random.shuffle(type_list)

            # TRAINING: First integration type learns the rule
            training_type = type_list[0]

            training.append(
                {
                    "task": f"Connect {source} for {training_type} ({data_type})",
                    "expected": f"{error_code} → {recovery}",
                    "black_swan_type": "Integration/FleetLearning_Train",
                    "precept_lesson": f"When {source} fails with {error_code}, use {recovery} for ANY integration",
                    "phase": "training",
                    "fleet_learning": {
                        "source": source,
                        "error_code": error_code,
                        "learned_recovery": recovery,
                        "training_type": training_type,
                    },
                }
            )

            # TESTING: Different integration types apply the SAME rule
            for test_type in type_list[1:3]:
                testing.append(
                    {
                        "task": f"Setup {source} integration for {test_type}",
                        "expected": f"Apply learned rule: {error_code} → {recovery}",
                        "black_swan_type": "Integration/FleetLearning_Test",
                        "precept_lesson": f"Cross-entity transfer: Rule for {error_code} applies to {test_type}",
                        "phase": "test",
                        "tests_learning": f"fleet_learning_{error_code}",
                        "fleet_learning": {
                            "source": source,
                            "error_code": error_code,
                            "expected_recovery": recovery,
                            "different_type": test_type,
                            "training_type": training_type,
                        },
                    }
                )

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(
            f"  🚀 Fleet Learning (Integration): {len(training)} train + {len(testing)} test scenarios"
        )
        print(
            "     Pattern: Source + Error Y → Rule Z | Source + Different Integration + Error Y → Apply Z"
        )

        return training + testing

    def generate_multi_condition_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 3,
        test_mode: str = "matched",
    ) -> List[Dict]:
        """
        Generate MULTI-CONDITION scenarios that test PRECEPT's deterministic advantage.

        Uses FIXED KEY POOL approach for β-coverage control (same as logistics).

        Pattern:
            Entity X + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → Rule Z

        Example (3 conditions):
            Salesforce + INT-401 + DAT-SYNC + NET-TIMEOUT → re-authenticate with retry

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)
            test_mode: "matched" for exact keys, "random" for mixed testing

        Returns:
            List of multi-condition scenarios
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)
        num_conditions = max(1, min(10, num_conditions))

        conditions_provider = IntegrationConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        oauth_sources = self.config.OAUTH_SOURCES
        integration_types = [
            "CRM_sync",
            "Marketing_automation",
            "Support_tickets",
            "Analytics",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATES: NO SOURCE NAMES - matches Logistics approach!
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Removing source names prevents ExpeL from using them as
        # retrieval anchors. This ensures fair comparison where only
        # condition_key determines the solution (like Logistics).
        # ═══════════════════════════════════════════════════════════════════════
        training_templates = [
            "Connect integration ({int_type})",
            "Setup connector ({int_type})",
            "Configure integration ({int_type})",
        ]
        test_templates = [
            "Setup integration ({int_type})",
            "Link connector ({int_type})",
        ]

        # Get valid sources for multi-condition scenarios (always valid sources)
        valid_sources = getattr(
            self.config,
            "MULTI_CONDITION_VALID_SOURCES",
            ["salesforce-backup", "hubspot-v2"],
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        K = len(oauth_sources)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per OAuth source
        # DETERMINISTIC ASSIGNMENT: Alternate solutions to ensure balance
        # This prevents baselines from learning one solution and applying it everywhere
        fixed_key_pool = {}
        source_list = list(oauth_sources.keys())
        for idx, source in enumerate(source_list):
            source_info = oauth_sources[source]
            error_code = source_info["error_code"]
            # Deterministic: alternate between valid sources based on index
            valid_source = valid_sources[idx % len(valid_sources)]

            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            fixed_key_pool[source] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": valid_source,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD TRAINING COMBOS USING FIXED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        all_training_combos = []
        all_test_combos = []

        for source, source_info in oauth_sources.items():
            error_code = source_info["error_code"]

            key_info = fixed_key_pool[source]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]
            valid_source = key_info["solution"]

            for int_type in integration_types:
                for template in training_templates:
                    all_training_combos.append(
                        {
                            "source": source,
                            "error_code": error_code,
                            "valid_source": valid_source,
                            "int_type": int_type,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
                        }
                    )

                for template in test_templates:
                    all_test_combos.append(
                        {
                            "source": source,
                            "error_code": error_code,
                            "valid_source": valid_source,
                            "int_type": int_type,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
                        }
                    )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: SAMPLE N TRAINING SCENARIOS WITH GUARANTEED β-COVERAGE
        # ═══════════════════════════════════════════════════════════════════════
        # Group combos by condition_key to ensure each key is seen exactly β times
        combos_by_key = {}
        for combo in all_training_combos:
            key = combo["condition_key"]
            if key not in combos_by_key:
                combos_by_key[key] = []
            combos_by_key[key].append(combo)

        # Sample exactly β combos from each key (guaranteed uniform coverage)
        sampled_training = []
        for key, combos in combos_by_key.items():
            # Sample β combos for this key (or all if fewer available)
            samples_for_key = random.sample(combos, min(beta, len(combos)))
            sampled_training.extend(samples_for_key)

        # If we need more to reach num_training, sample additional randomly
        if len(sampled_training) < num_training:
            remaining = [c for c in all_training_combos if c not in sampled_training]
            extra = random.sample(
                remaining, min(num_training - len(sampled_training), len(remaining))
            )
            sampled_training.extend(extra)

        # Shuffle to avoid clustering by key
        random.shuffle(sampled_training)
        sampled_training = sampled_training[:num_training]

        print(f"     ✓ Guaranteed β-coverage: {len(combos_by_key)} keys × {beta} = {len(combos_by_key) * beta} scenarios")

        training = []
        training_condition_keys = {}

        for combo in sampled_training:
            source = combo["source"]
            int_type = combo["int_type"]
            template = combo["template"]
            valid_source = combo["valid_source"]  # Working source, not recovery action
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(int_type=int_type)

            training.append(
                {
                    "task": task,
                    "expected": f"{condition_key} → {valid_source}",
                    "black_swan_type": f"Integration/MultiCondition_{num_conditions}C_Train",
                    "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {valid_source}",
                    "phase": "training",
                    "condition_key": condition_key,
                    "test_mode": test_mode,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "source": source,  # Keep for internal tracking
                        "solution": valid_source,  # Working source
                    },
                }
            )

            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "source": source,
                "solution": valid_source,  # Working source
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: GENERATE TEST SCENARIOS MATCHING LEARNED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        testing = []
        sampled_test = random.sample(
            all_test_combos, min(num_test * 2, len(all_test_combos))
        )

        for combo in sampled_test:
            if len(testing) >= num_test:
                break

            source = combo["source"]
            int_type = combo["int_type"]
            template = combo["template"]

            matching_keys = [
                (k, v)
                for k, v in training_condition_keys.items()
                if v["source"] == source
            ]
            if not matching_keys:
                continue

            condition_key, key_info = random.choice(matching_keys)
            all_conds = key_info["conditions"]
            solution = key_info["solution"]

            cond_str = " + ".join(all_conds)
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(int_type=int_type)

            testing.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Integration/MultiCondition_{num_conditions}C_Test",
                    "precept_lesson": f"Cross-entity transfer: ALL {num_conditions} conditions match → apply rule",
                    "phase": "test",
                    "tests_learning": f"multi_condition_{condition_key}",
                    "condition_key": condition_key,
                    "test_mode": test_mode,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "expected_solution": solution,
                    },
                }
            )

        print(
            f"  🔀 Multi-Condition ({num_conditions}C): {len(training)} train + {len(testing)} test"
        )
        print(
            f"     Pool: {len(all_training_combos)} training combos, {len(all_test_combos)} test combos"
        )
        print(
            f"     Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states!"
        )

        return training + testing

    def generate_all(
        self,
        include_generator_samples: bool = False,
        ensure_coverage: bool = True,
        include_fleet_learning: bool = True,
        num_conditions: int = 1,
        test_mode: str = "matched",
    ) -> List[Dict]:
        """
        Generate all integration scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN: Single-condition (num_conditions=1) is just a special case!

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, guarantees training covers ALL error types
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default)
                           - num_conditions>1: Multi-condition (for ablation)

        Returns:
            Combined list of all integration scenarios
        """
        # Calculate allocations
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training

        # Clamp num_conditions to valid range
        num_conditions = max(1, min(10, num_conditions))

        # ═══════════════════════════════════════════════════════════════════════
        # UNIFIED APPROACH: num_conditions=1 is single-condition (special case)
        # ═══════════════════════════════════════════════════════════════════════
        if num_conditions == 1:
            print(
                f"\n📋 Single-condition mode: {total_training} train + {total_test} test"
            )
        else:
            print(
                f"\n🔬 Multi-condition mode ({num_conditions}C): {total_training} train + {total_test} test"
            )
            print(
                f"   Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states"
            )

        # Generate scenarios using multi-condition approach
        scenarios = self.generate_multi_condition_scenarios(
            num_training=total_training,
            num_test=total_test,
            num_conditions=num_conditions,
            test_mode=test_mode,
        )

        # OPTION B: Always include supplementary scenarios for comprehensive testing
        if include_generator_samples:
            generator_scenarios = self.generate_from_universal_generator(num_samples=2)
            scenarios.extend(generator_scenarios)

        if include_fleet_learning:
            # Fleet learning tests cross-entity transfer
            fleet_train = max(1, total_training // 4)
            fleet_test = max(1, total_test // 2)
            fleet_scenarios = self.generate_fleet_learning_scenarios(
                num_training=fleet_train,
                num_test=fleet_test,
                num_conditions=num_conditions,  # Pass for consistency
            )
            scenarios.extend(fleet_scenarios)

        return scenarios

    def _generate_with_coverage_guarantee(
        self,
        total_training: int,
        total_test: int,
        include_generator_samples: bool = False,
    ) -> List[Dict]:
        """
        Generate scenarios with GUARANTEED coverage of all error types in training.

        This ensures that:
        1. Training includes at least one scenario for EACH OAuth source
        2. Training includes at least one scenario for EACH gateway endpoint
        3. Training includes at least one scenario for EACH webhook issue
        4. Test scenarios will ALWAYS have a corresponding learned rule

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. OAuth sources - one per source
        for source, source_info in self.config.OAUTH_SOURCES.items():
            template = random.choice(self.config.OAUTH_TRAINING_TEMPLATES)
            task = template.format(
                source=source.replace("_", " ").title(),
                data_type=source_info["data_type"],
            )

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{source_info['error_code']} → {source_info['failure_reason']}",
                    black_swan_type="Integration/Auth_Zombie",
                    precept_lesson=source_info["lesson"],
                    phase="training",
                )
            )

        # 1b. Gateway endpoints - one per endpoint
        for endpoint, endpoint_info in self.config.GATEWAY_ENDPOINTS.items():
            template = random.choice(self.config.GATEWAY_TRAINING_TEMPLATES)
            task = template.format(endpoint=endpoint.replace("-", " ").title())

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{endpoint_info['error_code']} → {endpoint_info['failure_reason']}",
                    black_swan_type="Integration/Gateway_Masking",
                    precept_lesson=endpoint_info["lesson"],
                    phase="training",
                )
            )

        # 1c. Webhook issues - one per issue type
        webhook_sources = ["stripe", "salesforce", "hubspot"]
        for issue, issue_info in self.config.WEBHOOK_ISSUES.items():
            source = random.choice(webhook_sources)
            template = random.choice(self.config.WEBHOOK_TRAINING_TEMPLATES)
            task = template.format(source=source.replace("_", " ").title())

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['issue_description']}",
                    black_swan_type="Integration/Webhook_Replay",
                    precept_lesson=issue_info["lesson"],
                    phase="training",
                )
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: FILL REMAINING TRAINING SLOTS (if any)
        # ═══════════════════════════════════════════════════════════════════════

        mandatory_count = len(training_scenarios)
        remaining_training = max(0, total_training - mandatory_count)

        if remaining_training > 0:
            extra = self.generate_oauth_scenarios(
                num_training=remaining_training, num_test=0
            )
            training_scenarios.extend(
                [s for s in extra if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS (evenly distributed)
        # ═══════════════════════════════════════════════════════════════════════

        oauth_test = max(1, int(total_test * 0.4))
        gateway_test = max(1, int(total_test * 0.35))
        webhook_test = total_test - oauth_test - gateway_test

        test_oauth = self.generate_oauth_scenarios(num_training=0, num_test=oauth_test)
        test_gateway = self.generate_gateway_scenarios(
            num_training=0, num_test=gateway_test
        )
        test_webhook = self.generate_webhook_scenarios(
            num_training=0, num_test=webhook_test
        )

        test_scenarios.extend([s for s in test_oauth if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_gateway if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_webhook if s.get("phase") == "test"])

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: LOG COVERAGE
        # ═══════════════════════════════════════════════════════════════════════

        training_error_codes = set()
        for s in training_scenarios:
            expected = s.get("expected", "")
            if "→" in expected:
                error_code = expected.split("→")[0].strip()
                training_error_codes.add(error_code)

        print(
            f"  📋 Coverage Guarantee: Training covers {len(training_error_codes)} error codes"
        )
        print(f"     Error codes: {sorted(training_error_codes)}")

        if include_generator_samples:
            training_scenarios.extend(
                self.generate_from_universal_generator(num_samples=2)
            )

        return training_scenarios + test_scenarios

    def generate_test_from_learned_keys(
        self,
        learned_rule_keys: Dict[str, str],
        num_test: int,
        mode: str = "matched",
        seed: int = None,
        all_training_keys: List[str] = None,
    ) -> List[Dict]:
        """
        Generate test scenarios from specific learned rule keys.

        This method is used AFTER training to generate test scenarios that:
        - MATCHED mode: Use EXACT condition keys from learned rules (O(1) lookup test)
        - RANDOM mode: MIXED - 50% exact match (Tier 1) + 50% novel (Tier 2/3)

        Args:
            learned_rule_keys: Dict mapping condition_key -> rule_text
            num_test: Number of test scenarios to generate
            mode: "matched" for exact key reuse, "random" for mixed testing
            seed: Random seed for reproducibility
            all_training_keys: ALL condition keys encountered during training

        Returns:
            List of test scenario dictionaries
        """
        if seed is not None:
            random.seed(seed)

        if mode == "matched" and not learned_rule_keys:
            print("  ⚠️ No learned rule keys provided for MATCHED mode")
            return []

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION: For MATCHED mode, ensure learned keys are in training pool
        # ═══════════════════════════════════════════════════════════════════════
        if mode == "matched" and all_training_keys:
            training_set = set(all_training_keys)
            valid_learned = {
                k: v for k, v in learned_rule_keys.items() if k in training_set
            }
            invalid_count = len(learned_rule_keys) - len(valid_learned)
            if invalid_count > 0:
                print(
                    f"  ⚠️ MATCHED mode: {invalid_count} learned keys not in training pool (filtered)"
                )
            learned_rule_keys = valid_learned
            if not learned_rule_keys:
                print("  ⚠️ No learned keys match training pool after filtering")
                return []

        # Initialize conditions provider
        conditions_provider = IntegrationConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Domain-specific templates
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Connect to API ({int_type})",
            "Authenticate with service ({int_type})",
            "Setup webhook ({int_type})",
            "Configure integration ({int_type})",
            "Establish connection ({int_type})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Use OAUTH_SOURCES for testing, not generic APIs!
        # Generic APIs (Stripe, GitHub) are NOT enforced by MCP server.
        # Only OAUTH_SOURCES have per-source authentication rules.
        # ═══════════════════════════════════════════════════════════════════════
        from ..config import IntegrationConfig
        oauth_sources = IntegrationConfig.OAUTH_SOURCES
        apis = list(oauth_sources.keys())  # salesforce, hubspot, etc.

        # Build reverse mapping: error_code -> source
        error_code_to_source = {}
        for src, src_info in oauth_sources.items():
            error_code_to_source[src_info["error_code"]] = src

        # ═══════════════════════════════════════════════════════════════════════
        # KEY SELECTION STRATEGY - FIXED FOR PROPER CROSS-EPISODE LEARNING TEST
        # ═══════════════════════════════════════════════════════════════════════
        learned_keys_list = list(learned_rule_keys.keys())
        random.shuffle(learned_keys_list)

        if mode == "matched":
            # Use ALL TRAINING KEYS for proper cross-episode learning test
            if all_training_keys:
                base_keys_list = list(all_training_keys)
                random.shuffle(base_keys_list)
                learned_count = len(learned_keys_list)
                unlearned_count = len(base_keys_list) - learned_count
                print(f"  📋 MATCHED mode: Using ALL {len(base_keys_list)} training keys")
                print(f"     ✓ {learned_count} keys with learned rules (TIER 1 hit expected)")
                print(f"     ○ {unlearned_count} keys without rules (cross-episode learning test)")
            else:
                base_keys_list = learned_keys_list
                print(f"  📋 MATCHED mode: Using {len(base_keys_list)} learned rule keys (fallback)")
            for i, key in enumerate(base_keys_list[:5]):
                solution = learned_rule_keys.get(key, "❌ NOT LEARNED")
                if isinstance(solution, str) and "→" in solution:
                    solution = "✓ " + solution.split("→")[-1].strip()
                print(f"     [{i + 1}] {key[:50]}... → {solution}")
            if len(base_keys_list) > 5:
                print(f"     ... and {len(base_keys_list) - 5} more")
        else:
            # RANDOM mode: Use learned keys as base for BOTH matched and novel
            base_keys_list = learned_keys_list
            print(f"  📋 RANDOM mode: Using {len(base_keys_list)} learned keys as base")

        if not base_keys_list:
            print("  ⚠️ No learned keys available for scenario generation")
            return []

        scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # MIXED MODE: GUARANTEED 50% exact match + 50% novel
        # ═══════════════════════════════════════════════════════════════════════
        if mode == "random":
            num_exact_match = num_test // 2
            num_novel = num_test - num_exact_match
            print(
                f"  🔀 MIXED RANDOM mode: {num_exact_match} exact-match + {num_novel} novel scenarios"
            )
            print(
                f"     Tier 1 (O(1) lookup): {num_exact_match} | Tier 2/3 (similarity): {num_novel}"
            )
            print(f"     Using {len(learned_keys_list)} learned keys (cycling if needed)")

        for i in range(num_test):
            # Cycle through learned keys (guaranteed to be in learned_rule_keys)
            base_key = base_keys_list[i % len(base_keys_list)]
            base_conditions = base_key.split("+")

            # Get solution - may not exist for unlearned keys
            from ..config import IntegrationConfig

            if base_key in learned_rule_keys:
                rule_text = learned_rule_keys[base_key]
                # ═══════════════════════════════════════════════════════════════
                # FIX: Extract solution robustly from various rule formats
                # ═══════════════════════════════════════════════════════════════
                solution = "unknown"
                if " → " in rule_text:
                    parts = rule_text.split(" → ", 1)
                    if len(parts) == 2:
                        solution_part = parts[1].strip()
                        if solution_part.upper().startswith("LLM") and "→" in solution_part:
                            path_parts = solution_part.split("→")
                            for part in path_parts:
                                part = part.strip()
                                if part and part.upper() != "LLM":
                                    solution = part
                                    break
                        else:
                            solution = solution_part
                elif "→" in rule_text:
                    solution = rule_text.split("→")[-1].strip()
                is_learned = True
            else:
                # Unlearned key - compute expected solution from hash
                solution = IntegrationConfig.get_valid_solution_for_conditions(base_key)
                is_learned = False

            # Track test type
            test_type = "exact_match"

            if mode == "matched":
                condition_key = base_key
                all_conds = base_conditions
                test_type = "exact_match"
            elif mode == "random":
                # ═══════════════════════════════════════════════════════════════════
                # RANDOM (MIXED): GUARANTEED 50% exact match + 50% novel
                # ═══════════════════════════════════════════════════════════════════
                num_exact_match = num_test // 2

                if i < num_exact_match:
                    # First 50%: Use EXACT key from learned_rule_keys (tests Tier 1)
                    # GUARANTEED to match because base_keys_list = learned_keys_list
                    condition_key = base_key
                    all_conds = base_conditions
                    test_type = "exact_match"
                else:
                    # ═══════════════════════════════════════════════════════════
                    # CRITICAL: Always keep the source's error code!
                    # This ensures the test uses the correct source for enforcement.
                    # ═══════════════════════════════════════════════════════════
                    source_error_codes = set(error_code_to_source.keys())
                    source_cond = None
                    for c in base_conditions:
                        if c in source_error_codes:
                            source_cond = c
                            break

                    other_conditions = [c for c in base_conditions if c != source_cond]
                    # Ensure keep_count doesn't exceed available conditions
                    keep_count = min(len(other_conditions), max(1, int(len(other_conditions) * 0.6)))
                    replace_count = max(0, len(other_conditions) - keep_count)

                    kept_others = random.sample(
                        other_conditions, keep_count
                    ) if other_conditions else []

                    if source_cond:
                        kept_conditions = [source_cond] + kept_others
                    else:
                        kept_conditions = kept_others

                    available_new = [c for c in condition_codes if c not in kept_conditions]
                    # Only sample new conditions if we have some to replace
                    new_conditions = random.sample(
                        available_new, min(replace_count, len(available_new))
                    ) if replace_count > 0 and available_new else []

                    all_conds = sorted(kept_conditions + new_conditions)
                    condition_key = "+".join(all_conds)
                    test_type = "novel"
            else:
                condition_key = base_key
                all_conds = base_conditions
                test_type = "exact_match"

            template = test_templates[i % len(test_templates)]

            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: Determine source from error_code in condition_key!
            # The condition_key contains the source's error_code (e.g., INT-401).
            # We must use the correct source so MCP enforcement is accurate.
            # ═══════════════════════════════════════════════════════════════════
            api = apis[i % len(apis)]  # Default fallback
            for cond in all_conds:
                if cond in error_code_to_source:
                    api = error_code_to_source[cond]
                    break

            # Use generic integration type context
            int_types = ["oauth", "webhook", "api", "realtime"]
            int_type = int_types[i % len(int_types)]

            # BLACK SWAN CSP: NO conditions in task - only integration type context
            task = template.format(int_type=int_type)

            num_conditions = len(all_conds)
            scenarios.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Integration/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
                    "precept_lesson": f"Transfer learning: apply rule for {condition_key}",
                    "phase": "test",
                    "tests_learning": f"multi_condition_{condition_key}",
                    "condition_key": condition_key,
                    "test_mode": mode,
                    "test_type": test_type,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "expected_solution": solution,
                        "base_key": base_key,
                        "test_mode": mode,
                        "test_type": test_type,
                    },
                }
            )

        mode_desc = (
            "MATCHED (exact keys)" if mode == "matched" else "MIXED (50% exact + 50% novel)"
        )
        print(
            f"  📋 Generated {len(scenarios)} test scenarios from learned keys ({mode_desc})"
        )
        return scenarios

    def generate_semantic_compositional_test(
        self,
        num_train: int = 8,
        num_test: int = 4,
        seed: int = None,
        beta: int = 1,
        filter_by_learned: bool = True,
        test_num_conditions: int = 2,
    ) -> tuple:
        """
        Generate SEMANTIC compositional tests for Integration domain where composite
        solutions ARE derivable from atomic precepts, enabling P₁ > 0%.

        Integration semantic conditions map to API patterns:
        - Tier 3 (Highest): Authentication - non-negotiable
        - Tier 2 (Middle): Reliability requirements
        - Tier 1 (Lowest): Efficiency preferences

        Returns:
            Tuple of (training_scenarios, test_scenarios, semantic_mappings)
        """
        if seed is not None:
            random.seed(seed)

        print(f"\n  🧠 INTEGRATION SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(f"     Beta={beta} (each atomic condition trained {beta}x)")
        print(f"     Test complexity: {test_num_conditions}-way combinations")

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Solutions MUST match IntegrationConfig.MULTI_CONDITION_VALID_SOLUTIONS
        # Valid sources: salesforce-backup, hubspot-v2 (tool validates against these)
        # ═══════════════════════════════════════════════════════════════════════
        semantic_conditions = {
            # Tier 3 (Highest): Authentication - non-negotiable
            "AUTH": {
                "meaning": "Secure authentication required",
                "solution": "hubspot-v2",  # HubSpot: OAuth2 support
                "reasoning": "Secure auth requires HubSpot v2 (hubspot-v2)",
                "tier": 3,
            },
            # Tier 2 (Middle): Reliability requirements
            "RETRY": {
                "meaning": "Retry logic for reliability",
                "solution": "salesforce-backup",  # Salesforce: reliable retries
                "reasoning": "Retry patterns use Salesforce backup (salesforce-backup)",
                "tier": 2,
            },
            "RATE": {
                "meaning": "Rate limiting compliance",
                "solution": "hubspot-v2",  # HubSpot: rate limit handling
                "reasoning": "Rate limits use HubSpot v2 (hubspot-v2)",
                "tier": 2,
            },
            "VERIFY": {
                "meaning": "Request verification required",
                "solution": "hubspot-v2",  # HubSpot: JWT verification
                "reasoning": "Verification uses HubSpot v2 (hubspot-v2)",
                "tier": 2,
            },
            # Tier 1 (Lowest): Efficiency preferences
            "BATCH": {
                "meaning": "Batch processing for efficiency",
                "solution": "salesforce-backup",  # Salesforce: batch API
                "reasoning": "Batching uses Salesforce backup (salesforce-backup)",
                "tier": 1,
            },
            "STREAM": {
                "meaning": "Real-time streaming data",
                "solution": "hubspot-v2",  # HubSpot: webhooks
                "reasoning": "Streaming uses HubSpot v2 (hubspot-v2)",
                "tier": 1,
            },
            "SIMPLE": {
                "meaning": "Simple API key authentication",
                "solution": "salesforce-backup",  # Salesforce: simple auth
                "reasoning": "Simple auth uses Salesforce backup (salesforce-backup)",
                "tier": 1,
            },
            "QUERY": {
                "meaning": "Flexible query patterns",
                "solution": "salesforce-backup",  # Salesforce: SOQL queries
                "reasoning": "Flexible queries use Salesforce backup (salesforce-backup)",
                "tier": 1,
            },
        }

        def compute_composite_solution(conditions: List[str]) -> str:
            if not conditions:
                return "salesforce-backup"  # Default to Salesforce

            best_cond = None
            best_tier = -1
            for cond in sorted(conditions):
                if cond in semantic_conditions:
                    tier = semantic_conditions[cond]["tier"]
                    if tier > best_tier:
                        best_tier = tier
                        best_cond = cond

            if best_cond:
                return semantic_conditions[best_cond]["solution"]
            return "salesforce-backup"  # Default to Salesforce

        available_conditions = list(semantic_conditions.keys())
        random.shuffle(available_conditions)

        def prioritize_solution_diversity(conditions: List[str]) -> List[str]:
            diverse = []
            seen_solutions = set()
            for cond in conditions:
                solution = semantic_conditions[cond]["solution"]
                if solution not in seen_solutions:
                    diverse.append(cond)
                    seen_solutions.add(solution)
            remaining = [cond for cond in conditions if cond not in diverse]
            return diverse + remaining

        ordered_conditions = prioritize_solution_diversity(available_conditions)
        train_conditions = ordered_conditions[:min(6, len(ordered_conditions))]

        print(f"     Training conditions: {train_conditions}")

        training_scenarios = []
        task_templates = [
            "Integrate with {service} API",
            "Connect to {service} service",
            "Setup {service} integration",
            "Configure {service} endpoint",
        ]
        services = ["stripe", "twilio", "sendgrid", "slack", "github", "salesforce"]

        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]
                template = task_templates[scenario_idx % len(task_templates)]
                service = services[scenario_idx % len(services)]

                scenario = {
                    "task": template.format(service=service),
                    "expected": solution,
                    "black_swan_type": "semantic_atomic",
                    "precept_lesson": f"{cond}: {cond_info['meaning']} → use {solution}",
                    "phase": "train",
                    "test_type": "atomic_semantic",
                    "tests_learning": "semantic_compositional",
                    "repetition": rep + 1,
                    "multi_condition": {
                        "condition_key": cond,
                        "conditions": [cond],
                        "num_conditions": 1,
                        "semantic_meaning": cond_info["meaning"],
                        "semantic_reasoning": cond_info["reasoning"],
                        "solution": solution,
                        "tier": cond_info["tier"],
                    },
                }
                training_scenarios.append(scenario)
                scenario_idx += 1

        print(f"     Generated {len(training_scenarios)} atomic training scenarios")

        from itertools import combinations

        num_atoms_for_combos = min(6, len(train_conditions))
        if num_atoms_for_combos < test_num_conditions:
            test_num_conditions = num_atoms_for_combos

        all_combos = list(combinations(train_conditions[:num_atoms_for_combos], test_num_conditions))

        def is_nontrivial_combo(combo: tuple) -> bool:
            solutions = {semantic_conditions[c]["solution"] for c in combo}
            tiers = {semantic_conditions[c]["tier"] for c in combo}
            return len(solutions) > 1 and len(tiers) > 1

        filtered_combos = [combo for combo in all_combos if is_nontrivial_combo(combo)]
        if filtered_combos:
            all_combos = filtered_combos
        else:
            print("     ⚠️ No non-trivial combos found; using full combo set")

        random.shuffle(all_combos)

        test_scenarios = []
        for i, combo in enumerate(all_combos):
            if i >= num_test:
                break

            condition_key = "+".join(sorted(combo))
            composite_solution = compute_composite_solution(list(combo))

            meanings = [semantic_conditions[c]["meaning"] for c in combo]
            solutions = [semantic_conditions[c]["solution"] for c in combo]
            tiers = [semantic_conditions[c]["tier"] for c in combo]
            reasonings = [semantic_conditions[c]["reasoning"] for c in combo]

            winning_idx = tiers.index(max(tiers))
            winning_cond = list(combo)[winning_idx]

            template = task_templates[i % len(task_templates)]
            service = services[i % len(services)]

            derivation_parts = [f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})" for c in combo]
            derivation_rule = f"{' vs '.join(derivation_parts)} → {winning_cond} wins → {composite_solution}"

            scenario = {
                "task": template.format(service=service),
                "expected": composite_solution,
                "black_swan_type": "semantic_compositional",
                "precept_lesson": f"Composite {condition_key}: {derivation_rule}",
                "phase": "test",
                "test_type": "compositional_semantic",
                "tests_learning": "semantic_compositional_generalization",
                "multi_condition": {
                    "condition_key": condition_key,
                    "conditions": list(combo),
                    "num_conditions": len(combo),
                    "all_atoms_learned": True,
                    "semantic_meanings": meanings,
                    "atomic_solutions": solutions,
                    "atomic_tiers": tiers,
                    "atomic_reasonings": reasonings,
                    "winning_condition": winning_cond,
                    "composite_solution": composite_solution,
                    "derivation_rule": derivation_rule,
                },
            }
            test_scenarios.append(scenario)

        print(f"     Generated {len(test_scenarios)} semantic {test_num_conditions}-way test scenarios")

        trained_atom_list = train_conditions[:min(num_train, len(train_conditions))]
        semantic_mappings = {
            "conditions": semantic_conditions,
            "trained_atoms": trained_atom_list,
            "derivation_rule": "composite_solution = solution of highest-tier condition",
            "beta": beta,
            "filter_by_learned": filter_by_learned,
            "test_num_conditions": test_num_conditions,
            "learned_atoms": None,
            "test_composite_requirements": {
                "+".join(sorted(list(combo))): list(combo)
                for combo in combinations(trained_atom_list[:min(4, len(trained_atom_list))], test_num_conditions)
            } if len(trained_atom_list) >= test_num_conditions else {},
        }

        return training_scenarios, test_scenarios, semantic_mappings


def generate_integration_scenarios(
    num_samples: int = 10,
    train_ratio: float = 0.6,
    include_generator_samples: bool = False,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate integration black swan scenarios.

    Args:
        num_samples: TOTAL number of scenarios (train + test).
        train_ratio: Ratio of training samples (0.0 to 1.0).
        include_generator_samples: Also include UniversalDataGenerator samples.
        include_fleet_learning: Include fleet learning scenarios. Default: True.
        num_conditions: Number of conditions per scenario (1-10). Default: 1.
        test_mode: "matched" (reuse training keys) or "random" (new random keys).

    Returns:
        List of scenario dictionaries with training and test phases
    """
    generator = IntegrationScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
