"""
DevOps Scenario Generator for PRECEPT Testing.

This module generates DevOps black swan scenarios using template-based variations.
Supports configurable num_samples and train_ratio for flexible train/test splits.

Configuration is imported from precept.config.devops - single source of truth.

Usage:
    from precept.scenario_generators import DevOpsScenarioGenerator

    generator = DevOpsScenarioGenerator(num_samples=20, train_ratio=0.6)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import DevOpsConfig
from ..config.multi_condition import (
    DevOpsConditions,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.devops
DevOpsScenarioConfig = DevOpsConfig


class DevOpsScenarioGenerator:
    """
    Generate DevOps black swan scenarios using template-based variations.

    COHERENCE GUARANTEE: Each scenario maintains semantic consistency:
    - Stack failures use the correct error codes and recovery actions
    - IAM propagation delays match the service type
    - K8s issues use appropriate resource limits

    Usage:
        generator = DevOpsScenarioGenerator(num_samples=20, train_ratio=0.6)
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
        self.devops_traps = BLACK_SWAN_DEFINITIONS.get("DevOps", {})
        self.config = DevOpsScenarioConfig

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

        for trap_name, trap_def in self.devops_traps.items():
            sample = self.generator.generate_sample(
                category="DevOps",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"DevOps/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios[:num_samples]

    def generate_cloudformation_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate CloudFormation scenarios with template-based variations.

        COHERENCE GUARANTEE: Each stack always uses:
        - Its own error_code (UPDATE_ROLLBACK_FAILED, etc.)
        - Its own recovery_action (continue_update_rollback, etc.)
        - Its own lesson text
        """
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training
        num_training = (
            num_training if num_training is not None else max(2, total_training // 3)
        )
        num_test = num_test if num_test is not None else max(1, total_test // 3)

        # Build all possible COHERENT combinations
        all_training_combos = []
        all_test_combos = []

        for stack, stack_info in self.config.STUCK_STACKS.items():
            for update_key, update_desc in self.config.UPDATE_TYPES.items():
                for template in self.config.CFN_TRAINING_TEMPLATES:
                    all_training_combos.append(
                        (stack, stack_info, update_key, update_desc, template)
                    )
                for template in self.config.CFN_TEST_TEMPLATES:
                    all_test_combos.append(
                        (stack, stack_info, update_key, update_desc, template)
                    )

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for stack, stack_info, update_key, update_desc, template in sampled_training:
            task = template.format(stack=stack, update_type=update_desc)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{stack_info['error_code']} → {stack_info['recovery_action']}",
                    black_swan_type="DevOps/Zombie_Stack",
                    precept_lesson=stack_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for stack, stack_info, update_key, update_desc, template in sampled_test:
            task = template.format(stack=stack, update_type=update_desc)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {stack_info['recovery_action']} (1 step)",
                    black_swan_type="DevOps/Zombie_Stack",
                    precept_lesson=f"PRECEPT applies {stack_info['recovery_action']} (learned)",
                    phase="test",
                    tests_learning=stack,
                )
            )

        return training + test_variations

    def generate_iam_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate IAM propagation delay scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Build combinations
        all_training = []
        all_test = []

        for role, role_info in self.config.IAM_ROLES.items():
            for template in self.config.IAM_TRAINING_TEMPLATES:
                all_training.append((role, role_info, template))
            for template in self.config.IAM_TEST_TEMPLATES:
                all_test.append((role, role_info, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for role, role_info, template in sampled_training:
            task = template.format(role=role, service=role_info["service"])
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{role_info['error_code']} → wait {role_info['wait_time']}",
                    black_swan_type="DevOps/Consistency_Race",
                    precept_lesson=role_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for role, role_info, template in sampled_test:
            task = template.format(role=role, service=role_info["service"])
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: wait {role_info['wait_time']} (1 step)",
                    black_swan_type="DevOps/Consistency_Race",
                    precept_lesson="PRECEPT waits for propagation (learned)",
                    phase="test",
                    tests_learning=role,
                )
            )

        return training + test_variations

    def generate_kubernetes_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate Kubernetes pod eviction scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Build combinations
        all_training = []
        all_test = []

        # K8S_ISSUES now maps pod_name -> issue_info (pod name IS the key)
        for pod_name, issue_info in self.config.K8S_ISSUES.items():
            for namespace in self.config.K8S_NAMESPACES:
                for template in self.config.K8S_TRAINING_TEMPLATES:
                    all_training.append((pod_name, issue_info, namespace, template))
                for template in self.config.K8S_TEST_TEMPLATES:
                    all_test.append((pod_name, issue_info, namespace, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for pod_name, issue_info, namespace, template in sampled_training:
            # Templates now use {pod} and {namespace}
            task = template.format(pod=pod_name, namespace=namespace)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['fix_action']}",
                    black_swan_type="DevOps/Pod_Eviction",
                    precept_lesson=issue_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for pod_name, issue_info, namespace, template in sampled_test:
            # Templates now use {pod} and {namespace}
            task = template.format(pod=pod_name, namespace=namespace)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {issue_info['fix_action']} (1 step)",
                    black_swan_type="DevOps/Pod_Eviction",
                    precept_lesson=f"PRECEPT applies {issue_info['fix_action']} (learned)",
                    phase="test",
                    tests_learning=pod_name,
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

        This is the "Fleet Learning" pattern (SRE Organization-wide):
        - Training: Region X + Error Y → learns Rule Z
        - Testing: Different Service (Entity K) + SAME Error Y → applies Rule Z

        GUARANTEED PATTERN:
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - prod-api-stack (CFN-881) in US-EAST-1 → learns "CFN-881 → continue_update_rollback"

        Testing (DIFFERENT services/regions, SAME conditions):
          - data-pipeline-stack (CFN-881) in US-WEST-2 → applies "CFN-881 → continue_update_rollback" ✓
          - auth-service-stack (CFN-881) in EU-WEST-1 → applies "CFN-881 → continue_update_rollback" ✓
        ═══════════════════════════════════════════════════════════════════════════

        Key insight: Rules are learned by ERROR CODE (condition), not by
        specific stack/service (entity). This enables cross-entity transfer.
        Organization-wide resilience from a single incident discovery.
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Get stuck stacks and regions
        stuck_stacks = self.config.STUCK_STACKS
        regions = self.config.REGIONS  # Use REGIONS from config
        environments = ["production", "staging", "development"]  # Standard environments

        # For each stuck stack, create:
        # - Training: One region/env that learns the rule
        # - Testing: Different region/env that applies the rule

        for stack, stack_info in stuck_stacks.items():
            error_code = stack_info["error_code"]
            recovery = stack_info["recovery_action"]
            stack_type = stack_info.get("stack_type", "infrastructure")

            if len(regions) < 2:
                continue

            # Shuffle regions
            region_list = regions.copy()
            random.shuffle(region_list)
            env_list = environments.copy()
            random.shuffle(env_list)

            # TRAINING: First region learns the rule
            training_region = region_list[0]
            training_env = env_list[0]

            # Build task directly (templates use update_type which we don't need here)
            task = (
                f"Deploy {stack} ({stack_type}) in {training_region} for {training_env}"
            )

            training.append(
                {
                    "task": task,
                    "expected": f"{error_code} → {recovery}",
                    "black_swan_type": "DevOps/FleetLearning_Train",
                    "precept_lesson": f"When {stack_type} fails with {error_code}, use {recovery} for ANY region/service",
                    "phase": "training",
                    "fleet_learning": {
                        "stack": stack,
                        "error_code": error_code,
                        "learned_recovery": recovery,
                        "training_region": training_region,
                        "training_env": training_env,
                    },
                }
            )

            # TESTING: Different regions/envs apply the SAME rule
            for i, test_region in enumerate(region_list[1:3]):
                test_env = env_list[(i + 1) % len(env_list)]

                # Build task directly
                test_task = (
                    f"Update {stack} in {test_region} for {test_env} environment"
                )

                testing.append(
                    {
                        "task": test_task,
                        "expected": f"Apply learned rule: {error_code} → {recovery}",
                        "black_swan_type": "DevOps/FleetLearning_Test",
                        "precept_lesson": f"Cross-entity transfer: Rule for {error_code} applies to {test_region}",
                        "phase": "test",
                        "tests_learning": f"fleet_learning_{error_code}",
                        "fleet_learning": {
                            "stack": stack,
                            "error_code": error_code,
                            "expected_recovery": recovery,
                            "different_region": test_region,
                            "different_env": test_env,
                            "training_region": training_region,
                        },
                    }
                )

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(
            f"  🚀 Fleet Learning (DevOps): {len(training)} train + {len(testing)} test scenarios"
        )
        print(
            "     Pattern: Stack X + Error Y → Rule Z | Stack X + Different Region + Error Y → Apply Z"
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
            prod-api-stack + CFN-881 + RG-FULL + SVC-CRIT → continue_update_rollback with priority

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

        conditions_provider = DevOpsConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        stuck_stacks = self.config.STUCK_STACKS
        regions = (
            self.config.REGIONS
            if self.config.REGIONS
            else ["us-east-1", "us-west-2", "eu-west-1"]
        )

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATES: NO STACK NAMES - matches Logistics approach!
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Removing stack names prevents ExpeL from using them as
        # retrieval anchors. This ensures fair comparison where only
        # condition_key determines the solution (like Logistics).
        # ═══════════════════════════════════════════════════════════════════════
        training_templates = [
            "Deploy stack ({region})",
            "Update stack ({region})",
            "Launch stack ({region})",
        ]
        test_templates = [
            "Rollout stack ({region})",
            "Deploy update ({region})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        K = len(stuck_stacks)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per stuck stack
        fixed_key_pool = {}
        for stack, stack_info in stuck_stacks.items():
            error_code = stack_info["error_code"]
            recovery = stack_info["recovery_action"]

            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            fixed_key_pool[stack] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": recovery,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD TRAINING COMBOS USING FIXED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        all_training_combos = []
        all_test_combos = []

        for stack, stack_info in stuck_stacks.items():
            error_code = stack_info["error_code"]

            key_info = fixed_key_pool[stack]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]
            recovery = key_info["solution"]

            for region in regions:
                for template in training_templates:
                    all_training_combos.append(
                        {
                            "stack": stack,
                            "error_code": error_code,
                            "recovery": recovery,
                            "region": region,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
                        }
                    )

                for template in test_templates:
                    all_test_combos.append(
                        {
                            "stack": stack,
                            "error_code": error_code,
                            "recovery": recovery,
                            "region": region,
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

        print(
            f"     ✓ Guaranteed β-coverage: {len(combos_by_key)} keys × {beta} = {len(combos_by_key) * beta} scenarios"
        )

        training = []
        training_condition_keys = {}

        for combo in sampled_training:
            stack = combo["stack"]
            region = combo["region"]
            template = combo["template"]
            recovery = combo["recovery"]
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            # NO STACK NAME in task - matches Logistics approach
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(region=region)

            training.append(
                {
                    "task": task,
                    "expected": f"{condition_key} → {recovery}",
                    "black_swan_type": f"DevOps/MultiCondition_{num_conditions}C_Train",
                    "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {recovery}",
                    "phase": "training",
                    "condition_key": condition_key,
                    "test_mode": test_mode,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "stack": stack,  # Keep for internal tracking
                        "solution": recovery,
                    },
                }
            )

            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "stack": stack,
                "solution": recovery,
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

            stack = combo["stack"]
            region = combo["region"]
            template = combo["template"]

            matching_keys = [
                (k, v)
                for k, v in training_condition_keys.items()
                if v["stack"] == stack
            ]
            if not matching_keys:
                continue

            condition_key, key_info = random.choice(matching_keys)
            all_conds = key_info["conditions"]
            solution = key_info["solution"]

            cond_str = " + ".join(all_conds)
            # NO STACK NAME in task - matches Logistics approach
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(region=region)

            testing.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"DevOps/MultiCondition_{num_conditions}C_Test",
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
        Generate all DevOps scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN: Single-condition (num_conditions=1) is just a special case!

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, guarantees training covers ALL error types
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default)
                           - num_conditions>1: Multi-condition (for ablation)

        Returns:
            Combined list of all DevOps scenarios
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
        1. Training includes at least one scenario for EACH stuck stack
        2. Training includes at least one scenario for EACH IAM role issue
        3. Training includes at least one scenario for EACH K8s issue
        4. Test scenarios will ALWAYS have a corresponding learned rule

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. Stuck stacks - one per stack
        for stack, stack_info in self.config.STUCK_STACKS.items():
            update_type = random.choice(list(self.config.UPDATE_TYPES.keys()))
            update_desc = self.config.UPDATE_TYPES[update_type]
            template = random.choice(self.config.CFN_TRAINING_TEMPLATES)

            task = template.format(stack=stack, update_type=update_desc)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{stack_info['error_code']} → {stack_info['recovery_action']}",
                    black_swan_type="DevOps/Zombie_Stack",
                    precept_lesson=stack_info["lesson"],
                    phase="training",
                )
            )

        # 1b. IAM roles - one per role
        for role, role_info in self.config.IAM_ROLES.items():
            template = random.choice(self.config.IAM_TRAINING_TEMPLATES)

            task = template.format(role=role, service=role_info["service"])

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{role_info['error_code']} → {role_info['recovery_action']}",
                    black_swan_type="DevOps/Consistency_Race",
                    precept_lesson=role_info["lesson"],
                    phase="training",
                )
            )

        # 1c. K8s issues - one per pod (pod_name is now the key in K8S_ISSUES)
        for pod_name, issue_info in self.config.K8S_ISSUES.items():
            namespace = random.choice(self.config.K8S_NAMESPACES)
            template = random.choice(self.config.K8S_TRAINING_TEMPLATES)

            # Templates now use {pod} and {namespace}
            task = template.format(pod=pod_name, namespace=namespace)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['fix_action']}",
                    black_swan_type="DevOps/Pod_Eviction",
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
            extra = self.generate_cloudformation_scenarios(
                num_training=remaining_training, num_test=0
            )
            training_scenarios.extend(
                [s for s in extra if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS
        # ═══════════════════════════════════════════════════════════════════════

        cfn_test = max(1, int(total_test * 0.4))
        iam_test = max(1, int(total_test * 0.3))
        k8s_test = total_test - cfn_test - iam_test

        test_cfn = self.generate_cloudformation_scenarios(
            num_training=0, num_test=cfn_test
        )
        test_iam = self.generate_iam_scenarios(num_training=0, num_test=iam_test)
        test_k8s = self.generate_kubernetes_scenarios(num_training=0, num_test=k8s_test)

        test_scenarios.extend([s for s in test_cfn if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_iam if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_k8s if s.get("phase") == "test"])

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
            f"  📋 Coverage Guarantee: Training covers {len(training_error_codes)} error types"
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
        conditions_provider = DevOpsConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Domain-specific templates
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Deploy stack ({region})",
            "Scale service ({region})",
            "Update infrastructure ({region})",
            "Configure cluster ({region})",
            "Provision resources ({region})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Use STUCK_STACKS for testing, not generic stacks!
        # Generic stacks (web-app, api-server) are NOT enforced by MCP server.
        # Only STUCK_STACKS have per-stack recovery action rules.
        # ═══════════════════════════════════════════════════════════════════════
        from ..config import DevOpsConfig

        stuck_stacks = DevOpsConfig.STUCK_STACKS
        stacks = list(stuck_stacks.keys())  # prod-api-stack, data-pipeline-stack, etc.
        regions = (
            DevOpsConfig.REGIONS
            if hasattr(DevOpsConfig, "REGIONS")
            else ["us-east-1", "eu-west-1", "ap-southeast-1", "us-west-2"]
        )

        # Build reverse mapping: error_code -> stack
        error_code_to_stack = {}
        for stk, stk_info in stuck_stacks.items():
            error_code_to_stack[stk_info["error_code"]] = stk

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
            from ..config import DevOpsConfig

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
                solution = DevOpsConfig.get_valid_solution_for_conditions(base_key)
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
                    # CRITICAL: Always keep the stack's error code!
                    # This ensures the test uses the correct stack for enforcement.
                    # ═══════════════════════════════════════════════════════════
                    stack_error_codes = set(error_code_to_stack.keys())
                    stack_cond = None
                    for c in base_conditions:
                        if c in stack_error_codes:
                            stack_cond = c
                            break

                    other_conditions = [c for c in base_conditions if c != stack_cond]
                    # Ensure keep_count doesn't exceed available conditions
                    keep_count = min(
                        len(other_conditions), max(1, int(len(other_conditions) * 0.6))
                    )
                    replace_count = max(0, len(other_conditions) - keep_count)

                    kept_others = (
                        random.sample(other_conditions, keep_count)
                        if other_conditions
                        else []
                    )

                    if stack_cond:
                        kept_conditions = [stack_cond] + kept_others
                    else:
                        kept_conditions = kept_others

                    available_new = [
                        c for c in condition_codes if c not in kept_conditions
                    ]
                    # Only sample new conditions if we have some to replace
                    new_conditions = (
                        random.sample(
                            available_new, min(replace_count, len(available_new))
                        )
                        if replace_count > 0 and available_new
                        else []
                    )

                    all_conds = sorted(kept_conditions + new_conditions)
                    condition_key = "+".join(all_conds)
                    test_type = "novel"
            else:
                condition_key = base_key
                all_conds = base_conditions
                test_type = "exact_match"

            template = test_templates[i % len(test_templates)]

            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL: Determine stack from error_code in condition_key!
            # The condition_key contains the stack's error_code (e.g., CFN-881).
            # We must use the correct stack so MCP enforcement is accurate.
            # ═══════════════════════════════════════════════════════════════════
            stack = stacks[i % len(stacks)]  # Default fallback
            for cond in all_conds:
                if cond in error_code_to_stack:
                    stack = error_code_to_stack[cond]
                    break

            region = regions[i % len(regions)]

            # BLACK SWAN CSP: NO conditions in task - only region context
            task = template.format(region=region)

            num_conditions = len(all_conds)
            scenarios.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"DevOps/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
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
            "MATCHED (exact keys)"
            if mode == "matched"
            else "MIXED (50% exact + 50% novel)"
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
        Generate SEMANTIC compositional tests for DevOps domain where composite
        solutions ARE derivable from atomic precepts, enabling P₁ > 0%.

        DevOps semantic conditions map to deployment strategies:
        - Tier 3 (Highest): Security - non-negotiable
        - Tier 2 (Middle): Compliance/Regional requirements
        - Tier 1 (Lowest): Performance/Cost preferences

        Args:
            num_train: Base number of atomic conditions to train
            num_test: Number of composite test scenarios
            seed: Random seed for reproducibility
            beta: Repetitions per atomic condition
            filter_by_learned: If True, filter test composites by learned atoms
            test_num_conditions: Number of conditions to combine in test scenarios

        Returns:
            Tuple of (training_scenarios, test_scenarios, semantic_mappings)
        """
        if seed is not None:
            random.seed(seed)

        print(f"\n  🧠 DEVOPS SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(f"     Beta={beta} (each atomic condition trained {beta}x)")
        print(f"     Test complexity: {test_num_conditions}-way combinations")

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Solutions MUST match DevOpsConfig.MULTI_CONDITION_VALID_STRATEGIES
        # Valid strategies: continue_update_rollback, wait_for_completion
        # ═══════════════════════════════════════════════════════════════════════
        semantic_conditions = {
            # Tier 3 (Highest): Security - non-negotiable
            "SECURE": {
                "meaning": "Security-critical deployment requiring zero-downtime",
                "solution": "wait_for_completion",  # Wait strategy: safest
                "reasoning": "Security-critical deployments wait for completion (safe)",
                "tier": 3,
            },
            # Tier 2 (Middle): Compliance/Regional requirements
            "AUDIT": {
                "meaning": "Audit-compliant deployment with full traceability",
                "solution": "continue_update_rollback",  # Continue with rollback
                "reasoning": "Audit compliance uses continue_update_rollback (traceable)",
                "tier": 2,
            },
            "HIPAA": {
                "meaning": "HIPAA-compliant healthcare deployment",
                "solution": "wait_for_completion",  # Wait: no data exposure
                "reasoning": "HIPAA compliance uses wait_for_completion (no PHI exposure)",
                "tier": 2,
            },
            "PCI": {
                "meaning": "PCI-DSS compliant payment deployment",
                "solution": "wait_for_completion",  # Wait: instant rollback
                "reasoning": "PCI compliance uses wait_for_completion (instant rollback)",
                "tier": 2,
            },
            # Tier 1 (Lowest): Performance/Cost preferences
            "FAST": {
                "meaning": "Fast deployment with minimal downtime",
                "solution": "continue_update_rollback",  # Continue: quick rollout
                "reasoning": "Fast deployments use continue_update_rollback (quick)",
                "tier": 1,
            },
            "CHEAP": {
                "meaning": "Cost-optimized deployment minimizing resources",
                "solution": "continue_update_rollback",  # Continue: minimal resources
                "reasoning": "Cost-optimized uses continue_update_rollback (efficient)",
                "tier": 1,
            },
            "SCALE": {
                "meaning": "High-scale deployment for traffic spikes",
                "solution": "wait_for_completion",  # Wait: safe scaling
                "reasoning": "High-scale uses wait_for_completion (gradual)",
                "tier": 1,
            },
            "TEST": {
                "meaning": "Testing/staging deployment for validation",
                "solution": "continue_update_rollback",  # Continue: fast for tests
                "reasoning": "Test deployments use continue_update_rollback (simple)",
                "tier": 1,
            },
        }

        def compute_composite_solution(conditions: List[str]) -> str:
            """Compute composite solution using priority-based resolution."""
            if not conditions:
                return "continue_update_rollback"  # Default

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
            return "rolling"

        # Select conditions for training
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

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE TRAINING SCENARIOS (Single conditions - atomic learning)
        # ═══════════════════════════════════════════════════════════════════════
        training_scenarios = []
        task_templates = [
            "Deploy service to {env} environment",
            "Release application to {env}",
            "Push update to {env} cluster",
            "Deploy microservice to {env}",
        ]
        environments = ["production", "staging", "dev", "qa", "uat", "perf"]

        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]
                template = task_templates[scenario_idx % len(task_templates)]
                env = environments[scenario_idx % len(environments)]

                scenario = {
                    "task": template.format(env=env),
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

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE TEST SCENARIOS (Composite conditions - derivable solutions)
        # ═══════════════════════════════════════════════════════════════════════
        from itertools import combinations

        num_atoms_for_combos = min(6, len(train_conditions))
        if num_atoms_for_combos < test_num_conditions:
            print(f"     ⚠️ Adjusting to {num_atoms_for_combos}-way combos")
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
            env = environments[i % len(environments)]

            derivation_parts = [f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})" for c in combo]
            derivation_rule = f"{' vs '.join(derivation_parts)} → {winning_cond} wins → {composite_solution}"

            scenario = {
                "task": template.format(env=env),
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

        # Build semantic mappings
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


def generate_devops_scenarios(
    num_samples: int = 10,
    train_ratio: float = 0.6,
    include_generator_samples: bool = False,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate DevOps black swan scenarios.

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
    generator = DevOpsScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
