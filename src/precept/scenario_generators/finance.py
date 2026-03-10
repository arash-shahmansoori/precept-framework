"""
Finance Scenario Generator for PRECEPT Testing.

This module generates finance/trading black swan scenarios using template-based variations.
Supports configurable num_samples and train_ratio for flexible train/test splits.

Configuration is imported from precept.config.finance - single source of truth.

Usage:
    from precept.scenario_generators import FinanceScenarioGenerator

    generator = FinanceScenarioGenerator(num_samples=20, train_ratio=0.6)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import FinanceConfig
from ..config.multi_condition import (
    FinanceConditions,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.finance
FinanceScenarioConfig = FinanceConfig


class FinanceScenarioGenerator:
    """
    Generate finance/trading black swan scenarios using template-based variations.

    COHERENCE GUARANTEE: Each scenario maintains semantic consistency:
    - Volatile symbols use correct error codes and order type solutions
    - Data sources have appropriate mitigation strategies
    - Compliance blocks have proper resolution paths

    Usage:
        generator = FinanceScenarioGenerator(num_samples=20, train_ratio=0.6)
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
        self.finance_traps = BLACK_SWAN_DEFINITIONS.get("Finance", {})
        self.config = FinanceScenarioConfig

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

        for trap_name, trap_def in self.finance_traps.items():
            sample = self.generator.generate_sample(
                category="Finance",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"Finance/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios[:num_samples]

    def generate_trading_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate trading/volatility scenarios with template-based variations.

        COHERENCE GUARANTEE: Each symbol always uses:
        - Its own error_code (FIX-REJECT-58)
        - Its own working_order_type (limit)
        - Its own volatility_reason
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

        for symbol, symbol_info in self.config.VOLATILE_SYMBOLS.items():
            for context_key, context_info in self.config.ORDER_CONTEXTS.items():
                for quantity in self.config.QUANTITIES:
                    for template in self.config.TRADING_TRAINING_TEMPLATES:
                        all_training_combos.append(
                            (
                                symbol,
                                symbol_info,
                                context_key,
                                context_info,
                                quantity,
                                template,
                            )
                        )
                    for template in self.config.TRADING_TEST_TEMPLATES:
                        all_test_combos.append(
                            (
                                symbol,
                                symbol_info,
                                context_key,
                                context_info,
                                quantity,
                                template,
                            )
                        )

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for (
            symbol,
            symbol_info,
            context_key,
            context_info,
            quantity,
            template,
        ) in sampled_training:
            task = template.format(
                symbol=symbol,
                context=context_info["prefix"],
                quantity=quantity,
            )
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{symbol_info['error_code']} → {symbol_info['volatility_reason']}",
                    black_swan_type="Finance/Volatility_Reject",
                    precept_lesson=symbol_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for (
            symbol,
            symbol_info,
            context_key,
            context_info,
            quantity,
            template,
        ) in sampled_test:
            task = template.format(
                symbol=symbol,
                context=context_info["prefix"],
                quantity=quantity,
            )
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {symbol} → {symbol_info['working_order_type']} order (1 step)",
                    black_swan_type="Finance/Volatility_Reject",
                    precept_lesson=f"PRECEPT uses {symbol_info['working_order_type']} order (learned)",
                    phase="test",
                    tests_learning=symbol,
                )
            )

        return training + test_variations

    def generate_data_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate stale data/feed scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Build combinations
        all_training = []
        all_test = []

        for source, source_info in self.config.STALE_DATA_SOURCES.items():
            for template in self.config.DATA_TRAINING_TEMPLATES:
                all_training.append((source, source_info, template))
            for template in self.config.DATA_TEST_TEMPLATES:
                all_test.append((source, source_info, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for source, source_info, template in sampled_training:
            task = template.format(source=source.replace("_", " ").title())
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{source_info['error_code']} → {source_info['gap_reason']}",
                    black_swan_type="Finance/Stale_Data",
                    precept_lesson=source_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for source, source_info, template in sampled_test:
            task = template.format(source=source.replace("_", " ").title())
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {source_info['mitigation']} (1 step)",
                    black_swan_type="Finance/Stale_Data",
                    precept_lesson=f"PRECEPT applies {source_info['mitigation']} (learned)",
                    phase="test",
                    tests_learning=source,
                )
            )

        return training + test_variations

    def generate_compliance_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate compliance block scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Use a subset of symbols for compliance scenarios
        compliance_symbols = ["GME", "AMC", "BTC-USD"]

        # Build combinations
        all_training = []
        all_test = []

        for block_type, block_info in self.config.COMPLIANCE_BLOCKS.items():
            for symbol in compliance_symbols:
                for template in self.config.COMPLIANCE_TRAINING_TEMPLATES:
                    all_training.append((block_type, block_info, symbol, template))
                for template in self.config.COMPLIANCE_TEST_TEMPLATES:
                    all_test.append((block_type, block_info, symbol, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for block_type, block_info, symbol, template in sampled_training:
            task = template.format(symbol=symbol)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{block_info['error_code']} → {block_info['block_reason']}",
                    black_swan_type="Finance/Compliance_Block",
                    precept_lesson=block_info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for block_type, block_info, symbol, template in sampled_test:
            task = template.format(symbol=symbol)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {block_info['resolution']} (1 step)",
                    black_swan_type="Finance/Compliance_Block",
                    precept_lesson=f"PRECEPT applies {block_info['resolution']} (learned)",
                    phase="test",
                    tests_learning=block_type,
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

        This is the "Fleet Learning" pattern (Trading Systems):
        - Training: Strategy X + Market Condition Y → learns Rule Z
        - Testing: Different Strategy (Entity K) + SAME Condition Y → applies Rule Z

        GUARANTEED PATTERN:
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - GME + Market Order (FIN-058) → learns "FIN-058 → limit orders"

        Testing (DIFFERENT trading strategies, SAME market conditions):
          - GME + Different Quantity + Market Order (FIN-058) → applies "FIN-058 → limit" ✓
          - GME + Different Account + Market Order (FIN-058) → applies "FIN-058 → limit" ✓
        ═══════════════════════════════════════════════════════════════════════════

        Key insight: Rules are learned by ERROR CODE (condition), not by
        specific strategy/quantity (entity). This enables cross-entity transfer.
        Pattern learned once applies across all trading strategies.
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Get volatile symbols and order contexts
        volatile_symbols = self.config.VOLATILE_SYMBOLS
        order_contexts = list(self.config.ORDER_CONTEXTS.keys())

        # For each volatile symbol, create:
        # - Training: One order context that learns the rule
        # - Testing: Different order contexts that apply the rule

        for symbol, symbol_info in volatile_symbols.items():
            error_code = symbol_info["error_code"]
            working_order = symbol_info["working_order_type"]
            asset_class = symbol_info.get("asset_class", "equity")

            if len(order_contexts) < 2:
                continue

            # Shuffle contexts
            context_list = order_contexts.copy()
            random.shuffle(context_list)

            # TRAINING: First context learns the rule
            training_context = context_list[0]
            context_info = self.config.ORDER_CONTEXTS[training_context]

            training.append(
                {
                    "task": f"Execute market order for {symbol} ({context_info.get('prefix', training_context)})",
                    "expected": f"{error_code} → use {working_order} orders",
                    "black_swan_type": "Finance/FleetLearning_Train",
                    "precept_lesson": f"When {symbol} ({asset_class}) fails with {error_code}, use {working_order} for ANY strategy",
                    "phase": "training",
                    "fleet_learning": {
                        "symbol": symbol,
                        "error_code": error_code,
                        "learned_solution": working_order,
                        "training_context": training_context,
                    },
                }
            )

            # TESTING: Different contexts apply the SAME rule
            for test_context in context_list[1:3]:
                test_context_info = self.config.ORDER_CONTEXTS[test_context]

                testing.append(
                    {
                        "task": f"Purchase {symbol} ({test_context_info.get('prefix', test_context)}) with market order",
                        "expected": f"Apply learned rule: {error_code} → {working_order}",
                        "black_swan_type": "Finance/FleetLearning_Test",
                        "precept_lesson": f"Cross-entity transfer: Rule for {error_code} applies to {test_context}",
                        "phase": "test",
                        "tests_learning": f"fleet_learning_{error_code}",
                        "fleet_learning": {
                            "symbol": symbol,
                            "error_code": error_code,
                            "expected_solution": working_order,
                            "different_context": test_context,
                            "training_context": training_context,
                        },
                    }
                )

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(
            f"  🚀 Fleet Learning (Finance): {len(training)} train + {len(testing)} test scenarios"
        )
        print(
            "     Pattern: Symbol + Error Y → Rule Z | Symbol + Different Strategy + Error Y → Apply Z"
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
            GME + FIN-058 + ORD-LRG + TM-AFTER → limit order with smaller chunks

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

        conditions_provider = FinanceConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        volatile_symbols = self.config.VOLATILE_SYMBOLS
        order_contexts = (
            list(self.config.ORDER_CONTEXTS.keys())
            if hasattr(self.config, "ORDER_CONTEXTS")
            else ["market", "limit", "stop_loss", "day_trade"]
        )

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATES: NO SYMBOL NAMES - matches Logistics approach!
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Removing symbol names prevents ExpeL from using them as
        # retrieval anchors. This ensures fair comparison where only
        # condition_key determines the solution (like Logistics).
        #
        # Before: "Trade GME (swing_trade)" → ExpeL retrieves by "GME"
        # After:  "Execute order (swing_trade)" → Must use condition_key
        # ═══════════════════════════════════════════════════════════════════════
        training_templates = [
            "Execute order ({context})",
            "Place trade ({context})",
            "Submit order request ({context})",
        ]
        test_templates = [
            "Process order ({context})",
            "Execute trade request ({context})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        # Instead of random sampling, pre-generate K unique composite keys and
        # reuse them with β coverage. This mirrors logistics approach.
        #
        # Formula: β = num_training / K
        # Example: K=7 symbols, train=21 → β=3 (each key seen 3 times)
        # ═══════════════════════════════════════════════════════════════════════

        # K = number of unique rule types = number of volatile symbols
        K = len(volatile_symbols)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per volatile symbol
        fixed_key_pool = {}
        for symbol, symbol_info in volatile_symbols.items():
            error_code = symbol_info["error_code"]
            working_order = symbol_info["working_order_type"]

            # Sample (num_conditions - 1) OTHER conditions deterministically per symbol
            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            fixed_key_pool[symbol] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": working_order,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD TRAINING COMBOS USING FIXED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        all_training_combos = []
        all_test_combos = []

        for symbol, symbol_info in volatile_symbols.items():
            error_code = symbol_info["error_code"]
            working_order = symbol_info["working_order_type"]

            # USE FIXED KEY from pool
            key_info = fixed_key_pool[symbol]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]

            for context in order_contexts:
                for template in training_templates:
                    all_training_combos.append(
                        {
                            "symbol": symbol,
                            "error_code": error_code,
                            "working_order": working_order,
                            "context": context,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
                        }
                    )

                for template in test_templates:
                    all_test_combos.append(
                        {
                            "symbol": symbol,
                            "error_code": error_code,
                            "working_order": working_order,
                            "context": context,
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
            symbol = combo["symbol"]
            context = combo["context"]
            template = combo["template"]
            working_order = combo["working_order"]
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            # ═══════════════════════════════════════════════════════════════════
            # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
            # ═══════════════════════════════════════════════════════════════════
            # Like Logistics, the task description should NOT reveal the conditions
            # that determine the solution. This prevents ExpeL from using vector
            # similarity on condition codes.
            #
            # PRECEPT uses condition_key via TIER 1 O(1) lookup (from parameters)
            # ExpeL can only use task description for similarity (which won't help)
            # ═══════════════════════════════════════════════════════════════════
            task = template.format(context=context)

            training.append(
                {
                    "task": task,
                    "expected": f"{condition_key} → {working_order}",
                    "black_swan_type": f"Finance/MultiCondition_{num_conditions}C_Train",
                    "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {working_order}",
                    "phase": "training",
                    "condition_key": condition_key,
                    "test_mode": test_mode,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "symbol": symbol,  # Keep for internal tracking
                        "solution": working_order,
                    },
                }
            )

            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "symbol": symbol,
                "solution": working_order,
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

            symbol = combo["symbol"]
            context = combo["context"]
            template = combo["template"]

            matching_keys = [
                (k, v)
                for k, v in training_condition_keys.items()
                if v["symbol"] == symbol
            ]
            if not matching_keys:
                continue

            condition_key, key_info = random.choice(matching_keys)
            all_conds = key_info["conditions"]
            solution = key_info["solution"]

            cond_str = " + ".join(all_conds)
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(context=context)

            testing.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Finance/MultiCondition_{num_conditions}C_Test",
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
        Generate all finance scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN: Single-condition (num_conditions=1) is just a special case!

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, guarantees training covers ALL error types
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default)
                           - num_conditions>1: Multi-condition (for ablation)

        Returns:
            Combined list of all finance scenarios
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
        1. Training includes at least one scenario for EACH volatile symbol
        2. Training includes at least one scenario for EACH stale data source
        3. Training includes at least one scenario for EACH compliance block type
        4. Test scenarios will ALWAYS have a corresponding learned rule

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. Volatile symbols - one per symbol
        for symbol, symbol_info in self.config.VOLATILE_SYMBOLS.items():
            context_key = random.choice(list(self.config.ORDER_CONTEXTS.keys()))
            context_info = self.config.ORDER_CONTEXTS[context_key]
            quantity = random.choice(self.config.QUANTITIES)
            template = random.choice(self.config.TRADING_TRAINING_TEMPLATES)

            task = template.format(
                symbol=symbol,
                context=context_info["prefix"],
                quantity=quantity,
            )

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{symbol_info['error_code']} → {symbol_info['volatility_reason']}",
                    black_swan_type="Finance/Volatility_Reject",
                    precept_lesson=symbol_info["lesson"],
                    phase="training",
                )
            )

        # 1b. Stale data sources - one per source
        for source, source_info in self.config.STALE_DATA_SOURCES.items():
            template = random.choice(self.config.DATA_TRAINING_TEMPLATES)
            task = template.format(source=source.replace("_", " ").title())

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{source_info['error_code']} → {source_info['gap_reason']}",
                    black_swan_type="Finance/Stale_Data",
                    precept_lesson=source_info["lesson"],
                    phase="training",
                )
            )

        # 1c. Compliance blocks - one per block type
        compliance_symbols = ["GME", "AMC", "BTC-USD"]
        for block_type, block_info in self.config.COMPLIANCE_BLOCKS.items():
            symbol = random.choice(compliance_symbols)
            template = random.choice(self.config.COMPLIANCE_TRAINING_TEMPLATES)
            task = template.format(symbol=symbol)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{block_info['error_code']} → {block_info['block_reason']}",
                    black_swan_type="Finance/Compliance_Block",
                    precept_lesson=block_info["lesson"],
                    phase="training",
                )
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: FILL REMAINING TRAINING SLOTS (if any)
        # ═══════════════════════════════════════════════════════════════════════

        mandatory_count = len(training_scenarios)
        remaining_training = max(0, total_training - mandatory_count)

        if remaining_training > 0:
            extra = self.generate_trading_scenarios(
                num_training=remaining_training, num_test=0
            )
            training_scenarios.extend(
                [s for s in extra if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS (evenly distributed)
        # ═══════════════════════════════════════════════════════════════════════

        trade_test = max(1, int(total_test * 0.5))
        data_test = max(1, int(total_test * 0.25))
        comp_test = total_test - trade_test - data_test

        test_trade = self.generate_trading_scenarios(
            num_training=0, num_test=trade_test
        )
        test_data = self.generate_data_scenarios(num_training=0, num_test=data_test)
        test_comp = self.generate_compliance_scenarios(
            num_training=0, num_test=comp_test
        )

        test_scenarios.extend([s for s in test_trade if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_data if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_comp if s.get("phase") == "test"])

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

        IMPORTANT FOR MATCHED MODE:
        - learned_rule_keys should be PRE-FILTERED to only include keys from the
          current training session's fixed key pool.

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

        # ═══════════════════════════════════════════════════════════════════════
        # COLD START SUPPORT: Allow testing with keys but no learned rules
        # ═══════════════════════════════════════════════════════════════════════
        if mode == "matched" and not learned_rule_keys and not all_training_keys:
            print("  ⚠️ No learned rules or training keys for MATCHED mode")
            return []

        # ═══════════════════════════════════════════════════════════════════════
        # VALIDATION: For MATCHED mode, ensure learned keys are in training pool
        # ═══════════════════════════════════════════════════════════════════════
        if mode == "matched" and all_training_keys and learned_rule_keys:
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

        # Initialize conditions provider
        conditions_provider = FinanceConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Domain-specific templates
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # ═══════════════════════════════════════════════════════════════════════
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # PRECEPT uses TIER 1 (O(1) hash lookup) on condition_key.
        # ExpeL can only use task description similarity (which won't help).
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Execute trade ({context})",
            "Process order ({context})",
            "Submit market order ({context})",
            "Handle transaction ({context})",
            "Complete purchase ({context})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Use VOLATILE symbols for testing, not generic ones!
        # Generic symbols (AAPL, GOOGL) are NOT enforced by MCP server.
        # Only volatile symbols (GME, BTC-USD) have per-symbol order type rules.
        # ═══════════════════════════════════════════════════════════════════════
        from ..config import FinanceConfig

        volatile_symbols = FinanceConfig.VOLATILE_SYMBOLS
        symbols = list(volatile_symbols.keys())  # GME, BTC-USD, etc.

        # Build reverse mapping: error_code -> symbol
        # This lets us determine which symbol a condition_key corresponds to
        error_code_to_symbol = {}
        for sym, sym_info in volatile_symbols.items():
            error_code_to_symbol[sym_info["error_code"]] = sym

        # ═══════════════════════════════════════════════════════════════════════
        # KEY SELECTION STRATEGY - FIXED FOR PROPER CROSS-EPISODE LEARNING TEST
        # ═══════════════════════════════════════════════════════════════════════
        learned_keys_list = list(learned_rule_keys.keys())
        random.shuffle(learned_keys_list)

        if mode == "matched":
            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Use ALL TRAINING KEYS, not just learned ones!
            # This ensures we test PRECEPT's ability to learn during testing.
            # Keys that failed during training will require cross-episode learning.
            # ═══════════════════════════════════════════════════════════════════
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
            print(
                f"     Using {len(learned_keys_list)} learned keys (cycling if needed)"
            )

        for i in range(num_test):
            # Cycle through base keys (may include unlearned keys in MATCHED mode)
            base_key = base_keys_list[i % len(base_keys_list)]
            base_conditions = base_key.split("+")

            # ═══════════════════════════════════════════════════════════════════
            # Get solution - may not exist in learned_rule_keys for unlearned keys!
            # For unlearned keys, compute expected solution from hash-based config.
            # ═══════════════════════════════════════════════════════════════════
            from ..config import FinanceConfig

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
                solution = FinanceConfig.get_valid_order_type_for_conditions(base_key)
                is_learned = False

            # Track test type for analysis
            test_type = "exact_match"

            if mode == "matched":
                condition_key = base_key
                all_conds = base_conditions
                test_type = "exact_match"
            elif mode == "random":
                # ═══════════════════════════════════════════════════════════════
                # RANDOM (MIXED): GUARANTEED 50% exact match + 50% novel
                # ═══════════════════════════════════════════════════════════════
                num_exact_match = num_test // 2

                if i < num_exact_match:
                    # First 50%: Use EXACT key from learned_rule_keys (tests Tier 1)
                    # GUARANTEED to match because base_keys_list = learned_keys_list
                    condition_key = base_key
                    all_conds = base_conditions
                    test_type = "exact_match"
                else:
                    # Second 50%: Generate novel key with 60% overlap (tests Tier 2/3)
                    # ═══════════════════════════════════════════════════════════
                    # CRITICAL: Always keep the symbol's error code!
                    # This ensures the test uses the correct symbol for enforcement.
                    # ═══════════════════════════════════════════════════════════
                    # First, identify which condition is the symbol error code
                    symbol_error_codes = set(error_code_to_symbol.keys())
                    symbol_cond = None
                    for c in base_conditions:
                        if c in symbol_error_codes:
                            symbol_cond = c
                            break

                    # Keep the symbol condition + 60% of other conditions
                    other_conditions = [c for c in base_conditions if c != symbol_cond]
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

                    # Always keep symbol condition if it exists
                    if symbol_cond:
                        kept_conditions = [symbol_cond] + kept_others
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
            # CRITICAL: Determine symbol from error_code in condition_key!
            # The condition_key contains the symbol's error_code (e.g., FIN-058).
            # We must use the correct symbol so MCP enforcement is accurate.
            # ═══════════════════════════════════════════════════════════════════
            symbol = symbols[i % len(symbols)]  # Default fallback
            for cond in all_conds:
                if cond in error_code_to_symbol:
                    symbol = error_code_to_symbol[cond]
                    break

            # Use generic context (like generate_multi_condition_scenarios)
            contexts = ["swing_trade", "day_trade", "position_trade", "momentum_trade"]
            context = contexts[i % len(contexts)]

            # BLACK SWAN CSP: NO conditions in task - only context
            # The condition_key is passed in multi_condition metadata
            task = template.format(context=context)

            num_conditions = len(all_conds)
            scenarios.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Finance/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
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
        Generate SEMANTIC compositional tests for Finance domain where composite
        solutions ARE derivable from atomic precepts, enabling P₁ > 0%.

        Finance semantic conditions map to order types:
        - Tier 3 (Highest): Risk management - non-negotiable
        - Tier 2 (Middle): Compliance requirements
        - Tier 1 (Lowest): Performance preferences

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

        print(f"\n  🧠 FINANCE SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(f"     Beta={beta} (each atomic condition trained {beta}x)")
        print(f"     Test complexity: {test_num_conditions}-way combinations")

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Solutions MUST match FinanceConfig.MULTI_CONDITION_VALID_ORDER_TYPES
        # Valid order types: limit, stop, market, bracket
        # ═══════════════════════════════════════════════════════════════════════
        semantic_conditions = {
            # Tier 3 (Highest): Risk management - non-negotiable
            "RISK": {
                "meaning": "Risk-managed order with price protection",
                "solution": "stop",  # Stop order: price protection
                "reasoning": "Risk management requires stop orders (price protection)",
                "tier": 3,
            },
            # Tier 2 (Middle): Compliance requirements
            "COMPLY": {
                "meaning": "Compliance-required order with audit trail",
                "solution": "limit",  # Limit: full audit trail
                "reasoning": "Compliance requires limit orders (full audit trail)",
                "tier": 2,
            },
            "AUDIT": {
                "meaning": "Auditable order execution",
                "solution": "limit",  # Limit: traceable execution
                "reasoning": "Audit requirements use limit (traceable execution)",
                "tier": 2,
            },
            "HEDGE": {
                "meaning": "Hedging position against market moves",
                "solution": "bracket",  # Bracket: triggered protection
                "reasoning": "Hedging uses bracket orders (triggered protection)",
                "tier": 2,
            },
            # Tier 1 (Lowest): Performance preferences
            "SPEED": {
                "meaning": "Fast execution priority",
                "solution": "market",  # Market: instant fill
                "reasoning": "Speed priority uses market orders (instant fill)",
                "tier": 1,
            },
            "COST": {
                "meaning": "Cost-optimized execution",
                "solution": "limit",  # Limit: price control
                "reasoning": "Cost optimization uses limit (price control)",
                "tier": 1,
            },
            "VOLUME": {
                "meaning": "High-volume order execution",
                "solution": "bracket",  # Bracket: staged execution
                "reasoning": "Volume orders use bracket (staged execution)",
                "tier": 1,
            },
            "STEALTH": {
                "meaning": "Hidden order to minimize market impact",
                "solution": "stop",  # Stop: delayed execution
                "reasoning": "Stealth orders use stop (delayed execution)",
                "tier": 1,
            },
        }

        def compute_composite_solution(conditions: List[str]) -> str:
            """Compute composite solution using priority-based resolution."""
            if not conditions:
                return "limit"

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
            return "limit"

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

        # Generate training scenarios
        training_scenarios = []
        task_templates = [
            "Execute trade for {symbol}",
            "Place order for {symbol}",
            "Submit trade request for {symbol}",
            "Process order for {symbol}",
        ]
        # BUGFIX: Only use symbols from FinanceConfig.KNOWN_SYMBOLS!
        # GOOGL, AMZN, META are NOT in KNOWN_SYMBOLS, causing parse_task
        # to default to "AAPL" regardless of what's in the task text.
        symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]

        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]
                template = task_templates[scenario_idx % len(task_templates)]
                symbol = symbols[scenario_idx % len(symbols)]

                scenario = {
                    "task": template.format(symbol=symbol),
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

        # Generate test scenarios
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
            symbol = symbols[i % len(symbols)]

            derivation_parts = [f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})" for c in combo]
            derivation_rule = f"{' vs '.join(derivation_parts)} → {winning_cond} wins → {composite_solution}"

            scenario = {
                "task": template.format(symbol=symbol),
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


def generate_finance_scenarios(
    num_samples: int = 10,
    train_ratio: float = 0.6,
    include_generator_samples: bool = False,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate finance black swan scenarios.

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
    generator = FinanceScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
