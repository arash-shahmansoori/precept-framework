"""
Logistics Scenario Generator for PRECEPT Testing.

This module generates logistics black swan scenarios using template-based variations.
Supports configurable num_samples and train_ratio for flexible train/test splits.

Configuration is imported from precept.config.logistics - single source of truth.

Usage:
    from precept.scenario_generators import LogisticsScenarioGenerator

    generator = LogisticsScenarioGenerator(num_samples=20, train_ratio=0.6)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import LogisticsConfig
from ..config.multi_condition import (
    LogisticsConditions,
    MultiConditionConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.logistics
LogisticsScenarioConfig = LogisticsConfig


class LogisticsScenarioGenerator:
    """
    Generate logistics black swan scenarios using template-based variations.

    COHERENCE GUARANTEE: Each scenario maintains semantic consistency:
    - Origin port failures use the correct error codes
    - Working alternatives match the blocked port's actual alternatives
    - Lessons are specific to the port/route combination

    Usage:
        generator = LogisticsScenarioGenerator(num_samples=20, train_ratio=0.6)
        scenarios = generator.generate_all()
    """

    def __init__(self, num_samples: int = 20, train_ratio: float = 0.6):
        """
        Initialize the generator.

        Args:
            num_samples: TOTAL number of scenarios (train + test combined)
            train_ratio: Ratio of training samples (0.0 to 1.0)
                        Default: 0.6 (60% training, 40% test)
        """
        self.num_samples = num_samples
        self.train_ratio = max(0.1, min(0.9, train_ratio))
        self.generator = UniversalDataGenerator(num_samples=num_samples)
        self.logistics_traps = BLACK_SWAN_DEFINITIONS.get("Logistics", {})
        self.config = LogisticsScenarioConfig

    def _build_scenario(
        self,
        task: str,
        expected: str,
        black_swan_type: str,
        precept_lesson: str,
        phase: str = None,
        tests_learning: str = None,
        condition_key: str = None,
        test_mode: str = None,
        **kwargs,
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
        if condition_key:
            scenario["condition_key"] = condition_key
        if test_mode:
            scenario["test_mode"] = test_mode
        # Add any additional kwargs
        scenario.update(kwargs)
        return scenario

    def generate_from_universal_generator(self, num_samples: int = 5) -> List[Dict]:
        """Generate scenarios using UniversalDataGenerator."""
        scenarios = []

        for trap_name, trap_def in self.logistics_traps.items():
            sample = self.generator.generate_sample(
                category="Logistics",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"Logistics/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios[:num_samples]

    def generate_port_closure_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 1,
    ) -> List[Dict]:
        """
        Generate port closure scenarios with template-based variations.

        COHERENCE GUARANTEE: Each port (e.g., rotterdam) always uses:
        - Its own error_code (R-482)
        - Its own working_alternatives (hamburg, antwerp)
        - Its own lesson text

        Random sampling varies:
        - Which port/destination combinations are selected
        - Which template phrasing is used
        - Which cargo type is applied

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)

        Returns:
            List of port closure scenarios with training/test phases
        """
        # Calculate defaults based on num_samples and train_ratio
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training
        num_training = num_training if num_training is not None else total_training
        num_test = num_test if num_test is not None else total_test

        training = []
        test_variations = []

        # Build all possible COHERENT combinations
        all_training_combos = []
        all_test_combos = []

        for port, port_info in self.config.BLOCKED_PORTS.items():
            # Get valid destinations for this port
            blocked_dests = port_info.get("blocked_destinations", [])

            for dest, dest_info in self.config.DESTINATIONS.items():
                # ═══════════════════════════════════════════════════════════════
                # VALIDATION: Skip nonsensical origin == destination scenarios
                # e.g., "Shanghai to Shanghai" doesn't make real-world sense
                # ═══════════════════════════════════════════════════════════════
                if port.lower() == dest.lower():
                    continue  # Skip same origin/destination

                # Skip if this destination is specifically blocked
                if dest_info["region"] in blocked_dests:
                    # This is a SPECIAL case - Hamburg blocks US destinations
                    for cargo_type, cargo_info in self.config.CARGO_TYPES.items():
                        for template in self.config.TRAINING_TEMPLATES:
                            all_training_combos.append(
                                (
                                    port,
                                    port_info,
                                    dest,
                                    dest_info,
                                    cargo_type,
                                    cargo_info,
                                    template,
                                    True,
                                )
                            )
                        for template in self.config.TEST_TEMPLATES:
                            all_test_combos.append(
                                (
                                    port,
                                    port_info,
                                    dest,
                                    dest_info,
                                    cargo_type,
                                    cargo_info,
                                    template,
                                    True,
                                )
                            )
                else:
                    # ═══════════════════════════════════════════════════════════
                    # SKIP: If port has blocked_destinations, ONLY use those
                    # ═══════════════════════════════════════════════════════════
                    # This prevents creating scenarios like Hamburg→London where
                    # H-903 won't fire (since London is not a US destination).
                    # We only want to create training scenarios that WILL trigger
                    # the expected error, allowing PRECEPT to learn the solution.
                    if blocked_dests:
                        # Skip destinations that won't trigger the error
                        continue

                    # Normal blocked port scenario (no region restrictions)
                    for cargo_type, cargo_info in self.config.CARGO_TYPES.items():
                        for template in self.config.TRAINING_TEMPLATES:
                            all_training_combos.append(
                                (
                                    port,
                                    port_info,
                                    dest,
                                    dest_info,
                                    cargo_type,
                                    cargo_info,
                                    template,
                                    False,
                                )
                            )
                        for template in self.config.TEST_TEMPLATES:
                            all_test_combos.append(
                                (
                                    port,
                                    port_info,
                                    dest,
                                    dest_info,
                                    cargo_type,
                                    cargo_info,
                                    template,
                                    False,
                                )
                            )

        # Initialize condition generator for multi-condition scenarios
        logistics_conditions = LogisticsConditions()

        # Randomly sample COHERENT combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for (
            port,
            port_info,
            dest,
            dest_info,
            cargo_type,
            cargo_info,
            template,
            is_special,
        ) in sampled_training:
            cargo_prefix = cargo_info["prefix"] or "Standard"
            error_code = port_info["error_code"]

            # Generate multi-condition key if num_conditions > 1
            if num_conditions > 1:
                additional_conditions = logistics_conditions.get_random_conditions(
                    n=num_conditions - 1
                )
                all_conditions = [error_code] + additional_conditions
                all_conditions.sort()
                condition_key = "+".join(all_conditions)
                condition_str = " + ".join(all_conditions)
                condition_tag = f" [Conditions: {condition_str}]"
            else:
                condition_key = error_code
                condition_tag = ""

            task = (
                template.format(
                    origin=port.replace("_", " ").title(),
                    destination=dest.replace("_", " ").title(),
                    cargo_prefix=cargo_prefix,
                )
                + condition_tag
            )

            working_alt = port_info["working_alternatives"][0]
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{condition_key} → {port_info['block_reason']}",
                    black_swan_type="Logistics/Port_Closure",
                    precept_lesson=port_info["lesson"],
                    phase="training",
                    condition_key=condition_key,
                )
            )

        # Randomly sample COHERENT combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for (
            port,
            port_info,
            dest,
            dest_info,
            cargo_type,
            cargo_info,
            template,
            is_special,
        ) in sampled_test:
            cargo_prefix = cargo_info["prefix"] or "Standard"
            error_code = port_info["error_code"]

            # Generate multi-condition key if num_conditions > 1
            if num_conditions > 1:
                additional_conditions = logistics_conditions.get_random_conditions(
                    n=num_conditions - 1
                )
                all_conditions = [error_code] + additional_conditions
                all_conditions.sort()
                condition_key = "+".join(all_conditions)
                condition_str = " + ".join(all_conditions)
                condition_tag = f" [Conditions: {condition_str}]"
            else:
                condition_key = error_code
                condition_tag = ""

            task = (
                template.format(
                    origin=port.replace("_", " ").title(),
                    destination=dest.replace("_", " ").title(),
                    cargo_prefix=cargo_prefix,
                )
                + condition_tag
            )

            working_alt = port_info["working_alternatives"][0]
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: {condition_key} → {working_alt} (1 step)",
                    black_swan_type="Logistics/Port_Closure",
                    precept_lesson=f"PRECEPT skips {port}, uses {working_alt} (learned)",
                    phase="test",
                    tests_learning=condition_key,
                    condition_key=condition_key,
                )
            )

        return training + test_variations

    def generate_customs_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 1,
        test_mode: str = "matched",
    ) -> List[Dict]:
        """
        Generate customs/documentation scenarios with template-based variations.

        These scenarios involve documentation issues that cause shipment delays.

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)
            test_mode: Testing mode - controls how test condition keys are generated:
                - "matched": Test scenarios REUSE condition keys from training (O(1) lookup test)
                - "random": Test scenarios generate NEW random condition keys (generalization test)
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        # Use configuration - no hardcoded values
        customs_issues = self.config.CUSTOMS_ISSUES
        training_templates = self.config.CUSTOMS_TRAINING_TEMPLATES
        test_templates = self.config.CUSTOMS_TEST_TEMPLATES
        destination_to_issue = self.config.DESTINATION_TO_CUSTOMS_ISSUE

        # Initialize condition generator for multi-condition scenarios
        logistics_conditions = LogisticsConditions()

        # Build combinations - USE CORRECT ISSUE TYPE for each destination
        all_training = []
        all_test = []

        for dest_lower, issue in destination_to_issue.items():
            dest = dest_lower.replace("_", " ").title()
            info = customs_issues[issue]

            for template in training_templates:
                all_training.append((issue, info, dest, template))
            for template in test_templates:
                all_test.append((issue, info, dest, template))

        training = []
        test_variations = []
        training_condition_keys = []  # Store training keys for matched testing

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for issue, info, dest, template in sampled_training:
            task = template.format(destination=dest)
            error_code = info["error_code"]

            # Generate multi-condition key if num_conditions > 1
            if num_conditions > 1:
                # Get additional conditions from logistics conditions
                additional_conditions = logistics_conditions.get_random_conditions(
                    n=num_conditions - 1
                )
                all_conditions = [error_code] + additional_conditions
                all_conditions.sort()  # Deterministic ordering
                condition_key = "+".join(all_conditions)
                condition_str = " + ".join(all_conditions)
                condition_tag = f" [Conditions: {condition_str}]"
            else:
                condition_key = error_code
                condition_tag = ""

            # Store for matched testing
            training_condition_keys.append(
                {
                    "condition_key": condition_key,
                    "condition_tag": condition_tag,
                    "info": info,
                }
            )

            training.append(
                self._build_scenario(
                    task=task + condition_tag,
                    expected=f"{condition_key} → {info['solution']}",
                    black_swan_type="Logistics/Customs_Hold",
                    precept_lesson=info["lesson"],
                    phase="training",
                    condition_key=condition_key,
                )
            )

        # Sample test scenarios
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for idx, (issue, info, dest, template) in enumerate(sampled_test):
            task = template.format(destination=dest)
            error_code = info["error_code"]

            # ═══════════════════════════════════════════════════════════════════
            # TEST MODE: Controls how condition keys are generated
            # ═══════════════════════════════════════════════════════════════════
            if test_mode == "matched" and training_condition_keys:
                # MATCHED: Reuse a training condition key (tests O(1) exact lookup)
                training_key_info = training_condition_keys[
                    idx % len(training_condition_keys)
                ]
                condition_key = training_key_info["condition_key"]
                condition_tag = training_key_info["condition_tag"]
            else:
                # RANDOM: Generate new random condition key (tests generalization)
                if num_conditions > 1:
                    additional_conditions = logistics_conditions.get_random_conditions(
                        n=num_conditions - 1
                    )
                    all_conditions = [error_code] + additional_conditions
                    all_conditions.sort()
                    condition_key = "+".join(all_conditions)
                    condition_str = " + ".join(all_conditions)
                    condition_tag = f" [Conditions: {condition_str}]"
                else:
                    condition_key = error_code
                    condition_tag = ""

            test_variations.append(
                self._build_scenario(
                    task=task + condition_tag,
                    expected=f"PRECEPT applies: {info['solution']} (1 step)",
                    black_swan_type="Logistics/Customs_Hold",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
                    phase="test",
                    tests_learning=condition_key,
                    condition_key=condition_key,
                    test_mode=test_mode,  # Tag for analysis
                )
            )

        return training + test_variations

    def generate_conflict_resolution_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 1,
    ) -> List[Dict]:
        """
        Generate scenarios specifically designed to test PRECEPT's conflict resolution.

        These scenarios are crafted to trigger conflicts between static knowledge
        (from logistics_kb.json) and dynamic knowledge (learned from experience).

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10). Multi-condition
                scenarios test conflict resolution in complex condition spaces.

        CONFLICT CATEGORIES TESTED:
        ═══════════════════════════════════════════════════════════════════════════
        1. DYNAMIC SHOULD OVERRIDE STATIC (Outdated static knowledge)
           - Hamburg labor "stable" (static says stable, but dynamic shows strikes)
           - Rotterdam customs "standard 3-day" (but experience shows delays)

        2. STATIC SHOULD WIN (Regulatory requirements)
           - Boston FDA pharmaceutical certificate (cannot be bypassed)
           - HS code verification (mandatory, no exceptions)
           - Temperature container pre-booking (48h requirement)

        3. DYNAMIC COMPLETES STATIC (Incomplete static knowledge)
           - Chicago hub procedures (static is vague, dynamic has specifics)
           - R-482 error fallback (static mentions procedure, dynamic has details)

        4. AGREEMENT (Static and Dynamic align - confidence boost)
           - Rotterdam as Hamburg fallback
           - Certificate of origin for customs
        ═══════════════════════════════════════════════════════════════════════════
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(2, self.num_samples // 4)

        training = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATE-BASED CONFLICT SCENARIO GENERATION
        # ═══════════════════════════════════════════════════════════════════════
        # Uses configuration from LogisticsScenarioConfig - NO HARDCODED VALUES

        # CATEGORY 1: DYNAMIC SHOULD OVERRIDE STATIC (Outdated static)
        # Templates from config
        dynamic_override_training_templates = (
            self.config.DYNAMIC_OVERRIDE_TRAINING_TEMPLATES
        )
        dynamic_override_test_templates = self.config.DYNAMIC_OVERRIDE_TEST_TEMPLATES

        # Use BLOCKED_PORTS from config - no hardcoded ports or error codes
        # Build dynamic lesson templates based on block_reason from config
        def build_lesson_template(block_reason: str) -> str:
            """Build lesson template based on configured block_reason."""
            # Use config for reason → template mapping
            for key, template in self.config.BLOCK_REASON_TEMPLATES.items():
                if key in block_reason.lower():
                    return template
            return "{port} port is currently unavailable. Use {alt} as alternative."

        # Get destinations and cargo types from config
        destinations = list(self.config.DESTINATIONS.keys())
        cargo_types = list(self.config.CARGO_TYPES.keys())

        # Build all dynamic override training combinations using CONFIG
        all_dynamic_override_training = []
        for port, port_config in self.config.BLOCKED_PORTS.items():
            error_code = port_config["error_code"]
            alternatives = port_config["working_alternatives"]
            block_reason = port_config.get("block_reason", "unavailable")
            lesson_template = build_lesson_template(block_reason)

            for dest in destinations:
                if port.lower() == dest.lower():
                    continue
                for cargo in cargo_types:
                    for template in dynamic_override_training_templates:
                        alt = random.choice(alternatives)
                        task = template.format(
                            port=port.title(), dest=dest.title(), cargo=cargo
                        )
                        all_dynamic_override_training.append(
                            {
                                "task": task,
                                "expected": f"{error_code} → use {alt}",
                                "black_swan_type": "Logistics/Conflict_DynamicOverride",
                                "precept_lesson": lesson_template.format(
                                    port=port.title(), alt=alt.title()
                                ),
                                "phase": "training",
                                "conflict_type": "dynamic_should_override",
                                "static_knowledge_tested": f"{port.title()} OPERATIONAL (outdated static)",
                            }
                        )

        # Build all dynamic override test combinations using CONFIG
        all_dynamic_override_test = []
        for port, port_config in self.config.BLOCKED_PORTS.items():
            for dest in destinations:
                if port.lower() == dest.lower():
                    continue
                for cargo in cargo_types:
                    for template in dynamic_override_test_templates:
                        task = template.format(
                            port=port.title(), dest=dest.title(), cargo=cargo
                        )
                        all_dynamic_override_test.append(
                            {
                                "task": task,
                                "expected": "PRECEPT detects conflict: dynamic vs static, uses dynamic",
                                "black_swan_type": "Logistics/Conflict_DynamicOverride",
                                "precept_lesson": f"{port.title()} is BLOCKED - dynamic overrides stale static",
                                "phase": "test",
                                "tests_learning": f"conflict_dynamic_override_{port}",
                                "conflict_type": "dynamic_should_override",
                            }
                        )

        # Sample from combinations
        dynamic_override_training = random.sample(
            all_dynamic_override_training,
            min(num_training // 4 + 1, len(all_dynamic_override_training)),
        )
        dynamic_override_test = random.sample(
            all_dynamic_override_test,
            min(num_test // 4 + 1, len(all_dynamic_override_test)),
        )

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 2: STATIC SHOULD WIN (Regulatory requirements)
        # ═══════════════════════════════════════════════════════════════════════
        # Use configuration - no hardcoded values
        regulatory_requirements = self.config.REGULATORY_REQUIREMENTS
        pharma_cargo = self.config.PHARMA_CARGO_TYPES

        all_static_wins_training = []
        all_static_wins_test = []

        for req_type, req_info in regulatory_requirements.items():
            for dest in req_info["destinations"]:
                # Use config flag to determine cargo type
                use_pharma = req_info.get("requires_pharma_cargo", False)
                for cargo in pharma_cargo if use_pharma else ["standard"]:
                    for template in req_info["training_templates"]:
                        task = template.format(dest=dest.title(), cargo=cargo)
                        all_static_wins_training.append(
                            {
                                "task": task,
                                "expected": f"FAILURE → {req_info['error']}, regulatory requirement",
                                "black_swan_type": "Logistics/Conflict_StaticWins",
                                "precept_lesson": req_info["lesson"].format(
                                    dest=dest.title(), cargo=cargo
                                ),
                                "phase": "training",
                                "conflict_type": "static_should_win",
                                "static_knowledge_tested": f"{req_type} requirement",
                            }
                        )
                    for template in req_info["test_templates"]:
                        task = template.format(dest=dest.title(), cargo=cargo)
                        all_static_wins_test.append(
                            {
                                "task": task,
                                "expected": f"PRECEPT applies static rule: {req_type} required",
                                "black_swan_type": "Logistics/Conflict_StaticWins",
                                "precept_lesson": f"Static regulatory knowledge applies - {req_type}",
                                "phase": "test",
                                "tests_learning": f"conflict_static_wins_{req_type}",
                                "conflict_type": "static_should_win",
                            }
                        )

        static_wins_training = random.sample(
            all_static_wins_training,
            min(num_training // 4 + 1, len(all_static_wins_training)),
        )
        static_wins_test = random.sample(
            all_static_wins_test, min(num_test // 4 + 1, len(all_static_wins_test))
        )

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 3: DYNAMIC COMPLETES STATIC (Incomplete static knowledge)
        # ═══════════════════════════════════════════════════════════════════════
        # Template-based generation for completion scenarios

        # Build error_port_map from CONFIG - no hardcoded error codes
        error_port_map = {}
        for port, port_config in self.config.BLOCKED_PORTS.items():
            error_code = port_config["error_code"]
            alternatives = port_config["working_alternatives"]
            alt1 = alternatives[0] if alternatives else "alternative"
            alt2 = alternatives[1] if len(alternatives) > 1 else alt1
            error_port_map[error_code] = (port, alt1, alt2)

        # Use configuration - no hardcoded values
        incomplete_knowledge = {
            "error_fallback": {
                "training_templates": self.config.INCOMPLETE_ERROR_FALLBACK_TEMPLATES[
                    "training"
                ],
                "test_templates": self.config.INCOMPLETE_ERROR_FALLBACK_TEMPLATES[
                    "test"
                ],
                "error_port_map": error_port_map,  # Built from BLOCKED_PORTS config
                "lesson": self.config.INCOMPLETE_ERROR_LESSON,
            },
            "hub_procedures": {
                "training_templates": self.config.INCOMPLETE_HUB_TEMPLATES["training"],
                "test_templates": self.config.INCOMPLETE_HUB_TEMPLATES["test"],
                "hubs": self.config.INLAND_HUBS,
                "lesson": self.config.INCOMPLETE_HUB_LESSON,
            },
            "express_clearance": {
                "training_templates": self.config.INCOMPLETE_EXPRESS_TEMPLATES[
                    "training"
                ],
                "test_templates": self.config.INCOMPLETE_EXPRESS_TEMPLATES["test"],
                "ports": self.config.EXPRESS_CLEARANCE_PORTS,
                "lesson": self.config.INCOMPLETE_EXPRESS_LESSON,
            },
        }

        all_dynamic_completes_training = []
        all_dynamic_completes_test = []

        # Error fallback scenarios
        for error, (port, alt1, alt2) in incomplete_knowledge["error_fallback"][
            "error_port_map"
        ].items():
            for template in incomplete_knowledge["error_fallback"][
                "training_templates"
            ]:
                task = template.format(error=error, port=port.title())
                all_dynamic_completes_training.append(
                    {
                        "task": task,
                        "expected": f"{error} → use {alt1} (learned specific fallback)",
                        "black_swan_type": "Logistics/Conflict_DynamicCompletes",
                        "precept_lesson": incomplete_knowledge["error_fallback"][
                            "lesson"
                        ].format(
                            port=port.title(),
                            error=error,
                            alt1=alt1.title(),
                            alt2=alt2.title(),
                        ),
                        "phase": "training",
                        "conflict_type": "dynamic_completes",
                        "static_knowledge_tested": f"{error} fallback procedure (incomplete)",
                    }
                )
            for template in incomplete_knowledge["error_fallback"]["test_templates"]:
                task = template.format(error=error, port=port.title())
                all_dynamic_completes_test.append(
                    {
                        "task": task,
                        "expected": f"PRECEPT combines static + dynamic for {error}",
                        "black_swan_type": "Logistics/Conflict_DynamicCompletes",
                        "precept_lesson": f"Dynamic completes incomplete static for {error}",
                        "phase": "test",
                        "tests_learning": f"conflict_dynamic_completes_{error.lower().replace('-', '_')}",
                        "conflict_type": "dynamic_completes",
                    }
                )

        # Hub procedure scenarios
        for hub in incomplete_knowledge["hub_procedures"]["hubs"]:
            for cargo in cargo_types:
                for template in incomplete_knowledge["hub_procedures"][
                    "training_templates"
                ]:
                    task = template.format(hub=hub.title(), cargo=cargo)
                    all_dynamic_completes_training.append(
                        {
                            "task": task,
                            "expected": f"Learn specific {hub} procedures",
                            "black_swan_type": "Logistics/Conflict_DynamicCompletes",
                            "precept_lesson": incomplete_knowledge["hub_procedures"][
                                "lesson"
                            ].format(hub=hub.title(), cargo=cargo),
                            "phase": "training",
                            "conflict_type": "dynamic_completes",
                            "static_knowledge_tested": f"{hub} procedures (incomplete)",
                        }
                    )

        # Express clearance scenarios
        for port in incomplete_knowledge["express_clearance"]["ports"]:
            for cargo in cargo_types[:3]:  # Limit combinations
                for template in incomplete_knowledge["express_clearance"][
                    "training_templates"
                ]:
                    task = template.format(port=port.title(), cargo=cargo)
                    all_dynamic_completes_training.append(
                        {
                            "task": task,
                            "expected": "Learn express clearance specifics",
                            "black_swan_type": "Logistics/Conflict_DynamicCompletes",
                            "precept_lesson": incomplete_knowledge["express_clearance"][
                                "lesson"
                            ].format(port=port.title()),
                            "phase": "training",
                            "conflict_type": "dynamic_completes",
                            "static_knowledge_tested": "Express clearance timing (incomplete)",
                        }
                    )

        dynamic_completes_training = random.sample(
            all_dynamic_completes_training,
            min(num_training // 4 + 1, len(all_dynamic_completes_training)),
        )
        dynamic_completes_test = random.sample(
            all_dynamic_completes_test,
            min(num_test // 4 + 1, len(all_dynamic_completes_test)),
        )

        # ═══════════════════════════════════════════════════════════════════════
        # CATEGORY 4: AGREEMENT (Static and Dynamic align - boost confidence)
        # ═══════════════════════════════════════════════════════════════════════
        # Template-based generation for agreement scenarios

        # Use configuration - no hardcoded values
        agreement_scenarios = {
            "fallback_confirmation": {
                "training_templates": self.config.AGREEMENT_FALLBACK_TEMPLATES[
                    "training"
                ],
                "test_templates": self.config.AGREEMENT_FALLBACK_TEMPLATES["test"],
                "pairs": self.config.FALLBACK_PAIRS,
                "lesson": self.config.AGREEMENT_FALLBACK_LESSON,
            },
            "documentation_success": {
                "training_templates": self.config.AGREEMENT_DOCUMENTATION_TEMPLATES[
                    "training"
                ],
                "test_templates": self.config.AGREEMENT_DOCUMENTATION_TEMPLATES["test"],
                "destinations": self.config.DOCUMENTATION_SUCCESS_DESTINATIONS,
                "lesson": self.config.AGREEMENT_DOCUMENTATION_LESSON,
            },
            "quality_confirmation": {
                "training_templates": self.config.AGREEMENT_QUALITY_TEMPLATES[
                    "training"
                ],
                "test_templates": self.config.AGREEMENT_QUALITY_TEMPLATES["test"],
                "port_cargo": self.config.QUALITY_PORT_CARGO_PAIRS,
                "lesson": self.config.AGREEMENT_QUALITY_LESSON,
            },
        }

        all_agreement_training = []
        all_agreement_test = []

        # Fallback confirmation scenarios
        for primary, alt in agreement_scenarios["fallback_confirmation"]["pairs"]:
            for template in agreement_scenarios["fallback_confirmation"][
                "training_templates"
            ]:
                task = template.format(primary=primary.title(), alt=alt.title())
                all_agreement_training.append(
                    {
                        "task": task,
                        "expected": f"SUCCESS → {alt.title()} works as {primary.title()} fallback",
                        "black_swan_type": "Logistics/Conflict_Agreement",
                        "precept_lesson": agreement_scenarios["fallback_confirmation"][
                            "lesson"
                        ].format(primary=primary.title(), alt=alt.title()),
                        "phase": "training",
                        "conflict_type": "agreement",
                        "static_knowledge_tested": f"{primary.title()} ↔ {alt.title()} fallback",
                    }
                )
            for template in agreement_scenarios["fallback_confirmation"][
                "test_templates"
            ]:
                task = template.format(primary=primary.title(), alt=alt.title())
                all_agreement_test.append(
                    {
                        "task": task,
                        "expected": f"HIGH CONFIDENCE: {alt.title()} (static and dynamic agree)",
                        "black_swan_type": "Logistics/Conflict_Agreement",
                        "precept_lesson": "Confidence boosted when static and dynamic align",
                        "phase": "test",
                        "tests_learning": f"conflict_agreement_{primary}_{alt}",
                        "conflict_type": "agreement",
                    }
                )

        # Documentation success scenarios
        for dest in agreement_scenarios["documentation_success"]["destinations"]:
            for template in agreement_scenarios["documentation_success"][
                "training_templates"
            ]:
                task = template.format(dest=dest.title())
                all_agreement_training.append(
                    {
                        "task": task,
                        "expected": "SUCCESS → Expedited processing with documentation",
                        "black_swan_type": "Logistics/Conflict_Agreement",
                        "precept_lesson": agreement_scenarios["documentation_success"][
                            "lesson"
                        ].format(dest=dest.title()),
                        "phase": "training",
                        "conflict_type": "agreement",
                        "static_knowledge_tested": "Documentation expedites customs",
                    }
                )

        # Quality confirmation scenarios
        for port, cargo in agreement_scenarios["quality_confirmation"]["port_cargo"]:
            for template in agreement_scenarios["quality_confirmation"][
                "training_templates"
            ]:
                task = template.format(port=port.title(), cargo=cargo)
                all_agreement_training.append(
                    {
                        "task": task,
                        "expected": f"SUCCESS → {port.title()} maintains quality handling",
                        "black_swan_type": "Logistics/Conflict_Agreement",
                        "precept_lesson": agreement_scenarios["quality_confirmation"][
                            "lesson"
                        ].format(port=port.title(), cargo=cargo),
                        "phase": "training",
                        "conflict_type": "agreement",
                        "static_knowledge_tested": f"{port.title()} {cargo} handling",
                    }
                )

        agreement_training = random.sample(
            all_agreement_training,
            min(num_training // 4 + 1, len(all_agreement_training)),
        )
        agreement_test = random.sample(
            all_agreement_test, min(num_test // 4 + 1, len(all_agreement_test))
        )

        # Combine all scenarios
        all_training = (
            dynamic_override_training
            + static_wins_training
            + dynamic_completes_training
            + agreement_training
        )
        all_test = (
            dynamic_override_test
            + static_wins_test
            + dynamic_completes_test
            + agreement_test
        )

        # Sample if needed
        training = random.sample(all_training, min(num_training, len(all_training)))
        test_scenarios = random.sample(all_test, min(num_test, len(all_test)))

        # ═══════════════════════════════════════════════════════════════════════
        # APPLY MULTI-CONDITION TRANSFORMATION
        # ═══════════════════════════════════════════════════════════════════════
        # If num_conditions > 1, transform all scenarios to use multi-condition keys
        if num_conditions > 1:
            logistics_conditions = LogisticsConditions()

            def add_multi_conditions(scenario: Dict) -> Dict:
                """Add multi-condition key and tag to a scenario."""
                # Get the base error code from expected field or use a default
                expected = scenario.get("expected", "")
                base_error = None

                # ═══════════════════════════════════════════════════════════════════
                # FIX: Validate that base_error is actually a valid condition code
                # Some scenarios have expected values like "SUCCESS → ..." which
                # would incorrectly extract "SUCCESS" as an error code.
                # Valid condition codes match patterns like: R-482, H-903, C-COLD, etc.
                # ═══════════════════════════════════════════════════════════════════
                valid_condition_codes = set(
                    logistics_conditions.get_all_conditions().keys()
                )

                # Extract error code from expected (format: "ERROR_CODE → action")
                if "→" in expected:
                    candidate = expected.split("→")[0].strip()
                    # Only use if it's a valid condition code
                    if candidate in valid_condition_codes:
                        base_error = candidate
                elif "R-482" in expected or "H-903" in expected:
                    # Fallback: look for known codes
                    for code in ["R-482", "H-903", "SH-701", "LA-550"]:
                        if code in expected:
                            base_error = code
                            break

                # Generate additional conditions
                additional_conditions = logistics_conditions.get_random_conditions(
                    n=num_conditions - 1
                )

                if base_error and base_error not in additional_conditions:
                    all_conditions = [base_error] + additional_conditions
                else:
                    all_conditions = logistics_conditions.get_random_conditions(
                        n=num_conditions
                    )

                all_conditions.sort()  # Deterministic ordering
                condition_key = "+".join(all_conditions)
                condition_str = " + ".join(all_conditions)
                condition_tag = f" [Conditions: {condition_str}]"

                # Update scenario
                scenario["task"] = scenario["task"] + condition_tag
                scenario["condition_key"] = condition_key
                if "expected" in scenario and "→" in scenario["expected"]:
                    parts = scenario["expected"].split("→")
                    scenario["expected"] = f"{condition_key} →" + "→".join(parts[1:])
                if "tests_learning" in scenario:
                    scenario["tests_learning"] = (
                        f"{scenario['tests_learning']}_{condition_key}"
                    )

                return scenario

            training = [add_multi_conditions(s.copy()) for s in training]
            test_scenarios = [add_multi_conditions(s.copy()) for s in test_scenarios]

        print(
            f"  🔀 Conflict Resolution: {len(training)} train + {len(test_scenarios)} test scenarios"
        )
        print(
            "     Testing: dynamic_override, static_wins, dynamic_completes, agreement"
        )
        if num_conditions > 1:
            print(f"     Multi-condition: {num_conditions} conditions per scenario")

        return training + test_scenarios

    def generate_fleet_learning_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
        num_conditions: int = 1,
    ) -> List[Dict]:
        """
        Generate scenarios that explicitly test CROSS-ENTITY TRANSFER.

        This is the "Fleet Learning" pattern (like Tesla's fleet):
        - Training: Entity X encounters Condition Y → learns Rule Z
        - Testing: Entity K (DIFFERENT!) encounters SAME Condition Y → applies Rule Z

        GUARANTEED PATTERN (single-condition):
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - Rotterdam → Boston (R-482) → learns "R-482 → antwerp"

        Testing (DIFFERENT entities, SAME conditions):
          - Rotterdam → London (R-482) → applies "R-482 → antwerp" ✓
        ═══════════════════════════════════════════════════════════════════════════

        MULTI-CONDITION PATTERN (num_conditions > 1):
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - Rotterdam → Boston [Conditions: R-482 + C-HIGH] → learns "C-HIGH+R-482 → antwerp"

        Testing (DIFFERENT entities, SAME conditions):
          - Rotterdam → London [Conditions: R-482 + C-HIGH] → applies rule ✓
        ═══════════════════════════════════════════════════════════════════════════

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Initialize condition generator for multi-condition scenarios
        logistics_conditions = LogisticsConditions()

        # Define blocked ports with their error codes
        blocked_ports = self.config.BLOCKED_PORTS
        destinations = list(self.config.DESTINATIONS.keys())
        cargo_types = list(self.config.CARGO_TYPES.keys())

        # For each blocked port, create:
        # - Training: One route (Port → Dest1) that learns the rule
        # - Testing: Different routes (Port → Dest2, Dest3) that apply the rule

        for port, port_info in blocked_ports.items():
            error_code = port_info["error_code"]
            alternatives = port_info["working_alternatives"]
            alt = alternatives[0] if alternatives else "alternative"

            # Get valid destinations (not same as origin)
            valid_dests = [d for d in destinations if d.lower() != port.lower()]
            if len(valid_dests) < 2:
                continue

            # Shuffle to randomize which dest is training vs testing
            random.shuffle(valid_dests)

            # Generate conditions for this scenario
            if num_conditions > 1:
                # Get additional conditions (error_code is always included)
                all_conditions = logistics_conditions.get_random_conditions(
                    n=num_conditions, must_include=error_code
                )
                condition_key = MultiConditionConfig.generate_condition_key(
                    all_conditions
                )
                condition_str = " + ".join(all_conditions)
                condition_tag = f" [Conditions: {condition_str}]"
            else:
                all_conditions = [error_code]
                condition_key = error_code
                condition_tag = ""

            # TRAINING: First destination learns the rule
            training_dest = valid_dests[0]
            cargo = random.choice(cargo_types)

            training.append(
                {
                    "task": f"Ship {cargo} cargo from {port.title()} port to {training_dest.title()}{condition_tag}",
                    "expected": f"{condition_key} → use {alt}",
                    "black_swan_type": "Logistics/FleetLearning_Train",
                    "precept_lesson": f"When {port.title()} is blocked ({condition_key}), use {alt.title()} for ANY destination",
                    "phase": "training",
                    "fleet_learning": {
                        "blocked_port": port,
                        "error_code": error_code,
                        "condition_key": condition_key,
                        "all_conditions": all_conditions,
                        "learned_alternative": alt,
                        "training_destination": training_dest,
                    },
                }
            )

            # TESTING: Different destinations apply the SAME rule (with SAME conditions)
            for test_dest in valid_dests[1:3]:  # Use 2 different test destinations
                test_cargo = random.choice(cargo_types)
                testing.append(
                    {
                        "task": f"Book {test_cargo} shipment from {port.title()} to {test_dest.title()}{condition_tag}",
                        "expected": f"Apply learned rule: {condition_key} → {alt}",
                        "black_swan_type": "Logistics/FleetLearning_Test",
                        "precept_lesson": f"Cross-entity transfer: Rule for {condition_key} applies to {test_dest.title()} route",
                        "phase": "test",
                        "tests_learning": f"fleet_learning_{condition_key}",
                        "fleet_learning": {
                            "blocked_port": port,
                            "error_code": error_code,
                            "condition_key": condition_key,
                            "all_conditions": all_conditions,
                            "expected_alternative": alt,
                            "different_destination": test_dest,
                            "training_destination": training_dest,
                        },
                    }
                )

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(
            f"  🚀 Fleet Learning: {len(training)} train + {len(testing)} test scenarios"
        )
        print(
            "     Pattern: Entity X + Condition Y → Rule Z | Entity K + Condition Y → Apply Rule Z"
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

        This is where PRECEPT truly shines over baselines:
        - With N conditions, baselines face 2^N possible states (combinatorial explosion)
        - Baselines may apply rules when only SOME conditions match (partial match error)
        - PRECEPT uses deterministic CSP matching - rules apply ONLY when ALL conditions match

        Pattern:
            Entity X + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → Rule Z

        Example (3 conditions):
            Rotterdam + R-482 + C-HIGH + T-PEAK → use hamburg_expedited

        IMPORTANT: This method builds ALL possible combinations first, then samples N.
        This ensures statistical reliability - requesting 16 training scenarios will
        generate 16 diverse scenarios (same error type with different variations),
        not just 4 scenarios (one per blocked port).

        Args:
            num_training: Number of training scenarios
            num_test: Number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           Higher = harder for baselines (2^N states)
            test_mode: Testing mode - controls how test condition keys are generated:
                - "matched": Test scenarios REUSE condition keys from training (O(1) lookup test)
                - "random": Test scenarios generate NEW random condition keys (generalization test)

        Returns:
            List of multi-condition scenarios
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        # Clamp conditions to valid range (1-10)
        num_conditions = max(1, min(10, num_conditions))

        # Initialize conditions provider
        conditions_provider = LogisticsConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Get ports and alternatives from config
        blocked_ports = self.config.BLOCKED_PORTS
        destinations = list(self.config.DESTINATIONS.keys())
        cargo_types = list(self.config.CARGO_TYPES.keys())

        # Templates for variation (similar to single-condition approach)
        training_templates = [
            "Ship {cargo} from {port} to {dest}",
            "Book {cargo} cargo from {port} port to {dest}",
            "Transport {cargo} shipment from {port} to {dest}",
            "Arrange {cargo} delivery from {port} to {dest}",
        ]
        test_templates = [
            "Book {cargo} shipment from {port} to {dest}",
            "Send {cargo} from {port} port to {dest}",
            "Deliver {cargo} cargo from {port} to {dest}",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        # Instead of random sampling from C(20,10) = 184,756 combinations,
        # we pre-generate K unique composite keys and reuse them with β coverage.
        #
        # Formula: β = num_training / K
        # Example: K=7 ports, train=21 → β=3 (each key seen 3 times)
        #
        # This mirrors single-condition behavior where:
        # - 7 error types with train=21 gives β=3
        # ═══════════════════════════════════════════════════════════════════════

        # Get valid solutions for multi-condition scenarios (always valid ports)
        valid_solutions = getattr(
            self.config, "MULTI_CONDITION_VALID_SOLUTIONS", ["antwerp", "hamburg"]
        )

        # K = number of unique rule types = number of blocked ports
        K = len(blocked_ports)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per blocked port
        # Each key includes the port's error code + (num_conditions-1) random other conditions
        fixed_key_pool = {}
        for port, port_info in blocked_ports.items():
            error_code = port_info["error_code"]
            # Sample (num_conditions - 1) OTHER conditions deterministically per port
            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            # Get valid alternative for this port
            alternatives = port_info["working_alternatives"]
            valid_alts = [a for a in alternatives if a in valid_solutions]
            alt = valid_alts[0] if valid_alts else valid_solutions[0]

            fixed_key_pool[port] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": alt,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        all_training_combos = []
        all_test_combos = []

        for port, port_info in blocked_ports.items():
            error_code = port_info["error_code"]
            # ═══════════════════════════════════════════════════════════════════
            # USE FIXED KEY from pool (not random sampling!)
            # This ensures β-coverage: each key is seen β times during training
            # ═══════════════════════════════════════════════════════════════════
            key_info = fixed_key_pool[port]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]
            alt = key_info["solution"]

            # Get valid destinations (not same as origin)
            # ═══════════════════════════════════════════════════════════════════
            # FIX: Filter destinations by solution compatibility!
            # - Hamburg → US is BLOCKED (H-903)
            # - Antwerp → Asia is BLOCKED (A-701)
            # We must generate scenarios where the expected solution WORKS.
            # ═══════════════════════════════════════════════════════════════════
            us_destinations = {
                "boston",
                "new_york",
                "chicago",
                "miami",
                "seattle",
                "los_angeles",
            }
            # Only include destinations from DESTINATIONS config
            asia_destinations = {"shanghai", "singapore"}
            europe_destinations = {
                "london",
                "hamburg",
            }

            if alt.lower() == "hamburg":
                # Hamburg → US is blocked, use non-US destinations
                blocked_regions = us_destinations
            elif alt.lower() == "antwerp":
                # Antwerp → Asia is blocked, use non-Asia destinations
                blocked_regions = asia_destinations
            else:
                # Unknown solution - don't block any regions
                blocked_regions = set()

            valid_dests = [
                d
                for d in destinations
                if d.lower() != port.lower() and d.lower() not in blocked_regions
            ]
            if not valid_dests:
                # Fallback to European destinations if everything is blocked
                valid_dests = [d for d in europe_destinations if d in destinations]
                if not valid_dests:
                    continue

            for dest in valid_dests:
                for cargo in cargo_types:
                    for template in training_templates:
                        # ═══════════════════════════════════════════════════════
                        # FIXED KEY: Reuse the same composite key for this port
                        # Only destination, cargo, and template vary
                        # ═══════════════════════════════════════════════════════
                        all_training_combos.append(
                            {
                                "port": port,
                                "port_info": port_info,
                                "dest": dest,
                                "cargo": cargo,
                                "template": template,
                                "error_code": error_code,
                                "alt": alt,
                                "all_conds": all_conds,
                                "condition_key": condition_key,
                            }
                        )

                    for template in test_templates:
                        # Test scenarios will get condition info from sampled training
                        all_test_combos.append(
                            {
                                "port": port,
                                "port_info": port_info,
                                "dest": dest,
                                "cargo": cargo,
                                "template": template,
                                "error_code": error_code,
                                "alt": alt,
                            }
                        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: SAMPLE N TRAINING SCENARIOS WITH GUARANTEED β-COVERAGE
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
        # Track condition keys used in training for test scenario matching
        training_condition_keys = {}

        for combo in sampled_training:
            port = combo["port"]
            dest = combo["dest"]
            cargo = combo["cargo"]
            template = combo["template"]
            alt = combo["alt"]
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            cond_descriptions = [all_conditions.get(c, c) for c in all_conds]

            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(
                port=port.title(),
                dest=dest.title(),
                cargo=cargo,
            )

            training.append(
                {
                    "task": task,
                    "expected": f"{condition_key} → {alt}",
                    "black_swan_type": f"Logistics/MultiCondition_{num_conditions}C_Train",
                    "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {alt}",
                    "phase": "training",
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "condition_descriptions": cond_descriptions,
                        "port": port,
                        "solution": alt,
                    },
                }
            )

            # Track for test matching
            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "port": port,
                "solution": alt,
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS
        # Mode "matched": Use SAME condition keys as training (O(1) lookup test)
        # Mode "random": Generate NEW random condition keys (generalization test)
        # ═══════════════════════════════════════════════════════════════════════
        testing = []

        # Sample test combinations
        sampled_test = random.sample(
            all_test_combos,
            min(num_test * 2, len(all_test_combos)),  # Sample extra to filter
        )

        for combo in sampled_test:
            if len(testing) >= num_test:
                break

            port = combo["port"]
            dest = combo["dest"]
            cargo = combo["cargo"]
            template = combo["template"]
            error_code = combo["error_code"]
            alt = combo["alt"]

            if test_mode == "matched":
                # ═══════════════════════════════════════════════════════════════
                # MATCHED MODE: Reuse condition keys from training
                # Tests O(1) exact lookup capability
                # ═══════════════════════════════════════════════════════════════
                matching_keys = [
                    (k, v)
                    for k, v in training_condition_keys.items()
                    if v["port"] == port
                ]

                if not matching_keys:
                    continue

                # Pick a random matching condition key
                condition_key, key_info = random.choice(matching_keys)
                all_conds = key_info["conditions"]
                solution = key_info["solution"]
            else:
                # ═══════════════════════════════════════════════════════════════
                # RANDOM MODE: Generate NEW random condition keys
                # Tests generalization and partial matching capability
                # ═══════════════════════════════════════════════════════════════
                other_conditions = random.sample(
                    [c for c in condition_codes if c != error_code],
                    min(num_conditions - 1, len(condition_codes) - 1),
                )
                all_conds = [error_code] + other_conditions
                condition_key = MultiConditionConfig.generate_condition_key(all_conds)
                solution = alt

            cond_str = " + ".join(all_conds)

            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(
                port=port.title(),
                dest=dest.title(),
                cargo=cargo,
            )

            testing.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Logistics/MultiCondition_{num_conditions}C_Test_{test_mode.capitalize()}",
                    "precept_lesson": f"Cross-entity transfer: ALL {num_conditions} conditions match → apply rule",
                    "phase": "test",
                    "tests_learning": f"multi_condition_{condition_key}",
                    "condition_key": condition_key,  # Add for analysis
                    "test_mode": test_mode,  # Tag for analysis
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "expected_solution": solution,
                        "different_destination": dest,
                        "test_mode": test_mode,
                    },
                }
            )

        mode_desc = (
            "MATCHED (O(1) lookup)"
            if test_mode == "matched"
            else "RANDOM (generalization)"
        )
        print(
            f"  🔀 Multi-Condition ({num_conditions}C, {mode_desc}): {len(training)} train + {len(testing)} test"
        )
        print(
            f"     Pool: {len(all_training_combos)} training combos, {len(all_test_combos)} test combos"
        )
        print(
            f"     Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states!"
        )
        print("     PRECEPT Advantage: Deterministic exact-match only")

        return training + testing

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
        - RANDOM mode: Use PARTIAL condition keys with ~60% overlap from ALL training keys
                       (includes learned rules, first-try successes, and exhausted retries)

        IMPORTANT FOR MATCHED MODE:
        - learned_rule_keys should be PRE-FILTERED to only include keys from the
          current training session's fixed key pool.
        - If all_training_keys is provided, we validate that learned keys are in it.
        - This prevents stale rules from previous runs from contaminating tests.

        Args:
            learned_rule_keys: Dict mapping condition_key -> rule_text (e.g., "key → solution")
                              For MATCHED mode: should be filtered to current training pool
            num_test: Number of test scenarios to generate
            mode: "matched" for exact key reuse, "random" for partial overlap
            seed: Random seed for reproducibility
            all_training_keys: ALL condition keys encountered during training (not just learned)
                              For MATCHED mode: used to validate learned keys
                              For RANDOM mode: provides broader coverage including:
                              - First-try successes (no rule learned)
                              - Exhausted retries (failed completely)

        Returns:
            List of test scenario dictionaries
        """
        if seed is not None:
            random.seed(seed)

        # For MATCHED mode: must have learned rules for O(1) lookup test
        # For RANDOM mode: can use all training keys (broader coverage)
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

        # Initialize conditions provider for generating random conditions
        conditions_provider = LogisticsConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Get config values
        blocked_ports = self.config.BLOCKED_PORTS

        # ═══════════════════════════════════════════════════════════════════════
        # BUILD REVERSE MAPPING: error_code → port_name
        # This allows us to find which blocked port corresponds to a condition_key
        # ═══════════════════════════════════════════════════════════════════════
        error_code_to_port = {}
        for port_name, port_info in blocked_ports.items():
            error_code = port_info.get("error_code", "")
            if error_code:
                error_code_to_port[error_code] = port_name

        # Templates for test scenarios (variation from training templates)
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Ship cargo from {src} to {dst}",
            "Book shipment from {src} to {dst}",
            "Transport goods from {src} to {dst}",
            "Arrange delivery from {src} to {dst}",
            "Route freight from {src} to {dst}",
            "Send shipment from {src} to {dst}",
            "Handle cargo from {src} to {dst}",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # KEY SELECTION STRATEGY
        # MATCHED mode: Use only learned rule keys (for O(1) exact lookup test)
        # RANDOM mode: Use ALL training keys (learned + first-try + exhausted)
        #              This tests generalization on the full training distribution
        # ═══════════════════════════════════════════════════════════════════════
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
                print(
                    f"  📋 MATCHED mode: Using ALL {len(base_keys_list)} training keys"
                )
                print(
                    f"     ✓ {learned_count} keys with learned rules (TIER 1 hit expected)"
                )
                print(
                    f"     ○ {unlearned_count} keys without rules (cross-episode learning test)"
                )
            else:
                base_keys_list = learned_keys_list
                print(
                    f"  📋 MATCHED mode: Using {len(base_keys_list)} learned rule keys (fallback)"
                )
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
            # Cycle through learned keys (guaranteed to be in learned_rule_keys)
            base_key = base_keys_list[i % len(base_keys_list)]
            base_conditions = base_key.split("+")

            # ═══════════════════════════════════════════════════════════════════
            # Get solution - may not exist in learned_rule_keys for unlearned keys!
            # For unlearned keys, compute expected solution from hash-based config.
            # ═══════════════════════════════════════════════════════════════════
            from ..config import LogisticsConfig

            if base_key in learned_rule_keys:
                rule_text = learned_rule_keys[base_key]
                # ═══════════════════════════════════════════════════════════════
                # FIX: Extract solution robustly from various rule formats
                # Rule formats:
                #   - "key → solution" → extract "solution"
                #   - "key: context → LLM→origin→dest" → extract "origin"
                # 
                # Strategy: Split by " → " (with spaces) FIRST to separate key
                # from solution, then parse the solution part for exploration paths
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
                solution = LogisticsConfig.get_valid_solution_for_conditions(base_key)
                is_learned = False

            # Track test type for analysis
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
                    # Second 50%: Generate novel key with 60% overlap (tests Tier 2/3)
                    keep_count = max(4, int(len(base_conditions) * 0.6))
                    replace_count = len(base_conditions) - keep_count

                    kept_conditions = random.sample(
                        base_conditions, min(keep_count, len(base_conditions))
                    )
                    available_new = [
                        c for c in condition_codes if c not in kept_conditions
                    ]
                    new_conditions = random.sample(
                        available_new, min(replace_count, len(available_new))
                    )

                    all_conds = sorted(kept_conditions + new_conditions)
                    condition_key = "+".join(all_conds)
                    test_type = "novel"
            else:
                # Fallback for any other mode
                condition_key = base_key
                all_conds = base_conditions
                test_type = "exact_match"

            # Build task with varied template
            template = test_templates[i % len(test_templates)]

            # ═══════════════════════════════════════════════════════════════════════
            # FIX: Use the blocked port as origin (source) in test scenarios
            # ═══════════════════════════════════════════════════════════════════════
            # The condition_key contains error codes like "R-482" (Rotterdam), "H-903" (Hamburg)
            # For MATCHED mode, the test origin should be the SAME blocked port used in training
            # This ensures the learned rule (use alternative origin) is applicable
            # ═══════════════════════════════════════════════════════════════════════
            src = None
            for cond in all_conds:
                if cond in error_code_to_port:
                    src = error_code_to_port[cond]
                    break

            # Fallback: if no blocked port found in condition_key, use a random one
            if not src:
                src = list(blocked_ports.keys())[i % len(blocked_ports)]

            # ═══════════════════════════════════════════════════════════════════════
            # FIX: Select destination compatible with the learned solution
            # ═══════════════════════════════════════════════════════════════════════
            # Hamburg → US is BLOCKED (H-903), Antwerp → Asia is BLOCKED (A-701)
            # So we must select destinations that work with the expected solution!
            # ═══════════════════════════════════════════════════════════════
            # BUGFIX: Only use destinations from LogisticsConfig.DESTINATION_PORTS!
            # The old hardcoded lists included paris, amsterdam, lisbon,
            # manchester, tokyo, hong_kong which are NOT in DESTINATION_PORTS.
            # When these are used in test tasks, parse_task cannot find them,
            # setting parsed_task.target=None. This causes Pydantic validation
            # errors in execute_logistics_multi_condition.
            # ═══════════════════════════════════════════════════════════════
            us_destinations = [
                "boston",
                "new_york",
                "chicago",
                "miami",
                "seattle",
                "los_angeles",
            ]
            asia_destinations = [
                "shanghai",
                "singapore",
            ]
            # European destinations - only include those in DESTINATION_PORTS!
            europe_destinations = [
                "london",
            ]

            if solution.lower() == "hamburg":
                # Hamburg → US is blocked, use non-US destinations
                valid_dests = europe_destinations + asia_destinations
            elif solution.lower() == "antwerp":
                # Antwerp → Asia is blocked, use non-Asia destinations
                valid_dests = europe_destinations + us_destinations
            else:
                # Unknown solution or not hamburg/antwerp - use Europe (safe)
                valid_dests = europe_destinations

            dst = valid_dests[i % len(valid_dests)]

            # Ensure src != dst
            if src.lower() == dst.lower():
                dst = valid_dests[(i + 1) % len(valid_dests)]

            # BLACK SWAN CSP: NO conditions in task - only src/dst context
            task = template.format(
                src=src.title(),
                dst=dst.title(),
            )

            num_conditions = len(all_conds)
            scenarios.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Logistics/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
                    "precept_lesson": f"Transfer learning: apply rule for {condition_key}",
                    "phase": "test",
                    "tests_learning": f"multi_condition_{condition_key}",
                    "condition_key": condition_key,
                    "test_mode": mode,
                    "test_type": test_type,  # "exact_match" (Tier 1) or "novel" (Tier 2/3)
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "expected_solution": solution,
                        "base_key": base_key,  # Original key for analysis
                        "test_mode": mode,
                        "test_type": test_type,  # For analysis by retrieval tier
                    },
                }
            )

        # Count by test type for logging
        exact_count = sum(1 for s in scenarios if s.get("test_type") == "exact_match")
        novel_count = sum(1 for s in scenarios if s.get("test_type") == "novel")

        if mode == "matched":
            mode_desc = "MATCHED (exact keys → Tier 1 O(1) lookup)"
        else:
            mode_desc = f"MIXED ({exact_count} exact-match + {novel_count} novel)"
        print(
            f"  📋 Generated {len(scenarios)} test scenarios from learned keys ({mode_desc})"
        )

        return scenarios

    def generate_compositional_test(
        self,
        learned_atoms: List[str],
        num_test: int,
        num_conditions: int = 2,
        include_conflicts: bool = True,
        seed: int = None,
    ) -> List[Dict]:
        """
        Generate test scenarios using ONLY learned atomic conditions.

        This method creates TRUE compositional generalization tests:
        - Test composites are formed by combining ONLY atoms learned during training
        - If training learned A, B, C, D then test A+B, B+C, A+C+D (never seen before!)
        - P₁ > 0% is possible because stacked precepts guide the LLM

        Args:
            learned_atoms: List of atomic condition codes learned during training
                          (e.g., ["LA-550", "R-482", "H-903", "SH-701"])
            num_test: Number of test scenarios to generate
            num_conditions: Number of conditions per composite (2, 3, 4...)
            include_conflicts: If True, include scenarios with conflicting precepts
            seed: Random seed for reproducibility

        Returns:
            List of test scenario dictionaries with novel composites
        """
        if seed is not None:
            random.seed(seed)

        if len(learned_atoms) < num_conditions:
            print(
                f"  ⚠️ Not enough learned atoms ({len(learned_atoms)}) for {num_conditions}-way composites"
            )
            return []

        # Remove duplicates and sort
        learned_atoms = sorted(set(learned_atoms))

        print(f"  🧬 COMPOSITIONAL TEST: Generating {num_test} test scenarios")
        print(f"     Learned atoms: {learned_atoms}")
        print(f"     Composite size: {num_conditions}-way combinations")

        # Generate all possible N-way combinations from learned atoms
        from itertools import combinations

        all_combos = list(combinations(learned_atoms, num_conditions))
        random.shuffle(all_combos)

        print(f"     Possible {num_conditions}-way combos: {len(all_combos)}")

        # Import config for solution computation
        from ..config import LogisticsConfig

        # Templates for test scenarios
        test_templates = [
            "Ship cargo from {src} to {dst}",
            "Book shipment from {src} to {dst}",
            "Transport goods from {src} to {dst}",
            "Arrange delivery from {src} to {dst}",
        ]

        # Source/destination pairs
        # CRITICAL: Destinations MUST match DESTINATION_PORTS in config/logistics.py
        # Valid destinations: boston, new_york, shanghai, singapore, london,
        #                     hamburg, los_angeles, chicago, miami, seattle
        src_dst_pairs = [
            ("Rotterdam", "los_angeles"),
            ("Shanghai", "hamburg"),
            ("Hamburg", "new_york"),
            ("Los_Angeles", "singapore"),
            ("Ningbo", "seattle"),
            ("Shenzhen", "boston"),
        ]

        scenarios = []
        combo_idx = 0

        for i in range(num_test):
            if combo_idx >= len(all_combos):
                combo_idx = 0  # Cycle if needed

            # Get the combination
            combo = all_combos[combo_idx]
            combo_idx += 1

            # Create condition key
            condition_key = "+".join(sorted(combo))

            # Get expected solution from config
            expected_solution = LogisticsConfig.get_valid_solution_for_conditions(
                condition_key
            )

            # Pick a template and src/dst
            template = test_templates[i % len(test_templates)]
            src, dst = src_dst_pairs[i % len(src_dst_pairs)]

            task_description = template.format(src=src, dst=dst)

            scenario = {
                "task": task_description,
                "expected": expected_solution,
                "black_swan_type": "compositional_generalization",
                "precept_lesson": f"Composite condition {condition_key} requires specific handling",
                "phase": "test",
                "test_type": "compositional",
                "tests_learning": "compositional_generalization",
                "multi_condition": {
                    "condition_key": condition_key,
                    "conditions": list(combo),
                    "num_conditions": len(combo),
                    "all_atoms_learned": True,
                },
            }
            scenarios.append(scenario)

        # Optionally add conflict scenarios
        if include_conflicts and len(scenarios) > 0:
            conflict_scenarios = self._generate_conflict_scenarios_from_atoms(
                learned_atoms, min(2, num_test // 4), seed
            )
            scenarios.extend(conflict_scenarios)
            print(f"     Added {len(conflict_scenarios)} conflict scenarios")

        print(f"  ✅ Generated {len(scenarios)} compositional test scenarios")

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
        Generate SEMANTIC compositional tests where composite solutions ARE derivable
        from atomic precepts, enabling P₁ > 0% through LLM reasoning.

        Unlike Black Swan CSPs where solutions are arbitrary, this creates scenarios
        where:
        - Atomic conditions have clear semantic meanings
        - Atomic solutions follow logical patterns
        - Composite solutions combine atomic solutions predictably
        - LLM can reason: "A needs X, B needs Y, so A+B needs X+Y"

        Args:
            num_train: Base number of atomic conditions to train
            num_test: Number of composite test scenarios
            seed: Random seed for reproducibility
            beta: Repetitions per atomic condition (higher = more likely to learn all)
                  With beta=3, each atom is trained 3 times → ~95%+ learning rate
            filter_by_learned: If True, only generates test composites from atoms that
                              were trained. The actual filtering by LEARNED atoms
                              happens post-training in the experiment runner.
            test_num_conditions: Number of conditions to combine in test scenarios (2, 3, 4, etc.)
                                 Supports 1→M compositional generalization where M is this value.

        Returns:
            Tuple of (training_scenarios, test_scenarios, semantic_mappings)
        """
        if seed is not None:
            random.seed(seed)

        print("\n  🧠 SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(
            f"     Beta={beta} (each atomic condition trained {beta}x for robust learning)"
        )
        print(
            f"     Test complexity: {test_num_conditions}-way combinations (1→{test_num_conditions} generalization)"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # SEMANTIC CONDITION DEFINITIONS
        # Each condition has a clear meaning AND maps to a valid port solution
        # The semantic meaning helps LLM reason about which port to use
        # ═══════════════════════════════════════════════════════════════════════
        # VALID PORTS: antwerp, hamburg, ningbo, long_beach
        semantic_conditions = {
            # Regional hub preferences - directly map to ports
            "ASIA": {
                "meaning": "Shipment to/from Asia - use Asian hub port",
                "solution": "ningbo",  # Primary Asian hub
                "reasoning": "Asian shipments route through Ningbo (major Asian port)",
                "tier": 2,
            },
            "EURO": {
                "meaning": "Shipment to/from Europe - use European hub port",
                "solution": "hamburg",  # Primary European hub
                "reasoning": "European shipments route through Hamburg (major EU port)",
                "tier": 2,
            },
            "AMER": {
                "meaning": "Shipment to/from Americas - use American hub port",
                "solution": "long_beach",  # Primary American hub
                "reasoning": "American shipments route through Long Beach (major US port)",
                "tier": 2,
            },
            "INTL": {
                "meaning": "International transshipment - use neutral hub",
                "solution": "antwerp",  # Neutral international hub
                "reasoning": "International transshipments use Antwerp (neutral hub)",
                "tier": 2,
            },
            # Service level preferences - influence port choice
            "FAST": {
                "meaning": "Time-critical express shipment",
                "solution": "long_beach",  # Fastest processing port
                "reasoning": "Express shipments use Long Beach (fastest customs)",
                "tier": 1,
            },
            "ECON": {
                "meaning": "Cost-optimized economical routing",
                "solution": "ningbo",  # Most economical port
                "reasoning": "Economy shipments use Ningbo (lowest costs)",
                "tier": 1,
            },
            "SAFE": {
                "meaning": "Safety-critical cargo requiring secure handling",
                "solution": "hamburg",  # Best safety protocols
                "reasoning": "Safety-critical cargo uses Hamburg (best protocols)",
                "tier": 3,
            },
            "BULK": {
                "meaning": "Bulk cargo requiring high-volume facilities",
                "solution": "antwerp",  # Best bulk handling
                "reasoning": "Bulk cargo uses Antwerp (best bulk facilities)",
                "tier": 1,
            },
        }

        # ═══════════════════════════════════════════════════════════════════════
        # COMPOSITIONAL SOLUTION RULES
        # When combining conditions, use PRIORITY-BASED resolution:
        # - Higher tier wins (SAFE tier=3 > ASIA tier=2 > FAST tier=1)
        # - Same tier: first condition alphabetically
        # This makes composite solutions DERIVABLE from atomic knowledge!
        # ═══════════════════════════════════════════════════════════════════════

        def compute_composite_solution(conditions: List[str]) -> str:
            """
            Compute composite solution using priority-based resolution.

            The LLM can reason: "ASIA needs ningbo (tier 2), FAST needs long_beach (tier 1).
            ASIA has higher priority, so ASIA+FAST → ningbo"

            This makes P₁ > 0% achievable through reasoning!
            """
            if not conditions:
                return "antwerp"  # Default

            # Find highest priority condition
            best_cond = None
            best_tier = -1
            for cond in sorted(conditions):  # Alphabetical for tie-breaking
                if cond in semantic_conditions:
                    tier = semantic_conditions[cond]["tier"]
                    if tier > best_tier:
                        best_tier = tier
                        best_cond = cond

            if best_cond:
                return semantic_conditions[best_cond]["solution"]
            return "antwerp"  # Fallback

        # Select conditions for this experiment
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

        # Use 4-6 conditions for training (atomic), prioritizing diverse solutions
        ordered_conditions = prioritize_solution_diversity(available_conditions)
        train_conditions = ordered_conditions[: min(6, len(ordered_conditions))]

        print(f"     Training conditions: {train_conditions}")
        print("     Each has semantic meaning and predictable solution")

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE TRAINING SCENARIOS (Single conditions - atomic learning)
        # ═══════════════════════════════════════════════════════════════════════
        training_scenarios = []
        task_templates = [
            "Ship cargo from {src} to {dst}",
            "Transport goods from {src} to {dst}",
            "Arrange delivery from {src} to {dst}",
            "Book shipment from {src} to {dst}",
        ]
        # CRITICAL: Destinations MUST match DESTINATION_PORTS in config/logistics.py
        # Valid destinations: boston, new_york, shanghai, singapore, london,
        #                     hamburg, los_angeles, chicago, miami, seattle
        src_dst_pairs = [
            ("Shanghai", "los_angeles"),
            ("Rotterdam", "new_york"),
            ("Hamburg", "singapore"),
            ("Ningbo", "chicago"),
            ("Shenzhen", "miami"),
            ("Busan", "seattle"),
        ]

        # With beta repetitions, each atom gets beta training opportunities
        # This dramatically increases the chance of learning all atomic precepts
        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]  # Now uses valid port names
                template = task_templates[scenario_idx % len(task_templates)]
                src, dst = src_dst_pairs[scenario_idx % len(src_dst_pairs)]

                scenario = {
                    "task": template.format(src=src, dst=dst),
                    "expected": solution,
                    "black_swan_type": "semantic_atomic",
                    "precept_lesson": f"{cond}: {cond_info['meaning']} → use {solution}",
                    "phase": "train",
                    "test_type": "atomic_semantic",
                    "tests_learning": "semantic_compositional",
                    "repetition": rep + 1,  # Track which repetition this is
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

        print(
            f"     Generated {len(training_scenarios)} atomic training scenarios ({len(train_conditions[:num_train])} atoms × {beta} repetitions)"
        )

        # ═══════════════════════════════════════════════════════════════════════
        # GENERATE TEST SCENARIOS (Composite conditions - derivable solutions)
        # ═══════════════════════════════════════════════════════════════════════
        from itertools import combinations

        # Generate M-way combinations from trained conditions
        # For 1→M generalization: train on single atoms, test on M-way combos
        num_atoms_for_combos = min(
            6, len(train_conditions)
        )  # Use up to 6 atoms for combos

        # Ensure we have enough atoms for the requested combination size
        if num_atoms_for_combos < test_num_conditions:
            print(
                f"     ⚠️ Not enough atoms ({num_atoms_for_combos}) for {test_num_conditions}-way combos, using {num_atoms_for_combos}-way"
            )
            test_num_conditions = num_atoms_for_combos

        all_combos = list(
            combinations(train_conditions[:num_atoms_for_combos], test_num_conditions)
        )

        def is_nontrivial_combo(combo: tuple) -> bool:
            solutions = {semantic_conditions[c]["solution"] for c in combo}
            tiers = {semantic_conditions[c]["tier"] for c in combo}
            return len(solutions) > 1 and len(tiers) > 1

        filtered_combos = [combo for combo in all_combos if is_nontrivial_combo(combo)]
        if filtered_combos:
            all_combos = filtered_combos
        else:
            print("     ⚠️ No non-trivial combos found; using full combo set")

        if len(all_combos) < num_test:
            print(
                f"     ⚠️ Only {len(all_combos)} unique non-trivial combos available; "
                f"cycling combos to reach {num_test} tests"
            )
            repeats = num_test - len(all_combos)
            all_combos = all_combos + [
                all_combos[i % len(all_combos)] for i in range(repeats)
            ]

        random.shuffle(all_combos)

        print(
            f"     Generating {test_num_conditions}-way combinations from {num_atoms_for_combos} atoms → {len(all_combos)} possible combos"
        )

        test_scenarios = []
        for i, combo in enumerate(all_combos):
            if i >= num_test:
                break

            condition_key = "+".join(sorted(combo))
            # CRITICAL: Composite solution is DERIVABLE via priority-based resolution
            composite_solution = compute_composite_solution(list(combo))

            # Build semantic context for the composite
            meanings = [semantic_conditions[c]["meaning"] for c in combo]
            solutions = [semantic_conditions[c]["solution"] for c in combo]
            tiers = [semantic_conditions[c]["tier"] for c in combo]
            reasonings = [semantic_conditions[c]["reasoning"] for c in combo]

            # Find which condition "wins" (highest tier)
            winning_idx = tiers.index(max(tiers))
            winning_cond = list(combo)[winning_idx]

            template = task_templates[i % len(task_templates)]
            src, dst = src_dst_pairs[i % len(src_dst_pairs)]

            # Build derivation explanation
            derivation_parts = [
                f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})"
                for c in combo
            ]
            derivation_rule = f"{' vs '.join(derivation_parts)} → {winning_cond} wins → {composite_solution}"

            scenario = {
                "task": template.format(src=src, dst=dst),
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

        print(
            f"     Generated {len(test_scenarios)} semantic {test_num_conditions}-way compositional test scenarios"
        )
        print("     Derivation rule: highest tier condition determines solution")
        print(
            "     LLM can reason: 'A(tier=2) vs B(tier=1) → A wins → use A's solution'"
        )

        # Build semantic mappings for reference
        # Include all atoms that were TRAINED (attempted)
        trained_atom_list = train_conditions[: min(num_train, len(train_conditions))]

        semantic_mappings = {
            "conditions": semantic_conditions,
            "trained_atoms": trained_atom_list,
            "derivation_rule": "composite_solution = solution of highest-tier condition",
            "beta": beta,
            "filter_by_learned": filter_by_learned,
            # This will be populated post-training with actually learned atoms
            "learned_atoms": None,  # Filled by experiment runner after training
            # All test composites and their required atoms (for filtering)
            "test_composite_requirements": {
                "+".join(sorted(list(combo))): list(combo)
                for combo in combinations(
                    trained_atom_list[: min(4, len(trained_atom_list))], 2
                )
            },
        }

        print(f"     Trained atoms: {trained_atom_list}")
        print(
            "     Post-training: will filter tests to only composites where ALL atoms were LEARNED"
        )

        return training_scenarios, test_scenarios, semantic_mappings

    def _generate_conflict_scenarios_from_atoms(
        self,
        learned_atoms: List[str],
        num_scenarios: int,
        seed: int = None,
    ) -> List[Dict]:
        """
        Generate test scenarios with conflicting atomic precepts.

        These test the hierarchical constraint resolution:
        - Atom A requires "use_port_X"
        - Atom B requires "avoid_port_X"
        - Test A+B → Should trigger tier-based resolution

        The conflict is synthetic but tests the resolution mechanism.
        """
        if seed is not None:
            random.seed(seed)

        if len(learned_atoms) < 2:
            return []

        from ..config import LogisticsConfig

        scenarios = []
        templates = [
            "Route shipment from {src} to {dst} with conflicting requirements",
            "Handle freight from {src} to {dst} under mixed constraints",
        ]

        # CRITICAL: Destinations MUST match DESTINATION_PORTS (lowercase)
        src_dst_pairs = [
            ("Rotterdam", "hamburg"),
            ("Shanghai", "los_angeles"),
        ]

        # Create conflict pairs - we'll mark these as having potential conflicts
        conflict_pairs = []
        for i in range(0, len(learned_atoms) - 1, 2):
            if i + 1 < len(learned_atoms):
                conflict_pairs.append((learned_atoms[i], learned_atoms[i + 1]))

        for i in range(min(num_scenarios, len(conflict_pairs))):
            pair = conflict_pairs[i % len(conflict_pairs)]
            condition_key = "+".join(sorted(pair))
            expected_solution = LogisticsConfig.get_valid_solution_for_conditions(
                condition_key
            )

            template = templates[i % len(templates)]
            src, dst = src_dst_pairs[i % len(src_dst_pairs)]
            task_description = template.format(src=src, dst=dst)

            scenario = {
                "task": task_description,
                "expected": expected_solution,
                "black_swan_type": "constraint_conflict",
                "precept_lesson": f"Conflicting requirements {pair[0]} vs {pair[1]} require hierarchical resolution",
                "phase": "test",
                "test_type": "conflict_resolution",
                "tests_learning": "hierarchical_constraint_resolution",
                "multi_condition": {
                    "condition_key": condition_key,
                    "conditions": list(pair),
                    "num_conditions": 2,
                    "has_potential_conflict": True,
                    "conflict_atoms": list(pair),
                },
            }
            scenarios.append(scenario)

        return scenarios

    def generate_all(
        self,
        include_generator_samples: bool = False,
        ensure_coverage: bool = True,
        include_conflict_resolution: bool = True,
        include_fleet_learning: bool = True,
        num_conditions: int = 1,
        test_mode: str = "matched",
    ) -> List[Dict]:
        """
        Generate all logistics scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN INSIGHT: Single-condition is just num_conditions=1, making this
        approach universal and mathematically elegant.

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, GUARANTEE that training covers all error types
            include_conflict_resolution: If True, include conflict resolution scenarios
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default, backward compatible)
                           - num_conditions>1: Multi-condition (for ablation studies)
            test_mode: Testing mode - controls how test condition keys are generated:
                - "matched": Test scenarios REUSE condition keys from training (O(1) lookup test)
                - "random": Test scenarios generate NEW random condition keys (generalization test)

        Returns:
            Combined list of all logistics scenarios

        UNIFIED CONDITION APPROACH:
        ═══════════════════════════════════════════════════════════════════════════
        - Pattern: Entity X + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → Rule Z
        - num_conditions=1: Traditional single-condition learning (baseline behavior)
        - num_conditions=3: 2^3=8 possible states (moderate challenge)
        - num_conditions=5: 2^5=32 possible states (high challenge)
        - num_conditions=10: 2^10=1024 possible states (extreme challenge)

        PRECEPT's DETERMINISTIC ADVANTAGE:
        - Baselines may apply rules with PARTIAL matches (dangerous in multi-condition!)
        - PRECEPT uses CSP - rules apply ONLY when ALL conditions match (safe!)
        ═══════════════════════════════════════════════════════════════════════════
        """
        # Calculate allocations
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training

        # Clamp num_conditions to valid range
        num_conditions = max(1, min(10, num_conditions))

        # ═══════════════════════════════════════════════════════════════════════
        # UNIFIED MULTI-CONDITION APPROACH
        # Single-condition (num_conditions=1) is just a special case!
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

        # Generate scenarios using multi-condition (num_conditions=1 = single-condition)
        scenarios = self.generate_multi_condition_scenarios(
            num_training=total_training,
            num_test=total_test,
            num_conditions=num_conditions,
            test_mode=test_mode,
        )

        # OPTION B: Always include supplementary scenarios for comprehensive testing
        # These test additional PRECEPT capabilities beyond multi-condition learning
        if include_generator_samples:
            generator_scenarios = self.generate_from_universal_generator(num_samples=2)
            scenarios.extend(generator_scenarios)

        if include_conflict_resolution:
            # Conflict resolution tests static vs dynamic knowledge
            conflict_train = max(1, total_training // 4)
            conflict_test = max(1, total_test // 2)
            conflict_scenarios = self.generate_conflict_resolution_scenarios(
                num_training=conflict_train,
                num_test=conflict_test,
                num_conditions=num_conditions,  # Pass num_conditions for consistency
            )
            scenarios.extend(conflict_scenarios)

        if include_fleet_learning:
            # Fleet learning tests cross-entity transfer
            fleet_train = max(1, total_training // 4)
            fleet_test = max(1, total_test // 2)
            fleet_scenarios = self.generate_fleet_learning_scenarios(
                num_training=fleet_train,
                num_test=fleet_test,
                num_conditions=num_conditions,  # Pass num_conditions for consistency
            )
            scenarios.extend(fleet_scenarios)

        return scenarios

    def _generate_with_coverage_guarantee(
        self,
        total_training: int,
        total_test: int,
        num_conditions: int = 1,
    ) -> List[Dict]:
        """
        Generate scenarios with GUARANTEED coverage of all error types in training.

        This ensures that:
        1. Training includes at least one scenario for EACH blocked port
        2. Training includes at least one scenario for EACH customs issue
        3. Test scenarios will ALWAYS have a corresponding learned rule

        Args:
            total_training: Total number of training scenarios
            total_test: Total number of test scenarios
            num_conditions: Number of conditions per scenario (1-10)

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. Port closures - one per blocked port
        for port, port_info in self.config.BLOCKED_PORTS.items():
            # Find a valid destination for this port
            blocked_dests = port_info.get("blocked_destinations", [])

            if blocked_dests:
                # Port has specific blocked destinations (e.g., Hamburg→US)
                # Pick a destination that IS blocked (to trigger the error)
                valid_dests = [
                    d
                    for d, info in self.config.DESTINATIONS.items()
                    if info["region"] in blocked_dests and d.lower() != port.lower()
                ]
            else:
                # Port blocks all destinations
                valid_dests = [
                    d
                    for d in self.config.DESTINATIONS.keys()
                    if d.lower() != port.lower()
                ]

            if valid_dests:
                dest = random.choice(valid_dests)
                template = random.choice(self.config.TRAINING_TEMPLATES)
                cargo = random.choice(list(self.config.CARGO_TYPES.keys()))
                cargo_prefix = self.config.CARGO_TYPES[cargo]["prefix"] or "Standard"

                task = template.format(
                    origin=port.replace("_", " ").title(),
                    destination=dest.replace("_", " ").title(),
                    cargo_prefix=cargo_prefix,
                )

                training_scenarios.append(
                    self._build_scenario(
                        task=task,
                        expected=f"{port_info['error_code']} → {port_info['block_reason']}",
                        black_swan_type="Logistics/Port_Closure",
                        precept_lesson=port_info["lesson"],
                        phase="training",
                    )
                )

        # 1b. Customs issues - one per issue type
        # ═══════════════════════════════════════════════════════════════════════
        # Use config - no hardcoded values
        # ═══════════════════════════════════════════════════════════════════════
        customs_issues = self.config.CUSTOMS_ISSUES
        customs_templates = self.config.CUSTOMS_TRAINING_TEMPLATES

        for issue, info in customs_issues.items():
            # Pick a destination that WILL trigger this error type
            valid_dests = info.get("valid_destinations", ["new_york"])
            dest = random.choice(valid_dests)
            template = random.choice(customs_templates)
            task = template.format(destination=dest.replace("_", " ").title())

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info['solution']}",
                    black_swan_type="Logistics/Customs_Hold",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: FILL REMAINING TRAINING SLOTS (if any)
        # ═══════════════════════════════════════════════════════════════════════

        mandatory_count = len(training_scenarios)
        remaining_training = max(0, total_training - mandatory_count)

        if remaining_training > 0:
            # Generate additional random scenarios
            extra_port = self.generate_port_closure_scenarios(
                num_training=remaining_training // 2,
                num_test=0,
                num_conditions=num_conditions,
            )
            extra_customs = self.generate_customs_scenarios(
                num_training=remaining_training - remaining_training // 2,
                num_test=0,
                num_conditions=num_conditions,
            )

            # Filter to training only
            training_scenarios.extend(
                [s for s in extra_port if s.get("phase") == "training"]
            )
            training_scenarios.extend(
                [s for s in extra_customs if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS
        # ═══════════════════════════════════════════════════════════════════════

        # Test can include any error type - training guarantees we have rules for all
        port_test = max(1, int(total_test * 0.7))
        customs_test = total_test - port_test

        test_port = self.generate_port_closure_scenarios(
            num_training=0, num_test=port_test, num_conditions=num_conditions
        )
        test_customs = self.generate_customs_scenarios(
            num_training=0, num_test=customs_test, num_conditions=num_conditions
        )

        test_scenarios.extend([s for s in test_port if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_customs if s.get("phase") == "test"])

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: LOG COVERAGE
        # ═══════════════════════════════════════════════════════════════════════

        training_error_codes = set()
        for s in training_scenarios:
            expected = s.get("expected", "")
            # Extract error code from "R-482 → labor strike" format
            if "→" in expected:
                error_code = expected.split("→")[0].strip()
                training_error_codes.add(error_code)

        print(
            f"  📋 Coverage Guarantee: Training covers {len(training_error_codes)} error types"
        )
        print(f"     Error codes: {sorted(training_error_codes)}")

        return training_scenarios + test_scenarios


def generate_logistics_scenarios(
    num_samples: int = 10,
    train_ratio: float = 0.6,
    include_generator_samples: bool = False,
    include_conflict_resolution: bool = True,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate logistics black swan scenarios.

    Args:
        num_samples: TOTAL number of scenarios (train + test).
                    Default: 10 → with train_ratio=0.6 produces 6 training + 4 test
        train_ratio: Ratio of training samples (0.0 to 1.0).
                    Default: 0.6 (60% training, 40% test)
        include_generator_samples: Also include UniversalDataGenerator samples.
        include_conflict_resolution: Include scenarios that test PRECEPT's conflict
                    resolution between static and dynamic knowledge. Default: True
        include_fleet_learning: Include fleet learning scenarios for cross-entity
                    transfer testing. Default: True
        num_conditions: Number of conditions per scenario (1-10). Default: 1.
                    Higher values create multi-condition scenarios that challenge
                    baselines exponentially (N=3 → 8 states, N=10 → 1024 states).
        test_mode: Testing mode - controls how test condition keys are generated:
                    - "matched": Test scenarios REUSE condition keys from training (O(1) lookup test)
                    - "random": Test scenarios generate NEW random condition keys (generalization test)

    Returns:
        List of scenario dictionaries with training and test phases
    """
    generator = LogisticsScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_conflict_resolution=include_conflict_resolution,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
