"""
Booking Scenario Generator for PRECEPT Testing.

This module generates booking/travel black swan scenarios using template-based variations.
Supports configurable num_samples and train_ratio for flexible train/test splits.

Configuration is imported from precept.config.booking - single source of truth.

Usage:
    from precept.scenario_generators import BookingScenarioGenerator

    generator = BookingScenarioGenerator(num_samples=20, train_ratio=0.6)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import BookingConfig
from ..config.multi_condition import (
    BookingConditions,
    MultiConditionConfig,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.booking
BookingScenarioConfig = BookingConfig


class BookingScenarioGenerator:
    """
    Generate booking/travel black swan scenarios using template-based variations.

    COHERENCE GUARANTEE: Each scenario maintains semantic consistency:
    - Flight failures use the correct error codes
    - Working alternatives match the blocked flight's actual alternatives
    - Lessons are specific to the flight/booking combination

    Usage:
        generator = BookingScenarioGenerator(num_samples=20, train_ratio=0.6)
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
        self.booking_traps = BLACK_SWAN_DEFINITIONS.get("Booking", {})
        self.config = BookingScenarioConfig

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

        for trap_name, trap_def in self.booking_traps.items():
            sample = self.generator.generate_sample(
                category="Booking",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"Booking/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios[:num_samples]

    def generate_reservation_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate flight reservation scenarios with template-based variations.

        COHERENCE GUARANTEE: Each flight (e.g., AA-999) always uses:
        - Its own error_code (HTTP-200-PHANTOM)
        - Its own working_alternative (DL-123)
        - Its own failure reason and lesson

        Returns:
            List of reservation scenarios with training/test phases
        """
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training
        num_training = num_training if num_training is not None else total_training
        num_test = num_test if num_test is not None else total_test

        # Build all possible COHERENT combinations
        all_training_combos = []
        all_test_combos = []

        for flight, flight_info in self.config.BLOCKED_FLIGHTS.items():
            for route_code, route_info in self.config.ROUTES.items():
                for context_key, context_info in self.config.BOOKING_CONTEXTS.items():
                    for template in self.config.TRAINING_TEMPLATES:
                        all_training_combos.append(
                            (
                                flight,
                                flight_info,
                                route_code,
                                route_info,
                                context_key,
                                context_info,
                                template,
                            )
                        )
                    for template in self.config.TEST_TEMPLATES:
                        all_test_combos.append(
                            (
                                flight,
                                flight_info,
                                route_code,
                                route_info,
                                context_key,
                                context_info,
                                template,
                            )
                        )

        training = []
        test_variations = []

        # Randomly sample COHERENT combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for (
            flight,
            flight_info,
            route_code,
            route_info,
            context_key,
            context_info,
            template,
        ) in sampled_training:
            task = template.format(
                flight=flight,
                route=f"{route_info['origin']} to {route_info['destination']}",
                context=context_info["prefix"],
            )
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{flight_info['error_code']} → {flight_info.get('block_reason', flight_info.get('failure_reason', 'blocked'))}",
                    black_swan_type="Booking/Phantom_Inventory",
                    precept_lesson=flight_info["lesson"],
                    phase="training",
                )
            )

        # Randomly sample COHERENT combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for (
            flight,
            flight_info,
            route_code,
            route_info,
            context_key,
            context_info,
            template,
        ) in sampled_test:
            task = template.format(
                flight=flight,
                route=f"{route_info['origin']} to {route_info['destination']}",
                context=context_info["prefix"],
            )
            # Use first working alternative from list
            working_alt = flight_info["working_alternatives"][0]
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: skip {flight} → {working_alt} (1 step)",
                    black_swan_type="Booking/Phantom_Inventory",
                    precept_lesson=f"PRECEPT skips {flight}, uses {working_alt} (learned)",
                    phase="test",
                    tests_learning=flight,
                )
            )

        return training + test_variations

    def generate_payment_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate payment processing scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        training_templates = [
            "Process payment for booking {booking_id}",
            "Charge card for reservation {booking_id}",
            "Complete purchase for order {booking_id}",
        ]

        test_templates = [
            "Retry payment for {booking_id}",
            "Finalize charge for booking {booking_id}",
            "Process refund for {booking_id}",
        ]

        booking_ids = ["Resv-882", "Resv-445", "Resv-773", "Resv-991", "Resv-556"]

        # Build combinations
        all_training = []
        all_test = []

        for issue, info in self.config.PAYMENT_ISSUES.items():
            for booking_id in booking_ids:
                for template in training_templates:
                    all_training.append((issue, info, booking_id, template))
                for template in test_templates:
                    all_test.append((issue, info, booking_id, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for issue, info, booking_id, template in sampled_training:
            task = template.format(booking_id=booking_id)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info['description']}",
                    black_swan_type="Booking/Gateway_Timeout",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for issue, info, booking_id, template in sampled_test:
            task = template.format(booking_id=booking_id)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {info['solution']} (1 step)",
                    black_swan_type="Booking/Gateway_Timeout",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
                    phase="test",
                    tests_learning=issue,
                )
            )

        return training + test_variations

    def generate_inventory_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate inventory/availability scenarios with template-based variations.
        """
        num_training = num_training or max(2, self.num_samples // 4)
        num_test = num_test or max(1, self.num_samples // 4)

        training_templates = [
            "Book last available seat for {route}",
            "Reserve final room at {hotel}",
            "Secure remaining inventory for {route}",
        ]

        test_templates = [
            "Grab last seat on {route}",
            "Book remaining room at {hotel}",
            "Finalize last-minute booking for {route}",
        ]

        items = [
            {"route": "JFK-LAX", "hotel": "Marriott NYC"},
            {"route": "LAX-ORD", "hotel": "Hilton Chicago"},
            {"route": "SFO-SEA", "hotel": "Hyatt Seattle"},
        ]

        # Build combinations
        all_training = []
        all_test = []

        for issue, info in self.config.INVENTORY_ISSUES.items():
            for item in items:
                for template in training_templates:
                    all_training.append((issue, info, item, template))
                for template in test_templates:
                    all_test.append((issue, info, item, template))

        training = []
        test_variations = []

        # Sample training
        sampled_training = random.sample(
            all_training, min(num_training, len(all_training))
        )
        for issue, info, item, template in sampled_training:
            task = template.format(**item)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info['description']}",
                    black_swan_type="Booking/Overbooking",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Sample test
        sampled_test = random.sample(all_test, min(num_test, len(all_test)))
        for issue, info, item, template in sampled_test:
            task = template.format(**item)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies: {info['solution']} (1 step)",
                    black_swan_type="Booking/Overbooking",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
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

        This is the "Fleet Learning" pattern (like Tesla's fleet):
        - Training: Flight X with Error Y → learns Rule Z
        - Testing: Flight K (DIFFERENT!) with SAME Error Y → applies Rule Z

        Args:
            num_conditions: Number of conditions per scenario (for interface consistency)

        GUARANTEED PATTERN:
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - AA-999 on NYC→LA route (BK-303) → learns "BK-303 → UA-200"

        Testing (DIFFERENT entities, SAME conditions):
          - AA-999 on CHI→SF route (BK-303) → applies "BK-303 → UA-200" ✓
          - AA-999 on BOS→MIA route (BK-303) → applies "BK-303 → UA-200" ✓
        ═══════════════════════════════════════════════════════════════════════════

        Key insight: Rules are learned by ERROR CODE (condition), not by
        specific route (entity). This enables cross-entity transfer.
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Get blocked flights and routes
        blocked_flights = self.config.BLOCKED_FLIGHTS
        routes = list(self.config.ROUTES.keys())
        contexts = list(self.config.BOOKING_CONTEXTS.keys())

        # For each blocked flight, create:
        # - Training: One route that learns the rule
        # - Testing: Different routes that apply the rule

        for flight, flight_info in blocked_flights.items():
            error_code = flight_info["error_code"]
            alternatives = flight_info.get(
                "working_alternatives", self.config.WORKING_FLIGHTS
            )
            alt = alternatives[0] if alternatives else "UA-200"

            if len(routes) < 2:
                continue

            # Shuffle routes
            route_list = routes.copy()
            random.shuffle(route_list)

            # TRAINING: First route learns the rule
            training_route = route_list[0]
            route_info = self.config.ROUTES[training_route]
            context = random.choice(contexts)
            context_info = self.config.BOOKING_CONTEXTS[context]

            template = random.choice(self.config.TRAINING_TEMPLATES)
            task = template.format(
                flight=flight,
                route=f"{route_info['origin']} to {route_info['destination']}",
                context=context_info["prefix"],
            )

            training.append(
                {
                    "task": task,
                    "expected": f"{error_code} → use {alt}",
                    "black_swan_type": "Booking/FleetLearning_Train",
                    "precept_lesson": f"When {flight} is blocked ({error_code}), use {alt} for ANY route",
                    "phase": "training",
                    "fleet_learning": {
                        "blocked_flight": flight,
                        "error_code": error_code,
                        "learned_alternative": alt,
                        "training_route": training_route,
                    },
                }
            )

            # TESTING: Different routes apply the SAME rule
            for test_route in route_list[1:3]:
                test_route_info = self.config.ROUTES[test_route]
                test_context = random.choice(contexts)
                test_context_info = self.config.BOOKING_CONTEXTS[test_context]

                test_template = random.choice(self.config.TEST_TEMPLATES)
                test_task = test_template.format(
                    flight=flight,
                    route=f"{test_route_info['origin']} to {test_route_info['destination']}",
                    context=test_context_info["prefix"],
                )

                testing.append(
                    {
                        "task": test_task,
                        "expected": f"Apply learned rule: {error_code} → {alt}",
                        "black_swan_type": "Booking/FleetLearning_Test",
                        "precept_lesson": f"Cross-entity transfer: Rule for {error_code} applies to {test_route} route",
                        "phase": "test",
                        "tests_learning": f"fleet_learning_{error_code}",
                        "fleet_learning": {
                            "blocked_flight": flight,
                            "error_code": error_code,
                            "expected_alternative": alt,
                            "different_route": test_route,
                            "training_route": training_route,
                        },
                    }
                )

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(
            f"  🚀 Fleet Learning (Booking): {len(training)} train + {len(testing)} test scenarios"
        )
        print(
            "     Pattern: Flight X + Error Y → Rule Z | Flight X + Different Route + Error Y → Apply Z"
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
            AA-999 + BK-401 + CX-VIP + TM-PKS → use UA-200 with priority boarding

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

        conditions_provider = BookingConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        blocked_flights = self.config.BLOCKED_FLIGHTS
        routes = list(self.config.ROUTES.keys())

        # Templates for variation
        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATES: NO FLIGHT NAMES - matches Logistics approach!
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Removing flight names prevents ExpeL from using them as
        # retrieval anchors. This ensures fair comparison where only
        # condition_key determines the solution (like Logistics).
        # ═══════════════════════════════════════════════════════════════════════
        training_templates = [
            "Book flight ({origin} to {dest})",
            "Reserve flight ({origin} to {dest})",
            "Get tickets ({origin} to {dest})",
        ]
        test_templates = [
            "Reserve flight ({origin} to {dest})",
            "Book seats ({origin} to {dest})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        # Instead of random sampling, pre-generate K unique composite keys and
        # reuse them with β coverage. This mirrors logistics approach.
        #
        # Formula: β = num_training / K
        # Example: K=5 flights, train=15 → β=3 (each key seen 3 times)
        # ═══════════════════════════════════════════════════════════════════════

        K = len(blocked_flights)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per blocked flight
        fixed_key_pool = {}
        for flight, flight_info in blocked_flights.items():
            error_code = flight_info["error_code"]
            alternatives = flight_info.get(
                "working_alternatives", self.config.WORKING_FLIGHTS
            )
            alt = alternatives[0] if alternatives else "UA-200"

            # Sample (num_conditions - 1) OTHER conditions deterministically per flight
            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            fixed_key_pool[flight] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": alt,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD TRAINING COMBOS USING FIXED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        all_training_combos = []
        all_test_combos = []

        for flight, flight_info in blocked_flights.items():
            error_code = flight_info["error_code"]

            # USE FIXED KEY from pool
            key_info = fixed_key_pool[flight]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]
            alt = key_info["solution"]

            for route in routes:
                route_info = self.config.ROUTES[route]
                for template in training_templates:
                    all_training_combos.append(
                        {
                            "flight": flight,
                            "error_code": error_code,
                            "alt": alt,
                            "route": route,
                            "route_info": route_info,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
                        }
                    )

                for template in test_templates:
                    all_test_combos.append(
                        {
                            "flight": flight,
                            "error_code": error_code,
                            "alt": alt,
                            "route": route,
                            "route_info": route_info,
                            "template": template,
                            "all_conds": all_conds,
                            "condition_key": condition_key,
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

        print(f"     ✓ Guaranteed β-coverage: {len(combos_by_key)} keys × {beta} = {len(combos_by_key) * beta} scenarios")

        training = []
        training_condition_keys = {}

        for combo in sampled_training:
            flight = combo["flight"]
            route_info = combo["route_info"]
            template = combo["template"]
            alt = combo["alt"]
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(
                origin=route_info["origin"],
                dest=route_info["destination"],
            )

            training.append(
                {
                    "task": task,
                    "expected": f"{condition_key} → {alt}",
                    "black_swan_type": f"Booking/MultiCondition_{num_conditions}C_Train",
                    "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {alt}",
                    "phase": "training",
                    "condition_key": condition_key,
                    "test_mode": test_mode,
                    "multi_condition": {
                        "num_conditions": num_conditions,
                        "conditions": all_conds,
                        "condition_key": condition_key,
                        "flight": flight,  # Keep for internal tracking
                        "solution": alt,
                    },
                }
            )

            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "flight": flight,
                "solution": alt,
            }

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS MATCHING LEARNED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        testing = []
        sampled_test = random.sample(
            all_test_combos, min(num_test * 2, len(all_test_combos))
        )

        for combo in sampled_test:
            if len(testing) >= num_test:
                break

            flight = combo["flight"]
            route_info = combo["route_info"]
            template = combo["template"]

            matching_keys = [
                (k, v)
                for k, v in training_condition_keys.items()
                if v["flight"] == flight
            ]
            if not matching_keys:
                continue

            condition_key, key_info = random.choice(matching_keys)
            all_conds = key_info["conditions"]
            solution = key_info["solution"]

            cond_str = " + ".join(all_conds)
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(
                origin=route_info["origin"],
                dest=route_info["destination"],
            )

            testing.append(
                {
                    "task": task,
                    "expected": f"Apply learned rule: {condition_key} → {solution}",
                    "black_swan_type": f"Booking/MultiCondition_{num_conditions}C_Test",
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
        Generate all booking scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN: Single-condition (num_conditions=1) is just a special case!

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, guarantees training covers ALL error types
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default)
                           - num_conditions>1: Multi-condition (for ablation)
            test_mode: "matched" (reuse training keys) or "random" (new random keys)
                       This is used by the experiment runner for post-generation filtering.

        Returns:
            Combined list of all booking scenarios
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
        1. Training includes at least one scenario for EACH blocked flight
        2. Training includes at least one scenario for EACH payment issue
        3. Training includes at least one scenario for EACH inventory issue
        4. Test scenarios will ALWAYS have a corresponding learned rule

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. Blocked flights - one per flight
        for flight, flight_info in self.config.BLOCKED_FLIGHTS.items():
            route = random.choice(list(self.config.ROUTES.keys()))
            route_info = self.config.ROUTES[route]
            context = random.choice(list(self.config.BOOKING_CONTEXTS.keys()))
            context_info = self.config.BOOKING_CONTEXTS[context]
            template = random.choice(self.config.TRAINING_TEMPLATES)

            task = template.format(
                flight=flight,
                route=f"{route_info['origin']} to {route_info['destination']}",
                context=context_info["prefix"],
            )

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{flight_info['error_code']} → {flight_info['block_reason']}",
                    black_swan_type="Booking/Phantom_Inventory",
                    precept_lesson=flight_info["lesson"],
                    phase="training",
                )
            )

        # 1b. Payment issues - one per issue type
        for issue, issue_info in self.config.PAYMENT_ISSUES.items():
            gateway = random.choice(issue_info.get("valid_gateways", ["stripe"]))
            booking_id = f"BK-{random.randint(1000, 9999)}"
            template = random.choice(self.config.PAYMENT_TRAINING_TEMPLATES)

            task = template.format(gateway=gateway, booking_id=booking_id)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['solution']}",
                    black_swan_type="Booking/Gateway_Timeout",
                    precept_lesson=issue_info["lesson"],
                    phase="training",
                )
            )

        # 1c. Inventory issues - one per issue type
        for issue, issue_info in self.config.INVENTORY_ISSUES.items():
            route = random.choice(list(self.config.ROUTES.keys()))
            route_info = self.config.ROUTES[route]
            template = random.choice(self.config.INVENTORY_TRAINING_TEMPLATES)

            task = template.format(
                route=f"{route_info['origin']} to {route_info['destination']}"
            )

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{issue_info['error_code']} → {issue_info['solution']}",
                    black_swan_type="Booking/Overbooking",
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
            extra_res = self.generate_reservation_scenarios(
                num_training=remaining_training // 2, num_test=0
            )
            extra_pay = self.generate_payment_scenarios(
                num_training=remaining_training - remaining_training // 2, num_test=0
            )

            training_scenarios.extend(
                [s for s in extra_res if s.get("phase") == "training"]
            )
            training_scenarios.extend(
                [s for s in extra_pay if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS
        # ═══════════════════════════════════════════════════════════════════════

        res_test = max(1, int(total_test * 0.5))
        pay_test = max(1, int(total_test * 0.3))
        inv_test = total_test - res_test - pay_test

        test_res = self.generate_reservation_scenarios(
            num_training=0, num_test=res_test
        )
        test_pay = self.generate_payment_scenarios(num_training=0, num_test=pay_test)
        test_inv = self.generate_inventory_scenarios(num_training=0, num_test=inv_test)

        test_scenarios.extend([s for s in test_res if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_pay if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_inv if s.get("phase") == "test"])

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

        # Optional: add generator samples
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
        conditions_provider = BookingConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Domain-specific templates
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Book flight ({origin} to {dest})",
            "Reserve seat ({origin} to {dest})",
            "Purchase ticket ({origin} to {dest})",
            "Arrange travel ({origin} to {dest})",
            "Schedule flight booking ({origin} to {dest})",
        ]

        # Domain-specific destinations
        destinations = ["Paris", "London", "Tokyo", "New_York", "Sydney", "Singapore"]

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
            from ..config import BookingConfig

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
                solution = BookingConfig.get_valid_solution_for_conditions(base_key)
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
                    # Ensure keep_count doesn't exceed available conditions
                    keep_count = min(len(base_conditions), max(1, int(len(base_conditions) * 0.6)))
                    replace_count = max(0, len(base_conditions) - keep_count)

                    kept_conditions = random.sample(
                        base_conditions, keep_count
                    ) if base_conditions else []
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
            dst = destinations[i % len(destinations)]
            origins = ["New_York", "London", "Tokyo", "Sydney", "Berlin"]
            origin = origins[i % len(origins)]

            # BLACK SWAN CSP: NO conditions in task - only origin/dest context
            task = template.format(origin=origin.title(), dest=dst.title())

            num_conditions = len(all_conds)
            scenarios.append({
                "task": task,
                "expected": f"Apply learned rule: {condition_key} → {solution}",
                "black_swan_type": f"Booking/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
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
            })

        mode_desc = "MATCHED (exact keys)" if mode == "matched" else "MIXED (50% exact + 50% novel)"
        print(f"  📋 Generated {len(scenarios)} test scenarios from learned keys ({mode_desc})")
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
        Generate SEMANTIC compositional tests for Booking domain where composite
        solutions ARE derivable from atomic precepts, enabling P₁ > 0%.

        Booking semantic conditions map to flight types:
        - Tier 3 (Highest): Protection - non-negotiable
        - Tier 2 (Middle): Flexibility requirements
        - Tier 1 (Lowest): Cost/convenience preferences

        Returns:
            Tuple of (training_scenarios, test_scenarios, semantic_mappings)
        """
        if seed is not None:
            random.seed(seed)

        print(f"\n  🧠 BOOKING SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(f"     Beta={beta} (each atomic condition trained {beta}x)")
        print(f"     Test complexity: {test_num_conditions}-way combinations")

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Solutions MUST match BookingConfig.MULTI_CONDITION_VALID_FLIGHTS
        # Valid flights: DL-123, UA-200 (tool validates against these exact IDs)
        # ═══════════════════════════════════════════════════════════════════════
        semantic_conditions = {
            # Tier 3 (Highest): Protection - non-negotiable
            "CANCEL": {
                "meaning": "Free cancellation required",
                "solution": "DL-123",  # Delta: best cancellation policy
                "reasoning": "Cancellation protection requires Delta (DL-123)",
                "tier": 3,
            },
            # Tier 2 (Middle): Flexibility requirements
            "REFUND": {
                "meaning": "Fully refundable ticket",
                "solution": "DL-123",  # Delta: best refund policy
                "reasoning": "Refundable tickets use Delta (DL-123)",
                "tier": 2,
            },
            "CHANGE": {
                "meaning": "Free date change allowed",
                "solution": "UA-200",  # United: flexible changes
                "reasoning": "Date changes use United (UA-200)",
                "tier": 2,
            },
            "BUSI": {
                "meaning": "Business travel requirements",
                "solution": "UA-200",  # United: business class
                "reasoning": "Business travel uses United (UA-200)",
                "tier": 2,
            },
            # Tier 1 (Lowest): Cost/convenience preferences
            "CHEAP": {
                "meaning": "Budget-conscious booking",
                "solution": "UA-200",  # United: budget options
                "reasoning": "Cost savings use United (UA-200)",
                "tier": 1,
            },
            "FAST": {
                "meaning": "Fastest route preferred",
                "solution": "DL-123",  # Delta: faster routes
                "reasoning": "Speed priority uses Delta (DL-123)",
                "tier": 1,
            },
            "NIGHT": {
                "meaning": "Overnight travel acceptable",
                "solution": "UA-200",  # United: redeye flights
                "reasoning": "Night travel uses United (UA-200)",
                "tier": 1,
            },
            "CONN": {
                "meaning": "Connections acceptable for savings",
                "solution": "DL-123",  # Delta: connecting hubs
                "reasoning": "Connection savings use Delta (DL-123)",
                "tier": 1,
            },
        }

        def compute_composite_solution(conditions: List[str]) -> str:
            if not conditions:
                return "DL-123"  # Default to Delta

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
            return "direct"

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
            "Book flight from {src} to {dst}",
            "Reserve travel from {src} to {dst}",
            "Find flights from {src} to {dst}",
            "Arrange journey from {src} to {dst}",
        ]
        routes = [("NYC", "LAX"), ("SFO", "ORD"), ("BOS", "MIA"), ("SEA", "DFW"), ("ATL", "DEN"), ("PHX", "JFK")]

        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]
                template = task_templates[scenario_idx % len(task_templates)]
                src, dst = routes[scenario_idx % len(routes)]

                scenario = {
                    "task": template.format(src=src, dst=dst),
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
            src, dst = routes[i % len(routes)]

            derivation_parts = [f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})" for c in combo]
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


def generate_booking_scenarios(
    num_samples: int = 10,
    train_ratio: float = 0.6,
    include_generator_samples: bool = False,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate booking black swan scenarios.

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
    generator = BookingScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
