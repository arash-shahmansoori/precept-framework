"""
Coding Scenario Generator for PRECEPT Testing.

This module generates coding black swan scenarios using UniversalDataGenerator.

Uses a data-driven, template-based approach for maintainability.
Configuration is imported from precept.config.coding - single source of truth.

The `num_samples` parameter controls how many random combinations are sampled
from the configurable parameters (modules, entities, contexts, goals, etc.)

Usage:
    from precept.scenario_generators import CodingScenarioGenerator

    generator = CodingScenarioGenerator(num_samples=10)
    scenarios = generator.generate_all()
"""

import random
from typing import Dict, List, Optional

from ..black_swan_gen import BLACK_SWAN_DEFINITIONS, UniversalDataGenerator
from ..config import CodingConfig
from ..config.multi_condition import (
    MultiConditionConfig,
    CodingConditions,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Import from single source of truth in config directory
# ═══════════════════════════════════════════════════════════════════════════════
# Alias for backward compatibility - all config is now in precept.config.coding
CodingScenarioConfig = CodingConfig


class CodingScenarioGenerator:
    """
    Generate coding black swan scenarios using data-driven templates.

    Black Swan Types for Coding:
    - Dependency_Zombie: Agent tries to install a deleted/missing library
    - Opaque_Crash: Agent triggers a Segfault/Bus Error with no stack trace
    - Concurrency_Race: Race condition in check-then-act logic
    - Import_Hell: Circular imports or missing submodules

    All scenario data is centralized in CodingScenarioConfig for maintainability.

    Usage:
        generator = CodingScenarioGenerator()

        # Get dependency scenarios
        scenarios = generator.generate_dependency_scenarios()

        # Get all scenarios
        scenarios = generator.generate_all()
    """

    def __init__(self, num_samples: int = 20, train_ratio: float = 0.8):
        """
        Initialize the generator.

        Args:
            num_samples: TOTAL number of scenarios to generate (train + test combined).
                        This is distributed across all scenario types.
                        Example: num_samples=20, train_ratio=0.8 → 16 training + 4 test
            train_ratio: Ratio of training samples (0.0 to 1.0).
                        Default: 0.8 (80% training, 20% test)
        """
        self.num_samples = num_samples
        self.train_ratio = max(0.1, min(0.9, train_ratio))  # Clamp between 10-90%
        self.generator = UniversalDataGenerator(num_samples=num_samples)
        self.coding_traps = BLACK_SWAN_DEFINITIONS.get("Coding", {})
        self.config = CodingScenarioConfig

        # Map black swan types to error codes from centralized config
        # These are the PRIMARY error codes for each black swan type
        self.error_patterns = {
            "dependency_zombie": CodingScenarioConfig.ERROR_CODE_PATTERNS[
                "ZOMBIE-DEP-404"
            ],
            "opaque_crash": CodingScenarioConfig.ERROR_CODE_PATTERNS["SEGFAULT-000"],
            "concurrency_race": CodingScenarioConfig.ERROR_CODE_PATTERNS[
                "RACE-COND-409"
            ],
            "import_hell": CodingScenarioConfig.ERROR_CODE_PATTERNS["IMPORT-CIRC-500"],
        }

    def generate_from_universal_generator(
        self, num_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Generate scenarios using UniversalDataGenerator.

        Generates exactly `num_samples` scenarios by cycling through trap types
        and generating random variations for each.

        Args:
            num_samples: Number of samples to generate (default: self.num_samples)

        Returns:
            List of scenario dictionaries
        """
        num_samples = num_samples or self.num_samples
        scenarios = []
        trap_items = list(self.coding_traps.items())

        if not trap_items:
            return []

        for i in range(num_samples):
            # Cycle through trap types
            trap_name, trap_def = trap_items[i % len(trap_items)]

            # Generate a random sample (entity, log variant, etc. are randomized)
            sample = self.generator.generate_sample(
                category="Coding",
                sub_type=trap_name,
                definition=trap_def,
            )

            scenarios.append(
                {
                    "task": sample.user_query,
                    "expected": sample.hidden_trap["root_cause"],
                    "black_swan_type": f"Coding/{sample.sub_category}",
                    "precept_lesson": sample.precept_instinct,
                    "ground_truth_log": sample.ground_truth_log,
                    "difficulty": sample.difficulty,
                }
            )

        return scenarios

    def _build_scenario(
        self,
        task: str,
        expected: str,
        black_swan_type: str,
        precept_lesson: str,
        phase: str = None,
        tests_learning: str = None,
    ) -> Dict:
        """
        Build a scenario dictionary with consistent structure.

        Args:
            task: The task description
            expected: Expected outcome/error
            black_swan_type: Category of black swan
            precept_lesson: What PRECEPT should learn
            phase: "training" or "test" (optional)
            tests_learning: Package being tested (for test phase)

        Returns:
            Scenario dictionary
        """
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

    def generate_dependency_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate dependency management scenarios by randomly sampling from configuration.

        COHERENCE GUARANTEE: Each package (e.g., fast_xml) always uses its own
        coherent set of attributes:
        - working_manager: The manager that works for THIS package
        - error_code: The error code for THIS package
        - lesson: The lesson for THIS package
        - description: The description for THIS package

        The random sampling only varies:
        - Which packages are selected
        - Which template phrasing is used

        Args:
            num_training: Number of training scenarios (default: all packages)
            num_test: Number of test scenarios (default: all packages)

        Returns:
            List of dependency scenarios
        """
        training = []
        test_variations = []

        # Build all possible COHERENT combinations
        # Each combo: (package_name, package_info, template)
        # package_info contains ALL coherent attributes for this package
        all_training_combos = []
        all_test_combos = []

        for package, info in self.config.BLOCKED_PACKAGES.items():
            # info contains: working_manager, error_code, lesson, description
            # All these attributes are COHERENT for this specific package
            for template in self.config.DEPENDENCY_TRAINING_TEMPLATES:
                all_training_combos.append((package, info, template))
            for template in self.config.DEPENDENCY_TEST_TEMPLATES:
                all_test_combos.append((package, info, template))

        # Determine sample sizes
        num_training = (
            num_training if num_training is not None else len(all_training_combos)
        )
        num_test = num_test if num_test is not None else len(all_test_combos)

        # Randomly sample COHERENT combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for package, info, template in sampled_training:
            # All attributes come from the SAME package's info dict
            task = template.format(
                package=package, description=info.get("description", package)
            )
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} on pip → use {info['working_manager']}",
                    black_swan_type="Coding/Dependency_Zombie",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Randomly sample COHERENT combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for package, info, template in sampled_test:
            # All attributes come from the SAME package's info dict
            task = template.format(
                package=package, description=info.get("description", package)
            )
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: {package} → {info['working_manager']} (1 step)",
                    black_swan_type="Coding/Dependency_Zombie",
                    precept_lesson=f"PRECEPT skips pip, uses {info['working_manager']} (learned)",
                    phase="test",
                    tests_learning=package,
                )
            )

        return training + test_variations

    def generate_crash_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate crash/segfault scenarios by randomly sampling from configuration.

        COHERENCE GUARANTEE: Each crash_type (e.g., c_extension_crash) always uses
        its own coherent set of attributes:
        - error_code: The error code for THIS crash type
        - expected_cause: The cause for THIS crash type
        - solution: The solution for THIS crash type
        - lesson: The lesson for THIS crash type

        The random sampling only varies:
        - Which crash_type + context combinations are selected
        - Which template phrasing is used

        Args:
            num_training: Number of training scenarios to generate (default: self.num_samples // 2)
            num_test: Number of test scenarios to generate (default: self.num_samples // 2)

        Returns:
            List of crash scenarios with training/test phases
        """
        num_training = num_training or max(2, self.num_samples // 2)
        num_test = num_test or max(2, self.num_samples // 2)

        training = []
        test_variations = []

        # Build all possible combinations for sampling
        all_training_combos = []
        all_test_combos = []

        for crash_type, info in self.config.CRASH_SCENARIOS.items():
            for context in info["contexts"]:
                for template in info["training_templates"]:
                    all_training_combos.append((crash_type, info, context, template))
                for template in info["test_templates"]:
                    all_test_combos.append((crash_type, info, context, template))

        # Randomly sample num_training combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for crash_type, info, context, template in sampled_training:
            task = template.format(context=context)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info.get('expected_cause', info.get('solution', 'use recovery'))}",
                    black_swan_type="Coding/Opaque_Crash",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Randomly sample num_test combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for crash_type, info, context, template in sampled_test:
            task = template.format(context=context)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: use {info['solution']} (1 step)",
                    black_swan_type="Coding/Opaque_Crash",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
                    phase="test",
                    tests_learning=crash_type,
                )
            )

        return training + test_variations

    def generate_concurrency_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate concurrency/race condition scenarios by randomly sampling from configuration.

        COHERENCE GUARANTEE: Each race_type (e.g., check_then_act) always uses
        its own coherent set of attributes:
        - error_code: The error code for THIS race type
        - expected_cause: The cause for THIS race type
        - solution: The solution for THIS race type
        - lesson: The lesson for THIS race type

        The random sampling only varies:
        - Which race_type + entity combinations are selected
        - Which template phrasing is used

        Args:
            num_training: Number of training scenarios to generate (default: self.num_samples // 2)
            num_test: Number of test scenarios to generate (default: self.num_samples // 2)

        Returns:
            List of concurrency scenarios with training/test phases
        """
        num_training = num_training or max(2, self.num_samples // 2)
        num_test = num_test or max(2, self.num_samples // 2)

        training = []
        test_variations = []

        # Build all possible combinations for sampling
        all_training_combos = []
        all_test_combos = []

        for race_type, info in self.config.CONCURRENCY_SCENARIOS.items():
            for entity_info in info["entities"]:
                for template in info["training_templates"]:
                    all_training_combos.append((race_type, info, entity_info, template))
                for template in info["test_templates"]:
                    all_test_combos.append((race_type, info, entity_info, template))

        # Randomly sample num_training combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for race_type, info, entity_info, template in sampled_training:
            task = template.format(**entity_info)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info.get('expected_cause', info.get('solution', 'use recovery'))}",
                    black_swan_type="Coding/Concurrency_Race",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Randomly sample num_test combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for race_type, info, entity_info, template in sampled_test:
            task = template.format(**entity_info)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: use {info['solution']} (1 step)",
                    black_swan_type="Coding/Concurrency_Race",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
                    phase="test",
                    tests_learning=race_type,
                )
            )

        return training + test_variations

    def generate_import_scenarios(
        self,
        num_training: Optional[int] = None,
        num_test: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate import/module scenarios by randomly sampling from configuration.

        COHERENCE GUARANTEE: Each import_type (e.g., circular_import) always uses
        its own coherent set of attributes:
        - error_code: The error code for THIS import issue type
        - expected_cause: The cause for THIS import issue type
        - solution: The solution for THIS import issue type
        - lesson: The lesson for THIS import issue type

        The random sampling only varies:
        - Which import_type + module combinations are selected
        - Which template phrasing is used

        Args:
            num_training: Number of training scenarios to generate (default: self.num_samples // 2)
            num_test: Number of test scenarios to generate (default: self.num_samples // 2)

        Returns:
            List of import scenarios with training/test phases
        """
        num_training = num_training or max(2, self.num_samples // 2)
        num_test = num_test or max(2, self.num_samples // 2)

        training = []
        test_variations = []

        # Build all possible combinations for sampling
        all_training_combos = []
        all_test_combos = []

        for import_type, info in self.config.IMPORT_SCENARIOS.items():
            for module_info in info["modules"]:
                for template in info["training_templates"]:
                    all_training_combos.append(
                        (import_type, info, module_info, template)
                    )
                for template in info["test_templates"]:
                    all_test_combos.append((import_type, info, module_info, template))

        # Randomly sample num_training combinations for training
        sampled_training = random.sample(
            all_training_combos, min(num_training, len(all_training_combos))
        )

        for import_type, info, module_info, template in sampled_training:
            task = template.format(**module_info)
            training.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info.get('expected_cause', info.get('solution', 'use recovery'))}",
                    black_swan_type="Coding/Import_Hell",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # Randomly sample num_test combinations for test
        sampled_test = random.sample(
            all_test_combos, min(num_test, len(all_test_combos))
        )

        for import_type, info, module_info, template in sampled_test:
            task = template.format(**module_info)
            test_variations.append(
                self._build_scenario(
                    task=task,
                    expected=f"PRECEPT applies learned: use {info['solution']} (1 step)",
                    black_swan_type="Coding/Import_Hell",
                    precept_lesson=f"PRECEPT applies {info['solution']} (learned)",
                    phase="test",
                    tests_learning=import_type,
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

        This is the "Fleet Learning" pattern (Development Teams):
        - Training: Project X + Dependency Error Y → learns Rule Z
        - Testing: Project K (DIFFERENT!) + SAME Error Y → applies Rule Z

        GUARANTEED PATTERN:
        ═══════════════════════════════════════════════════════════════════════════
        Training:
          - fast_xml in ML project (PKG-404) → learns "PKG-404 → conda"

        Testing (DIFFERENT projects, SAME dependency issues):
          - fast_xml in Web project (PKG-404) → applies "PKG-404 → conda" ✓
          - fast_xml in API project (PKG-404) → applies "PKG-404 → conda" ✓
        ═══════════════════════════════════════════════════════════════════════════

        Key insight: Rules are learned by ERROR CODE (condition), not by
        specific project/context (entity). This enables cross-entity transfer.
        One developer's discovery benefits all teams.
        """
        num_training = num_training or max(2, int(self.num_samples * self.train_ratio))
        num_test = num_test or max(2, self.num_samples - num_training)

        training = []
        testing = []

        # Get blocked packages and project contexts
        blocked_packages = self.config.BLOCKED_PACKAGES
        project_types = self.config.PROJECT_TYPES if hasattr(self.config, 'PROJECT_TYPES') else [
            "machine_learning", "web_app", "api_service", "data_pipeline", "microservice"
        ]

        # For each blocked package, create:
        # - Training: One project type that learns the rule
        # - Testing: Different project types that apply the rule

        for pkg, pkg_info in blocked_packages.items():
            error_code = pkg_info["error_code"]
            working_manager = pkg_info["working_manager"]
            description = pkg_info.get("description", "library")

            if len(project_types) < 2:
                continue

            # Shuffle project types
            project_list = project_types.copy()
            random.shuffle(project_list)

            # TRAINING: First project learns the rule
            training_project = project_list[0]

            training.append({
                "task": f"Install {pkg} ({description}) for {training_project} project using pip",
                "expected": f"{error_code} → use {working_manager}",
                "black_swan_type": "Coding/FleetLearning_Train",
                "precept_lesson": f"When {pkg} fails with {error_code}, use {working_manager} for ANY project",
                "phase": "training",
                "fleet_learning": {
                    "package": pkg,
                    "error_code": error_code,
                    "learned_manager": working_manager,
                    "training_project": training_project,
                },
            })

            # TESTING: Different projects apply the SAME rule
            for test_project in project_list[1:3]:
                testing.append({
                    "task": f"Setup {pkg} dependency for {test_project} using pip install",
                    "expected": f"Apply learned rule: {error_code} → {working_manager}",
                    "black_swan_type": "Coding/FleetLearning_Test",
                    "precept_lesson": f"Cross-entity transfer: Rule for {error_code} applies to {test_project}",
                    "phase": "test",
                    "tests_learning": f"fleet_learning_{error_code}",
                    "fleet_learning": {
                        "package": pkg,
                        "error_code": error_code,
                        "expected_manager": working_manager,
                        "different_project": test_project,
                        "training_project": training_project,
                    },
                })

        # Trim to requested counts
        training = training[:num_training]
        testing = testing[:num_test]

        print(f"  🚀 Fleet Learning (Coding): {len(training)} train + {len(testing)} test scenarios")
        print("     Pattern: Package + Error Y → Rule Z | Package + Different Project + Error Y → Apply Z")

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
            fast_xml + PKG-404 + ENV-ARM + BLD-NATIV → use conda with native flag

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

        conditions_provider = CodingConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        blocked_packages = self.config.BLOCKED_PACKAGES
        project_types = self.config.PROJECT_TYPES if hasattr(self.config, 'PROJECT_TYPES') else [
            "machine_learning", "web_app", "api_service", "data_pipeline"
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPLATES: NO PACKAGE NAMES - matches Logistics approach!
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Removing package names prevents ExpeL from using them as
        # retrieval anchors. This ensures fair comparison where only
        # condition_key determines the solution (like Logistics).
        # ═══════════════════════════════════════════════════════════════════════
        training_templates = [
            "Install package ({project})",
            "Setup dependency ({project})",
            "Add dependency ({project})",
        ]
        test_templates = [
            "Configure dependency ({project})",
            "Setup package ({project})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: CREATE FIXED KEY POOL FOR β-COVERAGE CONTROL
        # ═══════════════════════════════════════════════════════════════════════
        K = len(blocked_packages)
        beta = max(1, num_training // K)

        print(f"  🔑 FIXED KEY POOL: K={K} unique composite keys, β={beta} coverage")
        print(f"     Train={num_training} → Each key seen {beta} times during training")

        # Pre-generate ONE unique composite key per blocked package
        fixed_key_pool = {}
        for pkg, pkg_info in blocked_packages.items():
            error_code = pkg_info["error_code"]
            working_manager = pkg_info["working_manager"]

            other_conditions = random.sample(
                [c for c in condition_codes if c != error_code],
                min(num_conditions - 1, len(condition_codes) - 1),
            )
            all_conds = sorted([error_code] + other_conditions)
            condition_key = "+".join(all_conds)

            fixed_key_pool[pkg] = {
                "condition_key": condition_key,
                "conditions": all_conds,
                "error_code": error_code,
                "solution": working_manager,
            }

        print(f"     Keys: {list(fixed_key_pool.keys())}")

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: BUILD TRAINING COMBOS USING FIXED KEYS
        # ═══════════════════════════════════════════════════════════════════════
        all_training_combos = []
        all_test_combos = []

        for pkg, pkg_info in blocked_packages.items():
            error_code = pkg_info["error_code"]

            key_info = fixed_key_pool[pkg]
            condition_key = key_info["condition_key"]
            all_conds = key_info["conditions"]
            working_manager = key_info["solution"]

            for project in project_types:
                for template in training_templates:
                    all_training_combos.append({
                        "pkg": pkg,
                        "error_code": error_code,
                        "working_manager": working_manager,
                        "project": project,
                        "template": template,
                        "all_conds": all_conds,
                        "condition_key": condition_key,
                    })

                for template in test_templates:
                    all_test_combos.append({
                        "pkg": pkg,
                        "error_code": error_code,
                        "working_manager": working_manager,
                        "project": project,
                        "template": template,
                        "all_conds": all_conds,
                        "condition_key": condition_key,
                    })

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
            pkg = combo["pkg"]
            project = combo["project"]
            template = combo["template"]
            working_manager = combo["working_manager"]
            all_conds = combo["all_conds"]
            condition_key = combo["condition_key"]

            cond_str = " + ".join(all_conds)
            # NO PACKAGE NAME in task - matches Logistics approach
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(project=project)

            training.append({
                "task": task,
                "expected": f"{condition_key} → {working_manager}",
                "black_swan_type": f"Coding/MultiCondition_{num_conditions}C_Train",
                "precept_lesson": f"When ALL {num_conditions} conditions ({cond_str}) match, use {working_manager}",
                "phase": "training",
                "condition_key": condition_key,
                "test_mode": test_mode,
                "multi_condition": {
                    "num_conditions": num_conditions,
                    "conditions": all_conds,
                    "condition_key": condition_key,
                    "package": pkg,  # Keep for internal tracking
                    "solution": working_manager,
                },
            })

            training_condition_keys[condition_key] = {
                "conditions": all_conds,
                "package": pkg,
                "solution": working_manager,
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

            pkg = combo["pkg"]
            project = combo["project"]
            template = combo["template"]

            matching_keys = [
                (k, v) for k, v in training_condition_keys.items()
                if v["package"] == pkg
            ]
            if not matching_keys:
                continue

            condition_key, key_info = random.choice(matching_keys)
            all_conds = key_info["conditions"]
            solution = key_info["solution"]

            cond_str = " + ".join(all_conds)
            # NO PACKAGE NAME in task - matches Logistics approach
            # BLACK SWAN CSP: NO CONDITIONS IN TASK - prevents ExpeL similarity matching
            task = template.format(project=project)

            testing.append({
                "task": task,
                "expected": f"Apply learned rule: {condition_key} → {solution}",
                "black_swan_type": f"Coding/MultiCondition_{num_conditions}C_Test",
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
            })

        print(f"  🔀 Multi-Condition ({num_conditions}C): {len(training)} train + {len(testing)} test")
        print(f"     Pool: {len(all_training_combos)} training combos, {len(all_test_combos)} test combos")
        print(f"     Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states!")

        return training + testing

    def generate_all(
        self,
        include_generator_samples: bool = True,
        ensure_coverage: bool = True,
        include_fleet_learning: bool = True,
        num_conditions: int = 1,
        test_mode: str = "matched",
    ) -> List[Dict]:
        """
        Generate all coding scenarios using UNIFIED MULTI-CONDITION approach.

        DESIGN: Single-condition (num_conditions=1) is just a special case!

        Args:
            include_generator_samples: Also include UniversalDataGenerator samples
            ensure_coverage: If True, guarantees training covers ALL error types
            include_fleet_learning: If True, include CROSS-ENTITY TRANSFER scenarios
            num_conditions: Number of conditions per scenario (1-10)
                           - num_conditions=1: Single-condition (default)
                           - num_conditions>1: Multi-condition (for ablation)

        Returns:
            Combined list of all coding scenarios
        """
        # Calculate total training and test counts based on ratio
        total_training = int(self.num_samples * self.train_ratio)
        total_test = self.num_samples - total_training

        # Clamp num_conditions to valid range
        num_conditions = max(1, min(10, num_conditions))

        # ═══════════════════════════════════════════════════════════════════════
        # UNIFIED APPROACH: num_conditions=1 is single-condition (special case)
        # ═══════════════════════════════════════════════════════════════════════
        if num_conditions == 1:
            print(f"\n📋 Single-condition mode: {total_training} train + {total_test} test")
        else:
            print(f"\n🔬 Multi-condition mode ({num_conditions}C): {total_training} train + {total_test} test")
            print(f"   Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states")

        # Generate scenarios using multi-condition approach
        scenarios = self.generate_multi_condition_scenarios(
            num_training=total_training,
            num_test=total_test,
            test_mode=test_mode,
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
        include_generator_samples: bool = True,
    ) -> List[Dict]:
        """
        Generate scenarios with GUARANTEED coverage of all error types in training.

        This ensures that:
        1. Training includes at least one scenario for EACH blocked package
        2. Training includes at least one scenario for EACH crash type
        3. Training includes at least one scenario for EACH race condition type
        4. Training includes at least one scenario for EACH import issue type
        5. Test scenarios will ALWAYS have a corresponding learned rule

        Returns:
            List of scenarios with coverage guarantee
        """
        training_scenarios = []
        test_scenarios = []

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: MANDATORY COVERAGE - One scenario per error type
        # ═══════════════════════════════════════════════════════════════════════

        # 1a. Blocked packages - one per package
        for package, info in self.config.BLOCKED_PACKAGES.items():
            template = random.choice(self.config.DEPENDENCY_TRAINING_TEMPLATES)
            task = template.format(
                package=package, description=info.get("description", package)
            )

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} on pip → use {info['working_manager']}",
                    black_swan_type="Coding/Dependency_Zombie",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # 1b. Crash scenarios - one per crash type
        for crash_type, info in self.config.CRASH_SCENARIOS.items():
            context = random.choice(info["contexts"])
            template = random.choice(info["training_templates"])
            task = template.format(context=context)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info.get('expected_cause', info.get('solution', 'use recovery'))}",
                    black_swan_type="Coding/Opaque_Crash",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # 1c. Concurrency scenarios - one per race type
        for race_type, info in self.config.CONCURRENCY_SCENARIOS.items():
            entity_info = random.choice(info["entities"])
            template = random.choice(info["training_templates"])
            task = template.format(**entity_info)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → {info.get('expected_cause', info.get('solution', 'use recovery'))}",
                    black_swan_type="Coding/Concurrency_Race",
                    precept_lesson=info["lesson"],
                    phase="training",
                )
            )

        # 1d. Import scenarios - one per import issue type
        for import_type, info in self.config.IMPORT_SCENARIOS.items():
            module_info = random.choice(info["modules"])
            template = random.choice(info["training_templates"])
            task = template.format(**module_info)

            training_scenarios.append(
                self._build_scenario(
                    task=task,
                    expected=f"{info['error_code']} → use {info['solution']}",
                    black_swan_type="Coding/Import_Hell",
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
            # Distribute extra training slots across scenario types
            per_type = max(1, remaining_training // 4)

            extra_dep = self.generate_dependency_scenarios(
                num_training=per_type, num_test=0
            )
            training_scenarios.extend(
                [s for s in extra_dep if s.get("phase") == "training"]
            )

            extra_crash = self.generate_crash_scenarios(
                num_training=per_type, num_test=0
            )
            training_scenarios.extend(
                [s for s in extra_crash if s.get("phase") == "training"]
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: GENERATE TEST SCENARIOS (evenly distributed)
        # ═══════════════════════════════════════════════════════════════════════

        per_type_test = max(1, total_test // 4)

        test_dep = self.generate_dependency_scenarios(
            num_training=0, num_test=per_type_test
        )
        test_crash = self.generate_crash_scenarios(
            num_training=0, num_test=per_type_test
        )
        test_race = self.generate_concurrency_scenarios(
            num_training=0, num_test=per_type_test
        )
        test_import = self.generate_import_scenarios(
            num_training=0, num_test=per_type_test
        )

        test_scenarios.extend([s for s in test_dep if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_crash if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_race if s.get("phase") == "test"])
        test_scenarios.extend([s for s in test_import if s.get("phase") == "test"])

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: LOG COVERAGE
        # ═══════════════════════════════════════════════════════════════════════

        training_error_codes = set()
        for s in training_scenarios:
            expected = s.get("expected", "")
            if "→" in expected:
                error_code = expected.split("→")[0].strip()
                if " on pip" in error_code:
                    error_code = error_code.split(" on pip")[0].strip()
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
        conditions_provider = CodingConditions()
        all_conditions = conditions_provider.get_all_conditions()
        condition_codes = list(all_conditions.keys())

        # Domain-specific templates
        # ═══════════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP: NO CONDITIONS IN TASK DESCRIPTION!
        # Removing conditions prevents ExpeL from using them for similarity matching.
        # The condition_key is passed only in multi_condition metadata.
        # ═══════════════════════════════════════════════════════════════════════
        test_templates = [
            "Install package ({project})",
            "Setup development environment ({project})",
            "Resolve dependency conflicts ({project})",
            "Configure build system ({project})",
            "Fix import issues ({project})",
        ]

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: Use BLOCKED_PACKAGES for testing, not generic packages!
        # Generic packages (numpy, pandas) are NOT enforced by MCP server.
        # Only BLOCKED_PACKAGES have per-package manager rules.
        # ═══════════════════════════════════════════════════════════════════════
        from ..config import CodingConfig
        blocked_packages = CodingConfig.BLOCKED_PACKAGES
        packages = list(blocked_packages.keys())  # legacy_orm, async_lib, etc.

        # Build reverse mapping: error_code -> package
        error_code_to_package = {}
        for pkg, pkg_info in blocked_packages.items():
            error_code_to_package[pkg_info["error_code"]] = pkg

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
            from ..config import CodingConfig

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
                solution = CodingConfig.get_valid_solution_for_conditions(base_key)
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
                    # CRITICAL: Always keep the package's error code!
                    # This ensures the test uses the correct package for enforcement.
                    # ═══════════════════════════════════════════════════════════
                    package_error_codes = set(error_code_to_package.keys())
                    package_cond = None
                    for c in base_conditions:
                        if c in package_error_codes:
                            package_cond = c
                            break

                    other_conditions = [c for c in base_conditions if c != package_cond]
                    # Ensure keep_count doesn't exceed available conditions
                    keep_count = min(len(other_conditions), max(1, int(len(other_conditions) * 0.6)))
                    replace_count = max(0, len(other_conditions) - keep_count)

                    kept_others = random.sample(
                        other_conditions, keep_count
                    ) if other_conditions else []

                    if package_cond:
                        kept_conditions = [package_cond] + kept_others
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
            # CRITICAL: Determine package from error_code in condition_key!
            # The condition_key contains the package's error_code (e.g., PKG-404).
            # We must use the correct package so MCP enforcement is accurate.
            # ═══════════════════════════════════════════════════════════════════
            pkg = packages[i % len(packages)]  # Default fallback
            for cond in all_conds:
                if cond in error_code_to_package:
                    pkg = error_code_to_package[cond]
                    break

            # Use generic project context (like generate_multi_condition_scenarios)
            projects = ["machine_learning", "web_app", "api_service", "data_pipeline"]
            project = projects[i % len(projects)]

            # BLACK SWAN CSP: NO conditions in task - only project context
            task = template.format(project=project)

            num_conditions = len(all_conds)
            scenarios.append({
                "task": task,
                "expected": f"Apply learned rule: {condition_key} → {solution}",
                "black_swan_type": f"Coding/MultiCondition_{num_conditions}C_Test_{mode.capitalize()}",
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
        Generate SEMANTIC compositional tests for Coding domain where composite
        solutions ARE derivable from atomic precepts, enabling P₁ > 0%.

        Coding semantic conditions map to execution patterns:
        - Tier 3 (Highest): Security - non-negotiable
        - Tier 2 (Middle): Stability/compatibility
        - Tier 1 (Lowest): Performance preferences

        Returns:
            Tuple of (training_scenarios, test_scenarios, semantic_mappings)
        """
        if seed is not None:
            random.seed(seed)

        print(f"\n  🧠 CODING SEMANTIC COMPOSITIONAL TEST: Creating derivable solutions")
        print(f"     Beta={beta} (each atomic condition trained {beta}x)")
        print(f"     Test complexity: {test_num_conditions}-way combinations")

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Solutions MUST match CodingConfig.MULTI_CONDITION_VALID_MANAGERS
        # Valid package managers: conda, poetry
        # ═══════════════════════════════════════════════════════════════════════
        semantic_conditions = {
            # Tier 3 (Highest): Security - non-negotiable
            "SECURE": {
                "meaning": "Security-critical package management",
                "solution": "conda",  # Conda: isolated environments, secure
                "reasoning": "Security requires conda (isolated environments)",
                "tier": 3,
            },
            # Tier 2 (Middle): Stability/compatibility
            "STABLE": {
                "meaning": "Stability-first package management",
                "solution": "conda",  # Conda: stable environments
                "reasoning": "Stability requires conda (reproducible builds)",
                "tier": 2,
            },
            "COMPAT": {
                "meaning": "Legacy compatibility required",
                "solution": "poetry",  # Poetry: broad compatibility
                "reasoning": "Compatibility uses poetry (PyPI support)",
                "tier": 2,
            },
            "ATOMIC": {
                "meaning": "Atomic dependency resolution",
                "solution": "poetry",  # Poetry: lock files
                "reasoning": "Atomic deps use poetry (deterministic locks)",
                "tier": 2,
            },
            # Tier 1 (Lowest): Performance preferences
            "PERF": {
                "meaning": "Performance-optimized builds",
                "solution": "conda",  # Conda: binary packages
                "reasoning": "Performance uses conda (pre-compiled binaries)",
                "tier": 1,
            },
            "PARALLEL": {
                "meaning": "Parallel build support",
                "solution": "poetry",  # Poetry: parallel installs
                "reasoning": "Parallel builds use poetry (concurrent installs)",
                "tier": 1,
            },
            "CONC": {
                "meaning": "Concurrent dependency handling",
                "solution": "conda",  # Conda: parallel solving
                "reasoning": "Concurrent deps use conda (fast solver)",
                "tier": 1,
            },
            "CACHED": {
                "meaning": "Cached dependency management",
                "solution": "poetry",  # Poetry: global cache
                "reasoning": "Cached deps use poetry (shared cache)",
                "tier": 1,
            },
        }

        def compute_composite_solution(conditions: List[str]) -> str:
            if not conditions:
                return "conda"  # Default to conda

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
            return "conda"  # Default to conda

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
            "Implement {func} function",
            "Write handler for {func}",
            "Create {func} module",
            "Build {func} component",
        ]
        functions = ["data_processor", "api_handler", "file_manager", "cache_service", "auth_middleware", "event_handler"]

        scenario_idx = 0
        for rep in range(beta):
            for i, cond in enumerate(train_conditions):
                if i >= num_train:
                    break

                cond_info = semantic_conditions[cond]
                solution = cond_info["solution"]
                template = task_templates[scenario_idx % len(task_templates)]
                func = functions[scenario_idx % len(functions)]

                scenario = {
                    "task": template.format(func=func),
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
            func = functions[i % len(functions)]

            derivation_parts = [f"{c}→{semantic_conditions[c]['solution']}(tier={semantic_conditions[c]['tier']})" for c in combo]
            derivation_rule = f"{' vs '.join(derivation_parts)} → {winning_cond} wins → {composite_solution}"

            scenario = {
                "task": template.format(func=func),
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


def generate_coding_scenarios(
    num_samples: int = 20,
    train_ratio: float = 0.8,
    include_generator_samples: bool = True,
    include_fleet_learning: bool = True,
    num_conditions: int = 1,
    test_mode: str = "matched",
) -> List[Dict[str, str]]:
    """
    Generate coding black swan scenarios.

    Args:
        num_samples: TOTAL number of scenarios to generate (train + test).
                    Default: 20 → with train_ratio=0.8 produces 16 training + 4 test
        train_ratio: Ratio of training samples (0.0 to 1.0).
                    Default: 0.8 (80% training, 20% test)
                    Example: num_samples=20, train_ratio=0.8 → 16 train, 4 test
        include_generator_samples: Also include UniversalDataGenerator samples.
                                  Default: True (adds ~2 extra scenarios)
        include_fleet_learning: Include fleet learning scenarios. Default: True.
        num_conditions: Number of conditions per scenario (1-10). Default: 1.

    Returns:
        List of scenario dictionaries with training and test phases

    Note:
        The actual count may be less than num_samples if it exceeds the
        available unique combinations in the config (~96 max without
        universal generator). For typical use cases (20-50 samples),
        you'll get exactly what you request.

    Example:
        # 20 total: 16 training + 4 test (exact)
        scenarios = generate_coding_scenarios(num_samples=20, train_ratio=0.8)

        # 40 total: 32 training + 8 test (exact)
        scenarios = generate_coding_scenarios(num_samples=40, train_ratio=0.8)
    """
    generator = CodingScenarioGenerator(
        num_samples=num_samples, train_ratio=train_ratio
    )
    return generator.generate_all(
        include_generator_samples=include_generator_samples,
        include_fleet_learning=include_fleet_learning,
        num_conditions=num_conditions,
        test_mode=test_mode,
    )
