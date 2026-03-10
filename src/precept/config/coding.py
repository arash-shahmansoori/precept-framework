"""
Coding Domain Configuration for PRECEPT.

Single source of truth for all coding-related configuration including:
- Blocked packages and error codes
- Package managers and their capabilities
- Crash scenarios and recovery strategies
- Concurrency issues and solutions
- Import problems and fixes
- Scenario generation templates

Usage:
    from precept.config import CodingConfig

    # Access configuration
    config = CodingConfig
    blocked_packages = config.BLOCKED_PACKAGES
    crash_scenarios = config.CRASH_SCENARIOS
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List


@dataclass(frozen=True)
class CodingConfig:
    """
    Centralized configuration for coding domain.

    SINGLE SOURCE OF TRUTH for all coding-related data:
    - Package information and error codes
    - Package manager mappings
    - Crash scenario configurations
    - Concurrency issue patterns
    - Import problem configurations
    - Scenario generation templates

    COHERENCE GUARANTEE: Each package/error has consistent attributes:
    - error_code: The error for THIS specific package/scenario
    - working_solution: What works when THIS scenario fails
    - lesson: The lesson specific to THIS failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCKED PACKAGES CONFIGURATION
    # Maps package → (error_code, working_manager, block_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "PKG-404" means try conda.
    # Error codes do NOT hint at zombie deps, deprecation, etc.
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCKED_PACKAGES: ClassVar[Dict[str, Dict]] = {
        "fast_xml": {
            "error_code": "PKG-404",  # Vague: doesn't reveal zombie dep
            "working_manager": "conda",
            "working_alternatives": ["conda", "source_build"],
            "block_reason": "package deleted from PyPI",
            "lesson": "fast_xml removed from PyPI, only available via conda forge",
            "description": "XML parsing library",
            "pip_blocked": True,
            "error_message": "Installation failed. Error: PKG-404. Package not found.",
        },
        "auth_lib_v1": {
            "error_code": "PKG-505",  # Vague: doesn't reveal deprecation
            "working_manager": "poetry",
            "working_alternatives": ["poetry", "pipenv"],
            "block_reason": "package deprecated, only cached versions exist",
            "lesson": "auth_lib_v1 deprecated, poetry has cached version in lockfile",
            "description": "legacy authentication library",
            "pip_blocked": True,
            "error_message": "Could not install package. Code: PKG-505.",
        },
        "numpy_mkl": {
            "error_code": "PKG-606",  # Vague: doesn't reveal variant issue
            "working_manager": "conda",
            "working_alternatives": ["conda"],
            "block_reason": "MKL variant only in conda channels",
            "lesson": "numpy_mkl (Intel MKL variant) only available via conda",
            "description": "optimized matrix operations",
            "pip_blocked": True,
            "error_message": "Package resolution error. Reference: PKG-606.",
        },
        "legacy_orm": {
            "error_code": "PKG-707",  # Vague: doesn't reveal circular import
            "working_manager": "poetry",  # Changed to poetry (in MULTI_CONDITION_VALID_MANAGERS)
            "working_alternatives": ["poetry"],  # ONLY poetry works
            "block_reason": "circular imports require isolated environment",
            "lesson": "legacy_orm has circular imports, poetry isolation required",
            "description": "legacy database ORM",
            "pip_blocked": True,
            "error_message": "Import error during install. Code: PKG-707.",
        },
        "gpu_compute": {
            "error_code": "PKG-808",  # Vague: doesn't reveal CUDA requirement
            "working_manager": "conda",
            "working_alternatives": ["conda"],
            "block_reason": "requires CUDA toolkit pre-installed",
            "lesson": "gpu_compute needs CUDA, use conda for bundled CUDA",
            "description": "GPU computation library",
            "pip_blocked": True,
            "error_message": "Build failed. Error: PKG-808. Missing dependencies.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PACKAGE MANAGERS - Available options for installation
    # ═══════════════════════════════════════════════════════════════════════════
    # 8 managers to match logistics difficulty (2/8 = 25% vs logistics 2/9 = 22%)
    PACKAGE_MANAGERS: ClassVar[List[str]] = [
        "pip",
        "conda",
        "poetry",
        "pipenv",
        "uv",  # Modern fast package manager
        "mamba",  # Conda alternative
        "pdm",  # PEP 582 package manager
        "hatch",  # Modern Python project manager
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN PACKAGES - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_PACKAGES: ClassVar[List[str]] = [
        # Blocked packages
        "fast_xml",
        "auth_lib_v1",
        "numpy_mkl",
        "legacy_orm",
        "gpu_compute",
        # Normal packages (no issues)
        "numpy",
        "pandas",
        "requests",
        "flask",
        "django",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "lxml",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CRASH SCENARIOS CONFIGURATION
    # Maps crash_type → (error_code, solution, lesson, contexts)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "EXE-101" means use pure Python fallback.
    # Error codes do NOT reveal segfault, bus error, etc.
    # ═══════════════════════════════════════════════════════════════════════════
    CRASH_SCENARIOS: ClassVar[Dict[str, Dict]] = {
        "c_extension_crash": {
            "error_code": "EXE-101",  # Vague: doesn't reveal segfault
            "solution": "pure_python_fallback",
            "working_alternatives": ["pure_python_fallback", "enable_faulthandler"],
            "lesson": "C extension segfault, enable faulthandler and try pure Python fallback",
            "expected_cause": "Memory corruption in C extension",
            "error_message": "Process terminated unexpectedly. Error: EXE-101.",
            "training_templates": [
                "Run optimized C-wrapper for {context}",
                "Execute native extension for {context}",
            ],
            "test_templates": [
                "Use C-accelerated module for {context}",
                "Run compiled extension in {context}",
            ],
            "contexts": [
                "data processing",
                "image manipulation",
                "numerical computation",
            ],
        },
        "memory_alignment": {
            "error_code": "EXE-202",  # Vague: doesn't reveal bus error
            "solution": "streaming_mode",
            "working_alternatives": ["streaming_mode", "reduce_batch_size"],
            "lesson": "Bus error from memory alignment, use streaming mode or reduce batch size",
            "expected_cause": "Memory alignment issue with large data",
            "error_message": "Runtime error. Code: EXE-202. Operation aborted.",
            "training_templates": [
                "Execute binary with {context}",
                "Process {context} in memory",
            ],
            "test_templates": [
                "Load {context} into buffer",
                "Run batch operation on {context}",
            ],
            "contexts": ["large dataset", "high-resolution images", "large matrices"],
        },
        "stack_overflow": {
            "error_code": "EXE-303",  # Vague: doesn't reveal stack overflow
            "solution": "increase_recursion_limit",
            "working_alternatives": [
                "increase_recursion_limit",
                "iterative_implementation",
            ],
            "lesson": "Stack overflow in recursive call, increase limit or use iterative approach",
            "expected_cause": "Deep recursion exceeding stack limit",
            "error_message": "Execution limit exceeded. Reference: EXE-303.",
            "training_templates": [
                "Process deeply nested {context}",
                "Traverse recursive {context} structure",
            ],
            "test_templates": [
                "Handle large recursive {context}",
                "Parse nested {context} hierarchy",
            ],
            "contexts": ["JSON structure", "XML tree", "directory tree"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # CONCURRENCY SCENARIOS CONFIGURATION
    # Maps race_type → (error_code, solution, lesson, entities)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "SYNC-409" means use DB constraints.
    # Error codes do NOT reveal race condition, lost update, etc.
    # ═══════════════════════════════════════════════════════════════════════════
    CONCURRENCY_SCENARIOS: ClassVar[Dict[str, Dict]] = {
        "check_then_act": {
            "error_code": "SYNC-409",  # Vague: doesn't reveal race condition
            "solution": "db_constraints",
            "working_alternatives": ["db_constraints", "optimistic_locking"],
            "lesson": "Check-then-act race condition, use DB constraints instead of Python check",
            "expected_cause": "Duplicate key violation from race condition",
            "error_message": "Operation conflict. Error: SYNC-409. Retry required.",
            "training_templates": [
                "Register {entity} if {field} unique",
                "Create {entity} after checking {field}",
            ],
            "test_templates": [
                "Add {entity} with unique {field} validation",
                "Insert {entity} if {field} not exists",
            ],
            "entities": [
                {"entity": "user", "field": "username"},
                {"entity": "account", "field": "email"},
                {"entity": "order", "field": "order_id"},
            ],
        },
        "lost_update": {
            "error_code": "SYNC-510",  # Vague: doesn't reveal lost update
            "solution": "atomic_operations",
            "working_alternatives": ["atomic_operations", "distributed_lock"],
            "lesson": "Lost update from concurrent modification, use atomic operations",
            "expected_cause": "Concurrent update overwrote changes",
            "error_message": "Data consistency error. Code: SYNC-510.",
            "training_templates": [
                "Update {entity} with check-then-{action}",
                "Modify {entity} after reading current {action}",
            ],
            "test_templates": [
                "Increment {entity} {action} safely",
                "Update {entity} {action} concurrently",
            ],
            "entities": [
                {"entity": "counter", "action": "increment"},
                {"entity": "inventory", "action": "decrement"},
                {"entity": "balance", "action": "transfer"},
            ],
        },
        "deadlock": {
            "error_code": "SYNC-611",  # Vague: doesn't reveal deadlock
            "solution": "lock_ordering",
            "working_alternatives": ["lock_ordering", "lock_timeout"],
            "lesson": "Deadlock detected, ensure consistent lock ordering across operations",
            "expected_cause": "Circular lock dependency",
            "error_message": "Resource acquisition timeout. Reference: SYNC-611.",
            "training_templates": [
                "Transfer between {entity_a} and {entity_b}",
                "Swap {entity_a} and {entity_b} atomically",
            ],
            "test_templates": [
                "Cross-update {entity_a} and {entity_b}",
                "Concurrent {entity_a}-{entity_b} transfer",
            ],
            "entities": [
                {"entity_a": "account_A", "entity_b": "account_B"},
                {"entity_a": "resource_X", "entity_b": "resource_Y"},
            ],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPORT SCENARIOS CONFIGURATION
    # Maps import_issue → (error_code, solution, lesson, modules)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "IMP-500" means use lazy imports.
    # ═══════════════════════════════════════════════════════════════════════════
    IMPORT_SCENARIOS: ClassVar[Dict[str, Dict]] = {
        "circular_import": {
            "error_code": "IMP-500",  # Vague: doesn't reveal circular import
            "solution": "lazy_imports",
            "working_alternatives": ["lazy_imports", "restructure_modules"],
            "lesson": "Circular import detected, use lazy imports or restructure modules",
            "error_message": "Import failed. Error: IMP-500. Module initialization error.",
            "training_templates": [
                "Import {module_a} that depends on {module_b}",
                "Load {module_a} with circular dependency on {module_b}",
            ],
            "test_templates": [
                "Use {module_a} which imports {module_b}",
                "Initialize {module_a} with {module_b} dependency",
            ],
            "modules": [
                {"module_a": "models", "module_b": "services"},
                {"module_a": "handlers", "module_b": "validators"},
                {"module_a": "api", "module_b": "database"},
            ],
        },
        "missing_export": {
            "error_code": "IMP-601",  # Vague: doesn't reveal missing export
            "solution": "check_exports",
            "working_alternatives": ["check_exports", "explicit_import"],
            "lesson": "Module export missing, check __all__ or use explicit import path",
            "error_message": "Symbol not found. Code: IMP-601.",
            "training_templates": [
                "Import {symbol} from {module}",
                "Use {symbol} exported by {module}",
            ],
            "test_templates": [
                "Access {symbol} from {module} package",
                "Load {symbol} via {module} interface",
            ],
            "modules": [
                {"symbol": "Config", "module": "settings"},
                {"symbol": "BaseModel", "module": "orm"},
                {"symbol": "Handler", "module": "events"},
            ],
        },
        "version_conflict": {
            "error_code": "IMP-702",  # Vague: doesn't reveal version conflict
            "solution": "pin_versions",
            "working_alternatives": ["pin_versions", "use_virtualenv"],
            "lesson": "Package version conflict, pin versions or isolate in virtualenv",
            "error_message": "Dependency resolution failed. Reference: IMP-702.",
            "training_templates": [
                "Install {pkg_a} and {pkg_b} together",
                "Add {pkg_a} to project with {pkg_b}",
            ],
            "test_templates": [
                "Use {pkg_a} alongside {pkg_b}",
                "Import both {pkg_a} and {pkg_b}",
            ],
            "modules": [
                {"pkg_a": "numpy==1.21", "pkg_b": "tensorflow==2.5"},
                {"pkg_a": "requests==2.25", "pkg_b": "urllib3==2.0"},
            ],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ERROR CODE PATTERNS - For parsing error responses
    # ═══════════════════════════════════════════════════════════════════════════
    ERROR_CODE_PATTERNS: ClassVar[Dict[str, str]] = {
        "ZOMBIE-DEP-404": "ZOMBIE-DEP-404",
        "DEPRECATED-PKG": "DEPRECATED-PKG",
        "VARIANT-404": "VARIANT-404",
        "IMPORT-CIRC-500": "IMPORT-CIRC-500",
        "IMPORT-ERROR": "IMPORT-ERROR",
        "SEGFAULT-000": "SEGFAULT-000",
        "BUS-ERROR": "BUS-ERROR",
        "RACE-COND-409": "RACE-COND-409",
        "LOST-UPDATE": "LOST-UPDATE",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ERROR → RECOVERY OPTIONS MAPPING
    # Maps error codes (VAGUE codes!) to available recovery solutions
    # Uses prefix matching: PKG-* → package managers, EXE-* → execution modes, etc.
    # ═══════════════════════════════════════════════════════════════════════════
    ERROR_RECOVERY_OPTIONS: ClassVar[Dict[str, List[str]]] = {
        # Package errors (PKG-XXX) → try different package managers
        "PKG-404": ["conda", "poetry", "pipenv"],
        "PKG-505": ["poetry", "conda", "pipenv"],
        "PKG-606": ["conda", "source_build"],
        "PKG-707": ["pipenv", "virtualenv"],
        "PKG-808": ["conda"],
        # Execution errors (EXE-XXX) → try different execution modes
        "EXE-101": ["pure_python_fallback", "enable_faulthandler", "reduce_memory"],
        "EXE-202": ["streaming_mode", "reduce_batch_size", "memory_aligned_alloc"],
        "EXE-303": ["increase_recursion_limit", "iterative_rewrite"],
        # Concurrency errors (SYNC-XXX) → try different sync strategies
        "SYNC-409": ["db_constraints", "optimistic_locking", "serializable_txn"],
        "SYNC-510": ["atomic_operations", "distributed_lock", "compare_and_swap"],
        "SYNC-611": ["lock_ordering", "lock_timeout"],
        # Import errors (IMP-XXX) → try different import strategies
        "IMP-500": ["lazy_imports", "restructure_modules", "pipenv_isolation"],
        "IMP-601": ["check_exports", "explicit_import"],
        "IMP-702": ["pin_versions", "use_virtualenv"],
    }

    # Default recovery options by error code prefix
    ERROR_PREFIX_DEFAULTS: ClassVar[Dict[str, List[str]]] = {
        "PKG-": ["conda", "poetry", "pipenv"],
        "EXE-": ["pure_python_fallback", "streaming_mode", "reduce_memory"],
        "SYNC-": ["db_constraints", "atomic_operations", "distributed_lock"],
        "IMP-": ["lazy_imports", "check_exports", "restructure_modules"],
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ACTION TYPES - All supported actions in the coding domain
    # ═══════════════════════════════════════════════════════════════════════════
    ACTION_TYPES: ClassVar[Dict[str, Dict]] = {
        "install_package": {
            "keywords": ["install", "set up", "add", "dependency", "requirements"],
            "mcp_tool": "install_package",
            "description": "Install a Python package",
        },
        "import_module": {
            "keywords": ["import legacy", "import old", "load legacy", "load old"],
            "mcp_tool": "import_module",
            "description": "Import a Python module",
        },
        "run_code": {
            "keywords": ["run", "execute", "process", "binary", "wrapper"],
            "mcp_tool": "run_code",
            "description": "Execute Python code",
        },
        "check_unique": {
            "keywords": ["register", "create", "unique", "if not exists"],
            "mcp_tool": "check_unique",
            "description": "Check uniqueness before insert",
        },
        "update_counter": {
            "keywords": ["update", "increment", "decrement", "counter", "modify"],
            "mcp_tool": "update_counter",
            "description": "Update a counter or value",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK TEMPLATES - For dependency scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    DEPENDENCY_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Install {package} library for {description}",
        "Set up {package} for {description}",
        "Add {package} dependency for {description}",
        "Configure {package} for {description} project",
    ]

    DEPENDENCY_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Add {package} to production requirements.txt",
        "Configure {package} in the new microservice",
        "Set up {package} for the ML pipeline",
        "Install {package} for the data migration script",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "deleted": "{package} deleted from PyPI. Use {manager} to install from alternative channel.",
        "deprecated": "{package} is deprecated. Use {manager} which has cached version.",
        "variant": "{package} variant only in conda channels. Use conda install.",
        "circular": "{package} has circular imports. Use {manager} for isolated environment.",
        "cuda": "{package} requires CUDA. Use {manager} for bundled CUDA toolkit.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID MANAGERS - Strict enforcement for fair comparison
    # ═══════════════════════════════════════════════════════════════════════════
    # For multi-condition scenarios, ONLY these 2 managers are valid.
    # Each condition_key is deterministically mapped to exactly ONE manager.
    # This matches logistics domain difficulty where only antwerp/hamburg work.
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_MANAGERS: ClassVar[List[str]] = [
        "conda",
        "poetry",
    ]

    @classmethod
    def get_valid_manager_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid manager for a given condition_key.

        Uses deterministic hash to ensure:
        - Same condition_key always maps to same manager
        - Each manager has ~50% probability
        - Only ONE manager works per condition_key

        This matches logistics domain difficulty.
        """
        # Deterministic hash-based selection
        # FIX: Use hashlib.md5 for deterministic hashing across sessions
        import hashlib

        hash_bytes = hashlib.md5(condition_key.encode()).digest()
        hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
        idx = hash_val % len(cls.MULTI_CONDITION_VALID_MANAGERS)
        return cls.MULTI_CONDITION_VALID_MANAGERS[idx]


# Convenience function to get config
def get_coding_config() -> type:
    """Get the coding configuration class."""
    return CodingConfig
