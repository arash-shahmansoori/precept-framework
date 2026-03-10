"""
Coding Domain Strategy for PRECEPT.

Handles dependency/import black swan scenarios with optional Docker-based
code execution and dynamic learning from real execution feedback.

Black Swan Types (from black_swan_gen.py):
- Dependency_Zombie: Agent tries to install a deleted/missing library
- Opaque_Crash: Agent triggers a Segfault/Bus Error with no stack trace
- Concurrency_Race: Race condition in check-then-act logic
- Import_Hell: Circular imports or missing submodules

Features:
- Docker-based sandboxed code execution (optional)
- Dynamic learning from execution errors, warnings, and logs
- Auto-categorization of unknown errors using LLM
- Persistent configuration that improves over time

🚨 NO HARDCODED KNOWLEDGE - all solutions learned through experience.
"""

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from ..rule_parser import DynamicRuleParser
from .base import (
    ActionResult,
    BaselineDomainStrategy,
    BlackSwanCategory,
    DomainStrategy,
    ParsedTask,
)

# AutoGen imports (optional)
try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = None

# Code execution imports (optional)
try:
    from ..code_executor import CodeExecutionManager, ExecutionResult
    from ..dynamic_coding_config import DynamicCodingConfig
    from ..execution_feedback_processor import (
        ExecutionFeedbackProcessor,
        ProcessedFeedback,
    )
    from ..multiturn_docker_agent import MultiTurnDockerAgent
    from ..single_turn_docker_agent import SingleTurnDockerAgent

    CODE_EXECUTION_AVAILABLE = True
    DOCKER_AGENT_AVAILABLE = True
    MULTITURN_AGENT_AVAILABLE = True
except ImportError:
    CODE_EXECUTION_AVAILABLE = False
    DOCKER_AGENT_AVAILABLE = False
    MULTITURN_AGENT_AVAILABLE = False
    CodeExecutionManager = None
    ExecutionResult = None
    ExecutionFeedbackProcessor = None
    ProcessedFeedback = None
    DynamicCodingConfig = None
    SingleTurnDockerAgent = None
    MultiTurnDockerAgent = None


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CONFIGURATION - Single Source of Truth
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CodingDomainConfig:
    """
    Centralized configuration for the coding domain.

    This class serves as the single source of truth for:
    - Package managers available in the domain
    - Known packages (vocabulary for task parsing)
    - Error codes and their detection patterns
    - Action types and their keywords
    - Recovery solutions for each error type

    Note: This defines VOCABULARY only, not behavior.
    Which solution works for which error is LEARNED, not hardcoded.
    """

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
            "description": "Import a Python module (legacy modules)",
        },
        "run_code": {
            "keywords": [
                "run",
                "execute",
                "process",
                "binary",
                "wrapper",
                "extension",
                "compiled",
            ],
            "mcp_tool": "run_code",
            "description": "Execute Python code or binary",
        },
        "check_unique": {
            "keywords": ["register", "create", "unique", "if not exists", "insert if"],
            "mcp_tool": "check_unique",
            "description": "Check uniqueness before insert",
        },
        "update_counter": {
            "keywords": [
                "update",
                "increment",
                "decrement",
                "counter",
                "modify",
                "inventory",
                "balance",
            ],
            "mcp_tool": "update_counter",
            "description": "Update a counter or value",
        },
        "refactor_imports": {
            "keywords": [
                "refactor",
                "reorganize",
                "restructure",
                "circular",
                "fix",
                "resolve",
                "export",
                "visibility",
            ],
            "mcp_tool": "refactor_imports",
            "description": "Refactor module imports or fix export issues",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PACKAGE MANAGERS - For dependency installation
    # ═══════════════════════════════════════════════════════════════════════════
    PACKAGE_MANAGERS: ClassVar[List[str]] = ["pip", "conda", "poetry", "pipenv"]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN PACKAGES - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_PACKAGES: ClassVar[List[str]] = [
        # Blocked packages (which manager works is LEARNED!)
        "fast_xml",
        "auth_lib_v1",
        "numpy_mkl",
        "legacy_orm",
        # Normal packages
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
    # ERROR CODES - All error codes and their patterns
    # ═══════════════════════════════════════════════════════════════════════════
    ERROR_CODE_PATTERNS: ClassVar[Dict[str, str]] = {
        # Dependency errors
        "ZOMBIE-DEP-404": "ZOMBIE-DEP-404",
        "DEPRECATED-PKG": "DEPRECATED-PKG",
        "VARIANT-404": "VARIANT-404",
        # Import errors
        "IMPORT-CIRC-500": "IMPORT-CIRC-500",
        "IMPORT-ERROR": "IMPORT-ERROR",
        # Crash errors
        "SEGFAULT-000": "SEGFAULT-000",
        "BUS-ERROR": "BUS-ERROR",
        # Concurrency errors
        "RACE-COND-409": "RACE-COND-409",
        "LOST-UPDATE": "LOST-UPDATE",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ERROR → RECOVERY SOLUTIONS MAPPING
    # Maps error codes to available recovery solutions (which one works is LEARNED)
    # ═══════════════════════════════════════════════════════════════════════════
    ERROR_RECOVERY_OPTIONS: ClassVar[Dict[str, List[str]]] = {
        # Dependency errors → try different managers
        "ZOMBIE-DEP-404": ["conda", "poetry", "pipenv"],
        "DEPRECATED-PKG": ["poetry", "conda", "pipenv"],
        "VARIANT-404": ["conda", "poetry", "pipenv"],
        # Import errors → try different strategies
        "IMPORT-CIRC-500": ["lazy_imports", "restructure_modules", "pipenv_isolation"],
        "IMPORT-ERROR": ["check_exports", "explicit_import", "reload_module"],
        # Crash errors → try different execution modes
        "SEGFAULT-000": [
            "pure_python_fallback",
            "enable_faulthandler",
            "reduce_memory",
        ],
        "BUS-ERROR": ["streaming_mode", "reduce_batch_size", "memory_aligned_alloc"],
        # Concurrency errors → try different sync strategies
        "RACE-COND-409": ["db_constraints", "optimistic_locking", "serializable_txn"],
        "LOST-UPDATE": ["atomic_operations", "distributed_lock", "compare_and_swap"],
    }

    # Default error code when no pattern matches
    DEFAULT_ERROR_CODE: ClassVar[str] = "UNKNOWN-ERROR"

    # Generic error code regex pattern
    GENERIC_ERROR_PATTERN: ClassVar[str] = r"Error code: (\S+)"

    @classmethod
    def extract_error_code(cls, response: str) -> str:
        """
        Extract error code from MCP response string.

        Uses pattern matching against known error codes,
        then falls back to generic regex extraction.

        Args:
            response: The response string from MCP server

        Returns:
            Extracted error code or default if none found
        """
        if not response:
            return cls.DEFAULT_ERROR_CODE

        # Check known error patterns first
        for error_code, pattern in cls.ERROR_CODE_PATTERNS.items():
            if pattern in response:
                return error_code

        # Try generic pattern extraction
        match = re.search(cls.GENERIC_ERROR_PATTERN, response)
        if match:
            return match.group(1)

        return cls.DEFAULT_ERROR_CODE

    @classmethod
    def get_action_type(cls, task: str) -> str:
        """
        Detect action type from task string.

        Args:
            task: The task description

        Returns:
            Action type string (e.g., "install_package", "run_code")
        """
        task_lower = task.lower()

        # Score each action type by keyword matches
        scores = {}
        for action, info in cls.ACTION_TYPES.items():
            score = sum(1 for kw in info["keywords"] if kw in task_lower)
            if score > 0:
                scores[action] = score

        # Return highest scoring action, or default to run_code
        if scores:
            return max(scores, key=scores.get)
        return "run_code"

    @classmethod
    def get_recovery_options(cls, error_code: str) -> List[str]:
        """
        Get available recovery options for an error code.

        Uses prefix matching for VAGUE error codes:
        - PKG-XXX → package managers
        - EXE-XXX → execution modes
        - SYNC-XXX → sync strategies
        - IMP-XXX → import strategies

        Args:
            error_code: The error code (e.g., PKG-404, EXE-101)

        Returns:
            List of recovery options to try
        """
        # First try exact match
        if error_code in cls.ERROR_RECOVERY_OPTIONS:
            return cls.ERROR_RECOVERY_OPTIONS[error_code]

        # Then try prefix match for vague codes
        PREFIX_DEFAULTS = {
            "PKG-": ["conda", "poetry", "pipenv"],
            "EXE-": ["pure_python_fallback", "streaming_mode", "reduce_memory"],
            "SYNC-": ["db_constraints", "atomic_operations", "distributed_lock"],
            "IMP-": ["lazy_imports", "check_exports", "restructure_modules"],
        }
        for prefix, options in PREFIX_DEFAULTS.items():
            if error_code.startswith(prefix):
                return options

        # Default: return empty (unknown error type)
        return []


class CodingDomainStrategy(DomainStrategy):
    """
    Coding domain strategy for ALL coding black swan scenarios.

    Black Swan Types (from black_swan_gen.py):
    - Dependency_Zombie: Agent tries to install a deleted/missing library
    - Opaque_Crash: Agent triggers a Segfault/Bus Error with no stack trace
    - Concurrency_Race: Race condition in check-then-act logic
    - Import_Hell: Circular imports or missing submodules

    Features:
    - Docker-based sandboxed code execution (when enabled)
    - Dynamic learning from real execution errors, warnings, and logs
    - Auto-categorization of unknown errors using LLM
    - Persistent configuration that improves over time

    🚨 NO HARDCODED KNOWLEDGE - all solutions learned through experience.

    Configuration is centralized in CodingDomainConfig for maintainability.

    Usage:
        # Standard mode (simulated execution via MCP)
        strategy = CodingDomainStrategy()

        # Docker mode (real sandboxed execution)
        strategy = CodingDomainStrategy(enable_docker_execution=True)
    """

    # Reference shared configuration (Single Source of Truth)
    PACKAGE_MANAGERS = CodingDomainConfig.PACKAGE_MANAGERS
    KNOWN_PACKAGES = CodingDomainConfig.KNOWN_PACKAGES

    # ═══════════════════════════════════════════════════════════════════════════
    # OPAQUE OPTIONS - Prevent LLM from inferring solutions from option names
    # ═══════════════════════════════════════════════════════════════════════════
    # Maps neutral identifiers to real package managers
    # This prevents LLM from reasoning "conda for scientific packages"
    # MUST match the 8 managers in CodingConfig.PACKAGE_MANAGERS for ~12.5% random success
    # ═══════════════════════════════════════════════════════════════════════════
    MANAGER_OPTIONS_MAP: ClassVar[Dict[str, str]] = {
        "manager_a": "pip",
        "manager_b": "conda",  # Valid for multi-condition
        "manager_c": "poetry",  # Valid for multi-condition
        "manager_d": "pipenv",
        "manager_e": "uv",  # Modern fast package manager
        "manager_f": "mamba",  # Conda alternative
        "manager_g": "pdm",  # PEP 582 package manager
        "manager_h": "hatch",  # Modern Python project manager
    }
    MANAGER_OPTIONS_REVERSE: ClassVar[Dict[str, str]] = {
        v: k for k, v in MANAGER_OPTIONS_MAP.items()
    }

    def __init__(
        self,
        enable_docker_execution: bool = False,
        enable_multiturn: bool = False,
        enable_vector_store: bool = True,
        llm_client: Optional[Any] = None,
        data_dir: Optional[Any] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize the coding domain strategy.

        Args:
            enable_docker_execution: Whether to use Docker for real code execution.
                                    When False, uses simulated execution via MCP.
            enable_multiturn: Whether to use MultiTurnDockerAgent for conversation support.
                             When False, uses SingleTurnDockerAgent.
            enable_vector_store: Whether to enable ChromaDB vector store for semantic search.
            llm_client: Optional LLM client for auto-categorizing unknown errors.
            data_dir: Optional data directory for Docker agent persistence.
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                        - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                        - 4 = lenient (1 initial + 4 retries = 5 attempts)
        """
        super().__init__(max_retries=max_retries)

        # Dynamic rule parser - only knows vocabulary
        self.rule_parser = DynamicRuleParser(
            known_entities=self.PACKAGE_MANAGERS + self.KNOWN_PACKAGES
        )

        # ═══════════════════════════════════════════════════════════════════════
        # DOCKER CODE EXECUTION (Optional)
        # When enabled, uses SingleTurnDockerAgent or MultiTurnDockerAgent
        # ═══════════════════════════════════════════════════════════════════════
        self.enable_docker_execution = enable_docker_execution
        self.enable_multiturn = enable_multiturn
        self._enable_vector_store = enable_vector_store
        self._llm_client = llm_client
        self._data_dir = data_dir

        # ═══════════════════════════════════════════════════════════════════════
        # COMPOSITION: Use Docker Agents as the execution engine
        # SingleTurnDockerAgent: Docker execution + local learning + persistence
        # MultiTurnDockerAgent: Same + conversation history + session management
        # ═══════════════════════════════════════════════════════════════════════
        self._docker_agent: Optional[Any] = None
        self._multiturn_agent: Optional[Any] = None
        self._code_executor: Optional[Any] = None
        self._feedback_processor: Optional[Any] = None
        self._dynamic_config: Optional[Any] = None

        if enable_docker_execution:
            from pathlib import Path

            agent_data_dir = Path(data_dir) if data_dir else None

            # ─── MultiTurn Docker Agent (for conversation support) ───
            if enable_multiturn and MULTITURN_AGENT_AVAILABLE:
                self._multiturn_agent = MultiTurnDockerAgent(
                    enable_docker=True,
                    data_dir=agent_data_dir,
                    timeout=60,
                    auto_save=True,
                    enable_vector_store=enable_vector_store,
                )
                # Reference the agent's components for compatibility
                self._code_executor = self._multiturn_agent.executor
                self._feedback_processor = self._multiturn_agent.processor
                self._dynamic_config = self._multiturn_agent.config

            # ─── SingleTurn Docker Agent (default) ───
            elif DOCKER_AGENT_AVAILABLE:
                self._docker_agent = SingleTurnDockerAgent(
                    enable_docker=True,
                    data_dir=agent_data_dir,
                    timeout=60,
                    auto_save=True,
                    enable_vector_store=enable_vector_store,
                )
                # Reference the agent's components for compatibility
                self._code_executor = self._docker_agent.executor
                self._feedback_processor = self._docker_agent.processor
                self._dynamic_config = self._docker_agent.config

            # ─── Fallback: Direct component usage ───
            elif CODE_EXECUTION_AVAILABLE:
                self._code_executor = CodeExecutionManager(enable_docker=True)
                self._feedback_processor = ExecutionFeedbackProcessor(
                    llm_client=llm_client,
                    enable_llm_categorization=llm_client is not None,
                )
                self._dynamic_config = DynamicCodingConfig()
                self._dynamic_config.load_from_json()

        # ═══════════════════════════════════════════════════════════════════════
        # LEARNED MAPPINGS - What PRECEPT learns through experience
        # These are loaded from persistent storage on first MCP connection
        # ═══════════════════════════════════════════════════════════════════════

        # Package → Working manager (for dependency errors)
        self._learned_package_managers: Dict[str, str] = {}

        # Error code → Working solution (for all error types)
        # Key: "error_code:context" → Value: "working_solution"
        self._learned_error_solutions: Dict[str, str] = {}

        # Context pattern → Working solution (for pattern matching)
        # Key: context pattern → Value: "working_solution"
        self._learned_context_solutions: Dict[str, str] = {}

        # Flag to track if we've loaded from persistent storage
        self._mappings_loaded = False

        # Flag to track if we've synced with MCP
        self._mcp_synced = False

    @property
    def is_docker_available(self) -> bool:
        """Check if Docker execution is available."""
        if self._multiturn_agent:
            # MultiTurnDockerAgent.is_docker_available is a property
            return self._multiturn_agent.is_docker_available
        if self._docker_agent:
            # SingleTurnDockerAgent.is_docker_available is a property
            return self._docker_agent.is_docker_available
        if self._code_executor and hasattr(self._code_executor, "is_docker_available"):
            # CodeExecutionManager.is_docker_available() is a method
            return self._code_executor.is_docker_available()
        return False

    @property
    def dynamic_config(self) -> Optional[Any]:
        """Get the dynamic configuration (if available)."""
        return self._dynamic_config

    @property
    def docker_agent(self) -> Optional[Any]:
        """Get the internal Docker agent (if available)."""
        return self._docker_agent

    @property
    def multiturn_agent(self) -> Optional[Any]:
        """Get the internal MultiTurn Docker agent (if available)."""
        return self._multiturn_agent

    @property
    def active_agent(self) -> Optional[Any]:
        """Get the currently active agent (multi-turn or single-turn)."""
        return self._multiturn_agent or self._docker_agent

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics from code executor and dynamic config.

        Returns:
            Dictionary with execution statistics
        """
        stats = {
            "docker_execution_enabled": self.enable_docker_execution,
            "docker_available": self.is_docker_available,
            "code_execution_available": CODE_EXECUTION_AVAILABLE,
        }

        if self._code_executor:
            stats["executor_stats"] = self._code_executor.get_stats()

        if self._feedback_processor:
            stats["feedback_stats"] = self._feedback_processor.get_stats()

        if self._dynamic_config:
            stats["config_stats"] = self._dynamic_config.get_stats()

        # Add Docker agent stats if available
        if self._docker_agent:
            stats["docker_agent_stats"] = self._docker_agent.get_stats()

        # Add MultiTurn Docker agent stats if available
        if self._multiturn_agent:
            stats["multiturn_enabled"] = True
            # MultiTurnDockerAgent uses get_session_summary and get_persisted_stats
            if hasattr(self._multiturn_agent, "get_session_summary"):
                stats["multiturn_agent_stats"] = (
                    self._multiturn_agent.get_session_summary()
                )
            if hasattr(self._multiturn_agent, "get_persisted_stats"):
                stats["multiturn_persisted_stats"] = (
                    self._multiturn_agent.get_persisted_stats()
                )

        return stats

    # ═══════════════════════════════════════════════════════════════════════════
    # BIDIRECTIONAL SYNC: Docker Agent ↔ MCP Server
    # This ensures learned patterns are shared between local and server storage
    # ═══════════════════════════════════════════════════════════════════════════

    async def sync_with_mcp(self, mcp_client: Any) -> Dict[str, int]:
        """
        Bidirectional sync of learned patterns between Docker agent and MCP server.

        This is the key integration point:
        1. Push: Send Docker agent's learned patterns to MCP server
        2. Pull: Get MCP server's learned patterns to Docker agent

        Args:
            mcp_client: MCP client for server communication

        Returns:
            Dict with counts of synced items
        """
        sync_stats = {"pushed": 0, "pulled": 0}

        if self._mcp_synced:
            return sync_stats

        try:
            # ═══════════════════════════════════════════════════════════════════
            # PULL: MCP Server → Docker Agent (load server knowledge)
            # ═══════════════════════════════════════════════════════════════════
            pull_count = await self._pull_from_mcp(mcp_client)
            sync_stats["pulled"] = pull_count

            # ═══════════════════════════════════════════════════════════════════
            # PUSH: Docker Agent → MCP Server (share local knowledge)
            # ═══════════════════════════════════════════════════════════════════
            push_count = await self._push_to_mcp(mcp_client)
            sync_stats["pushed"] = push_count

            self._mcp_synced = True

        except Exception as e:
            # Sync is best-effort, don't fail the operation
            print(f"    ⚠️ MCP sync warning: {e}")

        return sync_stats

    async def _pull_from_mcp(self, mcp_client: Any) -> int:
        """
        Pull learned patterns from MCP server to Docker agent's config.

        Args:
            mcp_client: MCP client for server communication

        Returns:
            Count of patterns pulled
        """
        count = 0

        if not self._dynamic_config:
            return count

        try:
            # Get all domain mappings from MCP
            response = await mcp_client.call_tool(
                "get_domain_mappings", {"domain": "coding"}
            )

            if response and "No learned mappings" not in str(response):
                # The MCP returns formatted text, parse it
                response_str = str(response)

                # Parse error_patterns section
                if "error_patterns:" in response_str:
                    # Extract patterns and add to dynamic config
                    lines = response_str.split("\n")
                    in_error_patterns = False
                    for line in lines:
                        if "error_patterns:" in line:
                            in_error_patterns = True
                            continue
                        if in_error_patterns and "→" in line:
                            parts = line.split("→")
                            if len(parts) == 2:
                                pattern = parts[0].strip().lstrip("-").strip()
                                error_code = parts[1].strip()
                                if self._dynamic_config.add_error_pattern(
                                    pattern, error_code
                                ):
                                    count += 1
                        elif (
                            in_error_patterns
                            and line.strip()
                            and not line.startswith(" ")
                        ):
                            in_error_patterns = False

                # Parse recovery_solutions section
                if "recovery_solutions:" in response_str:
                    lines = response_str.split("\n")
                    in_recovery = False
                    for line in lines:
                        if "recovery_solutions:" in line:
                            in_recovery = True
                            continue
                        if in_recovery and "→" in line:
                            parts = line.split("→")
                            if len(parts) == 2:
                                error_code = parts[0].strip().lstrip("-").strip()
                                solution = parts[1].strip()
                                if self._dynamic_config.add_recovery_solution(
                                    error_code, solution
                                ):
                                    count += 1
                        elif in_recovery and line.strip() and not line.startswith(" "):
                            in_recovery = False

            # Also populate local caches for package managers
            for pkg in self.KNOWN_PACKAGES:
                result = await mcp_client.call_tool(
                    "get_domain_mapping",
                    {
                        "domain": "coding",
                        "mapping_type": "package_managers",
                        "key": pkg,
                    },
                )
                if result and "NOT_FOUND" not in str(result):
                    self._learned_package_managers[pkg] = str(result).strip()
                    count += 1

        except Exception:
            pass  # Best-effort

        return count

    async def _push_to_mcp(self, mcp_client: Any) -> int:
        """
        Push Docker agent's learned patterns to MCP server.

        Args:
            mcp_client: MCP client for server communication

        Returns:
            Count of patterns pushed
        """
        count = 0

        if not self._dynamic_config:
            return count

        try:
            # Get all learned patterns from dynamic config
            config_stats = self._dynamic_config.get_stats()

            # Push error patterns
            if hasattr(self._dynamic_config, "_learned_error_patterns"):
                for (
                    pattern,
                    error_code,
                ) in self._dynamic_config._learned_error_patterns.items():
                    await mcp_client.call_tool(
                        "store_domain_mapping",
                        {
                            "domain": "coding",
                            "mapping_type": "error_patterns",
                            "key": pattern[:100],  # Truncate long patterns
                            "value": error_code,
                        },
                    )
                    count += 1

            # Push recovery solutions
            if hasattr(self._dynamic_config, "_learned_recovery_solutions"):
                for (
                    error_code,
                    solution,
                ) in self._dynamic_config._learned_recovery_solutions.items():
                    await mcp_client.call_tool(
                        "store_domain_mapping",
                        {
                            "domain": "coding",
                            "mapping_type": "recovery_solutions",
                            "key": error_code,
                            "value": solution,
                        },
                    )
                    count += 1

            # Push local package manager mappings
            for pkg, manager in self._learned_package_managers.items():
                await mcp_client.call_tool(
                    "store_domain_mapping",
                    {
                        "domain": "coding",
                        "mapping_type": "package_managers",
                        "key": pkg,
                        "value": manager,
                    },
                )
                count += 1

        except Exception:
            pass  # Best-effort

        return count

    async def sync_docker_agent_learning(self, mcp_client: Any) -> None:
        """
        Sync Docker agent's learned patterns after execution.

        Call this after code execution to ensure MCP server has latest patterns.

        Args:
            mcp_client: MCP client for server communication
        """
        if not self._docker_agent or not self._dynamic_config:
            return

        try:
            # Get newly learned rules from Docker agent
            learned_rules = self._docker_agent.export_learned_rules()

            for rule in learned_rules:
                # Store as experience in MCP
                await mcp_client.store_experience(
                    task=f"Learned rule: {rule[:100]}",
                    outcome="success",
                    strategy="docker_learning",
                    lessons=rule,
                    domain="coding",
                )
        except Exception:
            pass  # Best-effort

    async def load_persisted_mappings(self, mcp_client: Any) -> None:
        """
        Load previously learned mappings from MCP server.

        This enables learning to persist across sessions!
        Called automatically on first task execution.
        """
        if self._mappings_loaded:
            return

        try:
            # Get all domain mappings from MCP
            response = await mcp_client.call_tool(
                "get_domain_mappings", {"domain": "coding"}
            )

            if (
                response
                and "NOT_FOUND" not in str(response)
                and "No learned mappings" not in str(response)
            ):
                # Parse the response and populate local caches
                # The MCP returns formatted text, so we need to get individual mappings
                for mapping_type in [
                    "package_managers",
                    "error_solutions",
                    "context_solutions",
                ]:
                    # Get each mapping type's data
                    for key in self.KNOWN_PACKAGES:
                        result = await mcp_client.call_tool(
                            "get_domain_mapping",
                            {
                                "domain": "coding",
                                "mapping_type": "package_managers",
                                "key": key,
                            },
                        )
                        if result and "NOT_FOUND" not in str(result):
                            self._learned_package_managers[key] = str(result).strip()

            self._mappings_loaded = True
        except Exception:
            # Silently handle errors - we'll learn fresh if loading fails
            self._mappings_loaded = True

    async def persist_mapping(
        self,
        mcp_client: Any,
        mapping_type: str,
        key: str,
        value: str,
    ) -> None:
        """
        Persist a learned mapping to the MCP server.

        Args:
            mcp_client: MCP client for server communication
            mapping_type: One of "package_managers", "error_solutions", "context_solutions"
            key: The key (e.g., package name, error_code:entity)
            value: The learned solution
        """
        try:
            await mcp_client.call_tool(
                "store_domain_mapping",
                {
                    "domain": "coding",
                    "mapping_type": mapping_type,
                    "key": key,
                    "value": value,
                },
            )
        except Exception:
            # Silently handle errors - runtime learning still works
            pass

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.CODING

    @property
    def domain_name(self) -> str:
        return "coding"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        """
        Generate system prompt with learned rules.

        Includes rules from:
        1. Passed learned_rules parameter
        2. Dynamic config (if Docker execution enabled)

        Args:
            learned_rules: Optional list of learned rules

        Returns:
            System prompt string
        """
        # Combine passed rules with dynamic config rules
        all_rules = list(learned_rules) if learned_rules else []

        # Add rules from dynamic config (if available)
        if self._dynamic_config:
            config_rules = self._dynamic_config.export_learned_rules()
            for rule in config_rules:
                if rule not in all_rules:
                    all_rules.append(rule)

        # Build execution mode info
        execution_mode = "simulated (MCP)"
        if self.enable_docker_execution:
            execution_mode = (
                "Docker sandbox" if self.is_docker_available else "subprocess fallback"
            )

        base = f"""You are a coding assistant with PRECEPT learning capabilities.

EXECUTION MODE: {execution_mode}

AVAILABLE ACTIONS:
{chr(10).join(f"- {name}: {info['description']}" for name, info in CodingDomainConfig.ACTION_TYPES.items())}

PACKAGE MANAGERS: {", ".join(self.PACKAGE_MANAGERS)}

PRECEPT ADVANTAGES:
- Learns from ALL error types (dependency, crash, concurrency, import)
- Remembers working solutions for each error pattern
- Applies learned knowledge immediately on repeat encounters
- Dynamically updates configuration from real execution feedback"""

        if all_rules:
            rules_section = "\n\n═══ LEARNED RULES ═══\n"
            for i, rule in enumerate(all_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base = rules_section + base

        return base

    def get_available_actions(self) -> List[str]:
        return list(CodingDomainConfig.ACTION_TYPES.keys())

    def get_available_entities(self) -> List[str]:
        return self.PACKAGE_MANAGERS + self.KNOWN_PACKAGES

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return OPAQUE options SHUFFLED for fair exploration.

        Shuffled so both PRECEPT and baselines have the same random chance.
        Uses neutral identifiers that reveal nothing about the solution.
        """
        import random

        options = list(self.MANAGER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str) -> str:
        """Convert opaque option back to real package manager."""
        return self.MANAGER_OPTIONS_MAP.get(opaque_option, opaque_option)

    def parse_task(self, task: str) -> ParsedTask:
        """
        Parse coding task and extract action type, entity, and parameters.

        Supports all action types defined in CodingDomainConfig.ACTION_TYPES.
        """
        import re

        task_lower = task.lower()

        # Detect action type using config
        action = CodingDomainConfig.get_action_type(task)

        # Detect package/module for dependency tasks
        entity = None
        for pkg in self.KNOWN_PACKAGES:
            if pkg.lower() in task_lower or pkg in task:
                entity = pkg
                break

        # Detect package manager (for install_package action)
        manager = "pip"  # Default
        for mgr in self.PACKAGE_MANAGERS:
            if mgr in task_lower:
                manager = mgr
                break

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            # Generate deterministic key (sorted, joined with +)
            condition_key = "+".join(sorted(conditions))

        # Build parameters based on action type
        parameters = {
            "action_type": action,
            "manager": manager,
            "package": entity or "unknown",
            "raw_task": task,
            "condition_key": condition_key,  # Multi-condition key for rule storage
            "conditions": conditions,  # Individual conditions
        }

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity or "unknown",
            source=manager,
            parameters=parameters,
        )

    def apply_learned_rules(
        self,
        parsed_task: ParsedTask,
        rules: List[str],
    ) -> Tuple[ParsedTask, bool, str]:
        """
        Apply learned rules - THE KEY PRECEPT ADVANTAGE!

        Handles ALL scenario types:
        1. Package → Manager mapping (for dependency errors)
        2. Error → Solution mapping (for all error types)
        3. Context → Solution mapping (for pattern matching)
        """
        action = parsed_task.action
        entity = parsed_task.entity
        task_lower = parsed_task.raw_task.lower()

        # ═══════════════════════════════════════════════════════════════════════
        # 1. Check package → manager mapping (for install_package)
        # ═══════════════════════════════════════════════════════════════════════
        if action == "install_package" and entity in self._learned_package_managers:
            working_manager = self._learned_package_managers[entity]
            parsed_task.source = working_manager
            parsed_task.parameters["manager"] = working_manager
            return parsed_task, True, f"Learned:{entity}→{working_manager}"

        # ═══════════════════════════════════════════════════════════════════════
        # 2. Check context → solution mapping (for other scenarios)
        # ═══════════════════════════════════════════════════════════════════════
        for context_pattern, solution in self._learned_context_solutions.items():
            if context_pattern in task_lower:
                parsed_task.parameters["preferred_solution"] = solution
                return parsed_task, True, f"Learned:{context_pattern}→{solution}"

        # ═══════════════════════════════════════════════════════════════════════
        # 3. Check rules from server
        # ═══════════════════════════════════════════════════════════════════════
        if not rules:
            return parsed_task, False, "Exploration"

        for rule in rules:
            rule_lower = rule.lower()

            # Check for package → manager rules
            if entity and entity.lower() in rule_lower:
                for mgr in self.PACKAGE_MANAGERS:
                    if mgr in rule_lower:
                        parsed_task.source = mgr
                        parsed_task.parameters["manager"] = mgr
                        self._learned_package_managers[entity] = mgr
                        return parsed_task, True, f"Rule:{entity}→{mgr}"

            # Check for solution hints in rules
            for (
                error_code,
                solutions,
            ) in CodingDomainConfig.ERROR_RECOVERY_OPTIONS.items():
                if error_code.lower() in rule_lower:
                    for solution in solutions:
                        if solution in rule_lower:
                            parsed_task.parameters["preferred_solution"] = solution
                            return parsed_task, True, f"Rule:{error_code}→{solution}"

        return parsed_task, False, "Exploration"

    async def execute_action(
        self,
        mcp_client: Any,
        parsed_task: ParsedTask,
    ) -> ActionResult:
        """
        Execute coding action via MCP server or Docker (if enabled).

        Routes to appropriate execution method based on:
        1. Action type (install_package, run_code, etc.)
        2. Docker execution flag (real vs simulated)

        When Docker execution is enabled for run_code actions:
        - Executes real Python code in Docker container
        - Captures stdout, stderr, exit code
        - Processes feedback for dynamic learning
        - Updates configuration based on execution results

        On first call, loads any previously learned mappings from persistent storage.
        """
        # 💾 LOAD: Get previously learned mappings from persistent storage
        if not self._mappings_loaded:
            await self.load_persisted_mappings(mcp_client)

        # 🔄 SYNC: Bidirectional sync with MCP on first action
        if not self._mcp_synced and (self._multiturn_agent or self._docker_agent):
            sync_stats = await self.sync_with_mcp(mcp_client)
            if sync_stats["pushed"] > 0 or sync_stats["pulled"] > 0:
                print(
                    f"    🔄 MCP Sync: pushed={sync_stats['pushed']}, pulled={sync_stats['pulled']}"
                )

        action = parsed_task.action
        entity = parsed_task.entity or "unknown"
        params = parsed_task.parameters

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use hash-based enforcement where
        # Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # ═══════════════════════════════════════════════════════════════════
        condition_key = params.get("condition_key")

        # Resolve preferred solution from LLM reasoning (opaque option)
        preferred = params.get("preferred_solution")
        if preferred:
            # Update entity to use the preferred manager
            params["manager"] = self._resolve_option(preferred)

        try:
            # ═══════════════════════════════════════════════════════════════════
            # DOCKER CODE EXECUTION (for run_code action when enabled)
            # Uses SingleTurnDockerAgent internally for full feature support
            # ═══════════════════════════════════════════════════════════════════
            if (
                action == "run_code"
                and self.enable_docker_execution
                and (self._multiturn_agent or self._docker_agent or self._code_executor)
            ):
                return await self._execute_with_docker(parsed_task, mcp_client)

            # ═══════════════════════════════════════════════════════════════════
            # Route to appropriate MCP tool based on action type
            # ═══════════════════════════════════════════════════════════════════
            if action == "install_package":
                manager = params.get("manager", "pip")
                if condition_key:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_coding_multi_condition",
                        {
                            "condition_key": condition_key,
                            "manager": manager,
                            "package": entity,
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "install_package", {"manager": manager, "package": entity}
                    )
                strategy_used = f"{action}:{manager}:{entity}"

            elif action == "import_module":
                solution = params.get("preferred_solution", "direct_import")
                response = await mcp_client.call_tool(
                    "import_module", {"module": entity, "strategy": solution}
                )
                strategy_used = f"{action}:{entity}:{solution}"

            elif action == "run_code":
                # Simulated execution via MCP (Docker not enabled)
                mode = params.get("preferred_solution", "default")
                response = await mcp_client.call_tool(
                    "run_code", {"task": parsed_task.raw_task, "mode": mode}
                )
                strategy_used = f"{action}:{mode}"

            elif action == "check_unique":
                strategy = params.get("preferred_solution", "python_check")
                response = await mcp_client.call_tool(
                    "check_unique", {"task": parsed_task.raw_task, "strategy": strategy}
                )
                strategy_used = f"{action}:{strategy}"

            elif action == "update_counter":
                strategy = params.get("preferred_solution", "direct_update")
                response = await mcp_client.call_tool(
                    "update_counter",
                    {"task": parsed_task.raw_task, "strategy": strategy},
                )
                strategy_used = f"{action}:{strategy}"

            elif action == "refactor_imports":
                strategy = params.get("preferred_solution", "direct_refactor")
                response = await mcp_client.call_tool(
                    "refactor_imports",
                    {"task": parsed_task.raw_task, "strategy": strategy},
                )
                strategy_used = f"{action}:{strategy}"

            else:
                # Fallback to run_code for unknown actions
                response = await mcp_client.call_tool(
                    "run_code", {"task": parsed_task.raw_task, "mode": "default"}
                )
                strategy_used = "run_code:default"

            # Parse response
            response_str = str(response) if response else ""

            if "SUCCESS" in response_str:
                # 🎓 LEARN: Remember what worked AND persist to disk
                await self._record_success(mcp_client, parsed_task, strategy_used)

                return ActionResult(
                    success=True,
                    response=response_str,
                    strategy_used=strategy_used,
                )

            # FAILED - extract error code
            error_code = CodingDomainConfig.extract_error_code(response_str)

            return ActionResult(
                success=False,
                response=response_str,
                error_code=error_code,
                strategy_used=strategy_used,
            )

        except Exception as e:
            return ActionResult(
                success=False,
                response=f"MCP call failed: {str(e)}",
                error_code="MCP-ERROR",
            )

    async def _execute_with_docker(
        self,
        parsed_task: ParsedTask,
        mcp_client: Any,
    ) -> ActionResult:
        """
        Execute code using Docker container with real execution feedback.

        This is the key feature for dynamic learning:
        1. Generates code via LLM if not provided (NO HARDCODING)
        2. Executes real Python code in Docker (via SingleTurnDockerAgent)
        3. Captures all output (stdout, stderr, exit code)
        4. Processes feedback to extract errors/warnings
        5. Updates dynamic config based on results
        6. Learns from execution for future improvement
        7. Syncs learned patterns to MCP server

        Args:
            parsed_task: The parsed task with code to execute
            mcp_client: MCP client for storing learned patterns

        Returns:
            ActionResult with execution details
        """
        # Extract code from task
        code = self._extract_code_from_task(parsed_task)
        mode = parsed_task.parameters.get("preferred_solution", "default")

        # ═══════════════════════════════════════════════════════════════════════
        # LLM-BASED CODE GENERATION (NO HARDCODING)
        # If code needs generation, use LLM to create it dynamically
        # This ensures fair comparison with Baseline (both use LLM for code gen)
        # ═══════════════════════════════════════════════════════════════════════
        if code.startswith("__GENERATE_CODE__:"):
            task_description = code[len("__GENERATE_CODE__:") :]
            code = await self._generate_code_via_llm(task_description, parsed_task)

        # Determine error category for Docker agent
        category = parsed_task.parameters.get("category", "UNKNOWN")

        # ═══════════════════════════════════════════════════════════════════════
        # USE DOCKER AGENT FOR EXECUTION (Composition Pattern)
        # This gives us all Docker agent features: execution, learning, persistence
        # Supports both SingleTurnDockerAgent and MultiTurnDockerAgent
        # ═══════════════════════════════════════════════════════════════════════

        # Get the active agent (multi-turn or single-turn)
        active_agent = self._multiturn_agent or self._docker_agent

        if active_agent:
            # Use the active agent's execute method
            if self._multiturn_agent:
                # MultiTurnDockerAgent uses chat_with_code
                execution_result, response = await self._multiturn_agent.chat_with_code(
                    message=parsed_task.raw_task,
                    code=code,
                )
                # Get feedback from the result
                feedback = (
                    await self._feedback_processor.process_result(execution_result)
                    if self._feedback_processor
                    else None
                )
            else:
                # SingleTurnDockerAgent uses execute_and_learn
                execution_result, feedback = await self._docker_agent.execute_and_learn(
                    code=code,
                    scenario_name=parsed_task.raw_task[:50],
                    category=category,
                )

            # Sync Docker agent's learning to MCP server
            await self._sync_agent_learning_to_mcp(mcp_client, feedback)

        else:
            # Fallback: Direct executor usage
            execution_result = await self._code_executor.execute_python(code)

            # Process feedback for learning
            feedback = None
            if self._feedback_processor:
                feedback = await self._feedback_processor.process_result(
                    execution_result
                )

                # DYNAMIC LEARNING FROM EXECUTION
                await self._learn_from_execution(feedback, mcp_client)

        strategy_used = f"run_code:docker:{mode}"

        if execution_result.success:
            # Record successful execution
            if self._dynamic_config:
                self._dynamic_config.record_execution(
                    code=code,
                    success=True,
                    execution_time=execution_result.execution_time,
                )

            return ActionResult(
                success=True,
                response=f"SUCCESS: {execution_result.stdout[:500]}",
                strategy_used=strategy_used,
            )

        # Execution failed - extract error information
        error_code = "UNKNOWN-ERROR"
        if feedback and hasattr(feedback, "error_category") and feedback.error_category:
            error_code = (
                feedback.error_category.value
                if hasattr(feedback.error_category, "value")
                else str(feedback.error_category)
            )
        elif execution_result.error_type:
            error_code = execution_result.error_type
        else:
            # Try to extract from CodingDomainConfig patterns
            error_code = CodingDomainConfig.extract_error_code(execution_result.stderr)

        # Also check dynamic config for learned patterns
        if self._dynamic_config:
            learned_code = self._dynamic_config.get_error_code(execution_result.stderr)
            if learned_code:
                error_code = learned_code

            # Record failed execution
            self._dynamic_config.record_execution(
                code=code,
                success=False,
                error_type=error_code,
                error_message=execution_result.error_message,
                execution_time=execution_result.execution_time,
            )

        # Build detailed error response
        error_response = f"ERROR [{error_code}]: {execution_result.stderr[:500]}"
        if execution_result.traceback:
            error_response += f"\n\nTraceback:\n{execution_result.traceback[:500]}"

        return ActionResult(
            success=False,
            response=error_response,
            error_code=error_code,
            strategy_used=strategy_used,
        )

    async def _sync_agent_learning_to_mcp(
        self,
        mcp_client: Any,
        feedback: Any,
    ) -> None:
        """
        Sync Docker agent's learned patterns to MCP server after execution.

        This ensures MCP server has all the knowledge from Docker agent.
        Works with both SingleTurnDockerAgent and MultiTurnDockerAgent.

        Args:
            mcp_client: MCP client for server communication
            feedback: ProcessedFeedback from execution
        """
        # Get the active agent
        active_agent = self._multiturn_agent or self._docker_agent

        if not active_agent or not mcp_client:
            return

        try:
            # Get newly learned rules from the active agent
            learned_rules = active_agent.export_learned_rules()

            # Store each new rule as a domain mapping in MCP
            for rule in learned_rules:
                # Parse rule to extract pattern and solution
                if "→" in rule:
                    parts = rule.split("→")
                    if len(parts) == 2:
                        pattern = parts[0].strip()
                        solution = parts[1].strip()

                        # Determine mapping type
                        if "Error" in pattern or "ERROR" in pattern:
                            mapping_type = "error_patterns"
                        else:
                            mapping_type = "recovery_solutions"

                        await mcp_client.call_tool(
                            "store_domain_mapping",
                            {
                                "domain": "coding",
                                "mapping_type": mapping_type,
                                "key": pattern[:100],
                                "value": solution,
                            },
                        )

            # Also store as experience for episodic memory
            if feedback:
                error_type = None
                if hasattr(feedback, "error_category") and feedback.error_category:
                    error_type = (
                        feedback.error_category.value
                        if hasattr(feedback.error_category, "value")
                        else str(feedback.error_category)
                    )

                await mcp_client.store_experience(
                    task=f"Docker execution: {feedback.original_code[:100] if hasattr(feedback, 'original_code') else 'code'}",
                    outcome="success" if not error_type else "failure",
                    strategy="docker_execution",
                    lessons=f"Error: {error_type}"
                    if error_type
                    else "Execution succeeded",
                    domain="coding",
                )

        except Exception:
            pass  # Sync is best-effort

    def _extract_code_from_task(self, parsed_task: ParsedTask) -> str:
        """
        Extract Python code from task description.

        Handles various formats:
        - Direct code in parameters
        - Code blocks in task description
        - File references
        - Generate realistic code for task (for fair comparison)

        Args:
            parsed_task: The parsed task

        Returns:
            Python code string
        """
        params = parsed_task.parameters

        # Check for explicit code parameter
        if "code" in params:
            return params["code"]

        # Check for code block in task
        raw_task = parsed_task.raw_task

        # Try to extract code block (```python ... ```)
        code_block_pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(code_block_pattern, raw_task, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to extract inline code
        if ":" in raw_task and "Run:" in raw_task:
            # Format: "Run: print('hello')"
            _, code = raw_task.split(":", 1)
            return code.strip()

        # ═══════════════════════════════════════════════════════════════════════
        # NO HARDCODED CODE - Return task description for LLM-based generation
        # The actual code generation happens via LLM in execute_action or Docker agent
        # This maintains PRECEPT's principle: "NO HARDCODED KNOWLEDGE"
        # ═══════════════════════════════════════════════════════════════════════

        # Store task for LLM code generation (will be used by Docker agent or execute_action)
        # The code will be generated dynamically by LLM based on task + learned rules
        return f"__GENERATE_CODE__:{raw_task}"

    async def _generate_code_via_llm(
        self,
        task_description: str,
        parsed_task: ParsedTask,
    ) -> str:
        """
        Generate Python code via LLM for the given task.

        This is the PRECEPT way - NO HARDCODED CODE TEMPLATES.
        The LLM generates code dynamically based on:
        1. Task description
        2. Learned rules from past experience
        3. Current context

        This ensures fair comparison with Baseline (both use LLM for code generation).

        Args:
            task_description: The task to generate code for
            parsed_task: Parsed task with additional context

        Returns:
            Generated Python code string
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI()
            try:
                # Build context from learned rules
                learned_context = ""
                if self._dynamic_config and hasattr(
                    self._dynamic_config, "get_learned_rules"
                ):
                    learned_rules = self._dynamic_config.get_learned_rules()
                    if learned_rules:
                        learned_context = (
                            "\n\nLearned patterns from past experience:\n"
                            + "\n".join(f"- {r}" for r in learned_rules[:5])
                        )

                # System prompt for code generation
                system_prompt = f"""You are an expert Python developer. Generate Python code to accomplish the given task.

Requirements:
1. Generate ONLY executable Python code (no markdown, no explanations)
2. Include proper error handling with try/except
3. Print success/error messages
4. Use sys.exit(1) on failure, sys.exit(0) or normal exit on success
5. Import all required modules at the top

Action type: {parsed_task.action}
Entity: {parsed_task.entity or "N/A"}
{learned_context}

Generate clean, working Python code."""

                # Generate code
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Generate Python code for this task:\n{task_description}",
                        },
                    ],
                    temperature=0.2,
                    max_tokens=1000,
                )

                code = response.choices[0].message.content.strip()

                # Extract code from markdown if present
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                return code
            finally:
                try:
                    await client.close()
                except Exception:
                    pass

        except Exception as e:
            # Fallback: Return simple code that will fail gracefully
            return f"""import sys

# LLM code generation failed: {str(e)[:50]}
# Task: {task_description[:100]}

print("Error: Could not generate code for task")
sys.exit(1)
"""

    async def _learn_from_execution(
        self,
        feedback: Any,  # ProcessedFeedback
        mcp_client: Any,
    ) -> None:
        """
        Learn from execution feedback and update dynamic config.

        This is the core of dynamic learning:
        1. If new error pattern discovered, add to config
        2. If recovery succeeded, record the solution
        3. Sync changes to MCP and JSON

        Args:
            feedback: ProcessedFeedback from execution
            mcp_client: MCP client for persistence
        """
        if not feedback or not self._dynamic_config:
            return

        # ═══════════════════════════════════════════════════════════════════════
        # LEARN NEW ERROR PATTERNS
        # ═══════════════════════════════════════════════════════════════════════
        if feedback.has_new_error_pattern and feedback.error_pattern:
            error_code = (
                feedback.error_category.value
                if feedback.error_category
                else "UNKNOWN-ERROR"
            )

            # Add to dynamic config
            added = self._dynamic_config.add_error_pattern(
                feedback.error_pattern,
                error_code,
            )

            if added:
                print(
                    f"    🎓 Learned new error pattern: {feedback.error_pattern[:50]}... → {error_code}"
                )

                # Persist to MCP
                try:
                    await mcp_client.call_tool(
                        "store_domain_mapping",
                        {
                            "domain": "coding",
                            "mapping_type": "error_patterns",
                            "key": feedback.error_pattern,
                            "value": error_code,
                        },
                    )
                except Exception:
                    pass  # MCP persistence is best-effort

        # ═══════════════════════════════════════════════════════════════════════
        # LEARN RECOVERY SOLUTIONS
        # ═══════════════════════════════════════════════════════════════════════
        if feedback.suggested_recovery and feedback.error_category:
            error_code = feedback.error_category.value

            # Add to dynamic config
            added = self._dynamic_config.add_recovery_solution(
                error_code,
                feedback.suggested_recovery,
            )

            if added:
                print(
                    f"    🎓 Learned recovery: {error_code} → {feedback.suggested_recovery}"
                )

                # Persist to MCP
                try:
                    await mcp_client.call_tool(
                        "store_domain_mapping",
                        {
                            "domain": "coding",
                            "mapping_type": "recovery_solutions",
                            "key": error_code,
                            "value": feedback.suggested_recovery,
                        },
                    )
                except Exception:
                    pass

        # ═══════════════════════════════════════════════════════════════════════
        # LEARN FROM WARNINGS (Proactive improvement)
        # ═══════════════════════════════════════════════════════════════════════
        for warning in feedback.warnings:
            # Store deprecation warnings for future reference
            if "Deprecation" in warning.category:
                try:
                    await mcp_client.store_experience(
                        task=f"Warning: {warning.message[:100]}",
                        outcome="warning",
                        strategy="deprecation_notice",
                        lessons=f"Consider updating: {warning.suggestion or 'check documentation'}",
                        domain="coding",
                    )
                except Exception:
                    pass

        # Save config to JSON (periodic save)
        try:
            self._dynamic_config.save_to_json()
        except Exception:
            pass

    def get_learned_rules_from_execution(self) -> List[str]:
        """
        Export learned rules from dynamic config for prompt evolution.

        Returns:
            List of rule strings for COMPASS prompt evolution
        """
        if not self._dynamic_config:
            return []

        return self._dynamic_config.export_learned_rules()

    async def _record_success(
        self,
        mcp_client: Any,
        parsed_task: ParsedTask,
        strategy_used: str,
    ):
        """
        Record successful strategy for future learning AND persist to MCP.

        This ensures learning survives across sessions!

        CRITICAL FOR MULTI-CONDITION: Uses condition_key for deterministic rule storage.
        """
        action = parsed_task.action
        entity = parsed_task.entity
        params = parsed_task.parameters or {}

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION RULE STORAGE: Use condition_key if available
        # This is CRITICAL for PRECEPT's O(1) lookup advantage!
        # ═══════════════════════════════════════════════════════════════════
        condition_key = params.get("condition_key")
        solution = params.get("preferred_solution") or params.get("manager", "pip")

        if condition_key and solution:
            # Store rule using condition_key for exact O(1) lookup
            await mcp_client.record_solution(
                error_code=condition_key,  # Multi-condition key: PKG-404+ENV-ARM+...
                solution=solution,  # The working solution: conda, poetry, etc.
                context=f"Coding multi-condition (conditions: {condition_key})",
            )

        if action == "install_package" and entity and entity != "unknown":
            manager = params.get("manager", "pip")
            self._learned_package_managers[entity] = manager

            # 💾 PERSIST: Store package→manager mapping
            await self.persist_mapping(
                mcp_client,
                "package_managers",
                entity,
                manager,
            )

        # Extract context pattern from task for non-install actions
        if action != "install_package":
            # Create a simplified context key from the task
            task_words = parsed_task.raw_task.lower().split()[:5]
            context_key_simple = " ".join(task_words)
            sol = params.get("preferred_solution", "default")
            if sol != "default":
                self._learned_context_solutions[context_key_simple] = sol

                # 💾 PERSIST: Store context→solution mapping
                await self.persist_mapping(
                    mcp_client,
                    "context_solutions",
                    context_key_simple,
                    sol,
                )

    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle error - try alternatives with LEARNED knowledge.

        Supports ALL error types:
        - Dependency errors: Try different package managers
        - Crash errors: Try different execution modes
        - Concurrency errors: Try different sync strategies
        - Import errors: Try different import strategies

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        """
        import random

        # Record error for learning
        await mcp_client.record_error(
            error_code,
            f"{parsed_task.action} {parsed_task.entity} with {parsed_task.source}",
        )

        # Track retries
        retries_made = context.get("retries_made", 0)
        tried_solutions = context.get("tried_solutions", set())

        # ═══════════════════════════════════════════════════════════════════════
        # Get recovery options based on ACTION TYPE and error type
        # ═══════════════════════════════════════════════════════════════════════
        recovery_options = CodingDomainConfig.get_recovery_options(error_code)

        # For install_package actions, ALWAYS use package managers as recovery
        # This handles cases like legacy_orm which fails with PKG-XXX
        # but the solution is to use a different package manager
        if parsed_task.action == "install_package":
            recovery_options = [
                m for m in self.PACKAGE_MANAGERS if m != parsed_task.source
            ]
        # For package errors (PKG-XXX), also use package managers
        # Uses prefix matching for VAGUE error codes - NO hardcoded old codes!
        elif error_code.startswith("PKG-"):
            recovery_options = [
                m for m in self.PACKAGE_MANAGERS if m != parsed_task.source
            ]

        # Filter out already tried solutions
        remaining = [opt for opt in recovery_options if opt not in tried_solutions]

        # ═══════════════════════════════════════════════════════════════════════
        # PRECEPT's advantage: Prioritize learned solutions
        # ═══════════════════════════════════════════════════════════════════════
        error_context_key = f"{error_code}:{parsed_task.entity}"
        if error_context_key in self._learned_error_solutions:
            learned_solution = self._learned_error_solutions[error_context_key]
            if learned_solution in remaining:
                remaining.remove(learned_solution)
                remaining.insert(0, learned_solution)  # Try learned solution first

        # Shuffle unknown solutions (fair comparison with baseline)
        if len(remaining) > 1:
            known = (
                remaining[:1]
                if error_context_key in self._learned_error_solutions
                else []
            )
            unknown = remaining[1:] if known else remaining
            random.shuffle(unknown)
            remaining = known + unknown

        # ═══════════════════════════════════════════════════════════════════════
        # Try recovery options within retry budget
        # ═══════════════════════════════════════════════════════════════════════
        for solution in remaining:
            if retries_made >= self.MAX_RETRIES:
                break

            tried_solutions.add(solution)
            retries_made += 1

            try:
                # Update parameters with new solution
                parsed_task.parameters["preferred_solution"] = solution
                if solution in self.PACKAGE_MANAGERS:
                    parsed_task.source = solution
                    parsed_task.parameters["manager"] = solution

                # Re-execute with new solution
                result = await self.execute_action(mcp_client, parsed_task)

                if result.success:
                    # 🎓 LEARN: Store what worked for this error + context
                    self._learned_error_solutions[error_context_key] = solution

                    # 💾 PERSIST: Store error→solution mapping
                    await self.persist_mapping(
                        mcp_client,
                        "error_solutions",
                        error_context_key,
                        solution,
                    )

                    # ═══════════════════════════════════════════════════════════
                    # MULTI-CONDITION RULE STORAGE: Use condition_key if available
                    # This is CRITICAL for PRECEPT's O(1) lookup advantage!
                    # ═══════════════════════════════════════════════════════════
                    condition_key = (parsed_task.parameters or {}).get("condition_key")
                    if condition_key:
                        await mcp_client.record_solution(
                            error_code=condition_key,  # Multi-condition key
                            solution=solution,  # The working solution
                            context=f"Coding recovery (conditions: {condition_key})",
                        )

                    result.strategy_used = (
                        f"Recovery:{solution} (retry {retries_made}/{self.MAX_RETRIES})"
                    )
                    return result

            except Exception:
                continue

        return ActionResult(
            success=False,
            response=f"All retries exhausted ({retries_made}/{self.MAX_RETRIES})",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        """Return coding-specific tools for all action types."""
        if not AUTOGEN_AVAILABLE:
            return []

        tools = []

        # Generate tools for each action type
        async def install_package(manager: str, package: str) -> str:
            """Install a package using specified manager."""
            return f"Installing {package} with {manager}"

        tools.append(
            FunctionTool(install_package, description="Install a Python package.")
        )

        async def import_module(name: str, strategy: str = "direct") -> str:
            """Import a Python module with specified strategy."""
            return f"Importing {name} using {strategy}"

        tools.append(FunctionTool(import_module, description="Import a Python module."))

        async def run_code(task: str, mode: str = "default") -> str:
            """Execute code with specified mode."""
            return f"Running code in {mode} mode: {task[:50]}..."

        tools.append(FunctionTool(run_code, description="Execute Python code."))

        async def check_unique(task: str, strategy: str = "python_check") -> str:
            """Check uniqueness using specified strategy."""
            return f"Checking uniqueness with {strategy}: {task[:50]}..."

        tools.append(
            FunctionTool(check_unique, description="Check uniqueness before insert.")
        )

        async def update_counter(task: str, strategy: str = "direct") -> str:
            """Update counter using specified strategy."""
            return f"Updating counter with {strategy}: {task[:50]}..."

        tools.append(
            FunctionTool(update_counter, description="Update a counter safely.")
        )

        async def refactor_imports(task: str, strategy: str = "direct") -> str:
            """Refactor imports using specified strategy."""
            return f"Refactoring imports with {strategy}: {task[:50]}..."

        tools.append(
            FunctionTool(refactor_imports, description="Refactor module imports.")
        )

        return tools


class CodingBaselineStrategy(BaselineDomainStrategy):
    """
    Coding baseline strategy - NO LEARNING.

    Behavior:
    - Always tries default manager/strategy first
    - On failure, tries RANDOM alternative options
    - Does NOT know which packages have issues
    - Does NOT learn from failures

    Uses CodingDomainConfig for shared vocabulary (DRY principle).
    Supports ALL scenario types but without learning capability.
    """

    # Reference shared configuration (Single Source of Truth)
    PACKAGE_MANAGERS = CodingDomainConfig.PACKAGE_MANAGERS
    KNOWN_PACKAGES = CodingDomainConfig.KNOWN_PACKAGES

    # Share opaque options with PRECEPT strategy for fair comparison
    MANAGER_OPTIONS_MAP = CodingDomainStrategy.MANAGER_OPTIONS_MAP
    MANAGER_OPTIONS_REVERSE = CodingDomainStrategy.MANAGER_OPTIONS_REVERSE

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the Coding baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "coding"

    def get_available_options(self) -> List[str]:
        """Return OPAQUE options SHUFFLED for fair exploration."""
        import random

        options = list(self.MANAGER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str) -> str:
        """Convert opaque option back to real package manager."""
        return self.MANAGER_OPTIONS_MAP.get(opaque_option, opaque_option)

    def get_options_for_error(self, error_code: str) -> List[str]:
        """Get recovery options SHUFFLED for fair exploration."""
        import random

        options = list(self.MANAGER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def parse_task(self, task: str) -> ParsedTask:
        """Parse task and extract action type, entity, and parameters."""
        import re

        task_lower = task.lower()

        # Detect action type using config
        action = CodingDomainConfig.get_action_type(task)

        # Extract package name using shared vocabulary
        entity = "unknown"
        for pkg in self.KNOWN_PACKAGES:
            if pkg.lower() in task_lower:
                entity = pkg
                break

        # Build parameters
        parameters = {
            "action_type": action,
            "package": entity,
            "raw_task": task,
        }

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z] pattern
        # CRITICAL: Must match PRECEPT strategy to ensure fair comparison
        # ═══════════════════════════════════════════════════════════════════
        # BUGFIX: Parse and SORT conditions like PRECEPT does. The old code
        # used the raw string without sorting, producing a different
        # condition_key than PRECEPT for the same conditions. Since the hash
        # is computed on condition_key, this caused different hash results
        # and made baselines always fail multi-condition enforcement.
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))
            parameters["condition_key"] = condition_key
            parameters["conditions"] = conditions

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source="pip",
            parameters=parameters,
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        """Get default option based on action type."""
        action = parsed_task.action
        if action == "install_package":
            return "pip"
        elif action == "run_code":
            return "default"
        elif action == "check_unique":
            return "python_check"
        elif action == "update_counter":
            return "direct_update"
        elif action == "import_module":
            return "direct_import"
        elif action == "refactor_imports":
            return "direct_refactor"
        return "default"

    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """
        Execute via MCP server - NO LEARNING, just tries options.

        Supports all action types defined in CodingDomainConfig.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)). This is FAIR - baselines face the
        same strict enforcement as PRECEPT.
        """
        action = parsed_task.action
        entity = parsed_task.entity or "unknown"
        condition_key = parsed_task.parameters.get("condition_key")

        # Resolve opaque option to real manager name for MCP call
        real_option = self._resolve_option(option)

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use the multi-condition tool which
        # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # This is FAIR - baselines face the same strict enforcement as PRECEPT.
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Route to appropriate MCP tool based on action type
            if action == "install_package":
                if condition_key:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_coding_multi_condition",
                        {
                            "condition_key": condition_key,
                            "manager": real_option,
                            "package": entity,
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "install_package", {"manager": real_option, "package": entity}
                    )
            elif action == "import_module":
                response = await mcp_client.call_tool(
                    "import_module", {"module": entity, "strategy": option}
                )
            elif action == "run_code":
                response = await mcp_client.call_tool(
                    "run_code", {"task": parsed_task.raw_task, "mode": option}
                )
            elif action == "check_unique":
                response = await mcp_client.call_tool(
                    "check_unique", {"task": parsed_task.raw_task, "strategy": option}
                )
            elif action == "update_counter":
                response = await mcp_client.call_tool(
                    "update_counter", {"task": parsed_task.raw_task, "strategy": option}
                )
            elif action == "refactor_imports":
                response = await mcp_client.call_tool(
                    "refactor_imports",
                    {"task": parsed_task.raw_task, "strategy": option},
                )
            else:
                # Fallback
                response = await mcp_client.call_tool(
                    "run_code", {"task": parsed_task.raw_task, "mode": "default"}
                )

            if isinstance(response, str):
                if "SUCCESS" in response:
                    return True, response
                else:
                    return False, response

            return True, str(response)

        except Exception as e:
            return False, f"MCP call failed: {str(e)}"
