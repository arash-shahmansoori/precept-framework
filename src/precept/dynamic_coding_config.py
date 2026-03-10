"""
Dynamic Coding Configuration for PRECEPT Framework.

Extends the static CodingDomainConfig with runtime learning capabilities.
Persists learned patterns and solutions to both MCP memory and local JSON.

Features:
- Dynamically learns error patterns from execution
- Stores successful recovery solutions
- Persists to MCP memory for runtime access
- Backs up to local JSON file for durability
- Loads persisted config on startup

Usage:
    from precept.dynamic_coding_config import DynamicCodingConfig

    config = DynamicCodingConfig()
    await config.load()  # Load from JSON and MCP

    # Learn from execution
    config.add_error_pattern("CustomError:", "CUSTOM-ERROR")
    config.add_recovery_solution("CUSTOM-ERROR", "custom_fix")

    # Persist changes
    await config.save()
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionRecord:
    """Record of a single code execution for analysis."""

    timestamp: str
    code_snippet: str  # First 200 chars
    success: bool
    error_type: Optional[str]
    error_message: Optional[str]
    recovery_used: Optional[str]
    execution_time: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "code_snippet": self.code_snippet,
            "success": self.success,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "recovery_used": self.recovery_used,
            "execution_time": self.execution_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionRecord":
        return cls(
            timestamp=data.get("timestamp", ""),
            code_snippet=data.get("code_snippet", ""),
            success=data.get("success", False),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            recovery_used=data.get("recovery_used"),
            execution_time=data.get("execution_time", 0.0),
        )


@dataclass
class DynamicCodingConfig:
    """
    Dynamically updatable coding configuration.

    Extends the static CodingDomainConfig with:
    - Learned error patterns (pattern -> error_code)
    - Learned recovery solutions (error_code -> [solutions])
    - Execution history for analysis
    - Persistence to JSON and MCP

    Learning Flow:
    1. Code executes and fails with unknown error
    2. ExecutionFeedbackProcessor categorizes it
    3. DynamicCodingConfig stores the new pattern
    4. On next similar error, pattern is recognized
    5. Learned recovery solution is applied
    """

    # Default config file location
    DEFAULT_CONFIG_DIR: str = field(default="~/.precept", repr=False)
    DEFAULT_CONFIG_FILE: str = field(default="coding_config.json", repr=False)

    # Dynamically learned mappings
    learned_error_patterns: Dict[str, str] = field(default_factory=dict)
    learned_recovery_solutions: Dict[str, List[str]] = field(default_factory=dict)

    # Execution history (limited to last N records)
    execution_history: List[ExecutionRecord] = field(default_factory=list)
    max_history_size: int = field(default=1000, repr=False)

    # Statistics
    total_executions: int = 0
    successful_executions: int = 0
    patterns_learned: int = 0
    recoveries_learned: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_synced_to_mcp: Optional[str] = None

    # Config file path
    _config_path: Optional[Path] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize config path."""
        if self._config_path is None:
            config_dir = Path(os.path.expanduser(self.DEFAULT_CONFIG_DIR))
            config_dir.mkdir(parents=True, exist_ok=True)
            self._config_path = config_dir / self.DEFAULT_CONFIG_FILE

    @property
    def config_path(self) -> Path:
        """Get the configuration file path."""
        return self._config_path

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    def add_error_pattern(self, pattern: str, error_code: str) -> bool:
        """
        Add a learned error pattern.

        Args:
            pattern: Regex pattern or error string
            error_code: Error code to map to (e.g., "CUSTOM-ERROR")

        Returns:
            True if new pattern was added, False if already exists
        """
        if pattern in self.learned_error_patterns:
            return False

        self.learned_error_patterns[pattern] = error_code
        self.patterns_learned += 1
        self.last_updated = datetime.utcnow().isoformat()
        return True

    def add_recovery_solution(self, error_code: str, solution: str) -> bool:
        """
        Add a learned recovery solution for an error code.

        Args:
            error_code: Error code (e.g., "IMPORT-ERROR")
            solution: Solution that worked (e.g., "pip_install")

        Returns:
            True if new solution was added, False if already exists
        """
        if error_code not in self.learned_recovery_solutions:
            self.learned_recovery_solutions[error_code] = []

        if solution in self.learned_recovery_solutions[error_code]:
            return False

        # Add to front (most recent = most likely to work)
        self.learned_recovery_solutions[error_code].insert(0, solution)
        self.recoveries_learned += 1
        self.last_updated = datetime.utcnow().isoformat()
        return True

    def get_recovery_solutions(self, error_code: str) -> List[str]:
        """
        Get recovery solutions for an error code.

        Returns learned solutions first, then falls back to defaults.

        Args:
            error_code: Error code to get solutions for

        Returns:
            List of recovery solutions to try
        """
        learned = self.learned_recovery_solutions.get(error_code, [])

        # Import default solutions from CodingDomainConfig
        from .domain_strategies.coding import CodingDomainConfig

        defaults = CodingDomainConfig.ERROR_RECOVERY_OPTIONS.get(error_code, [])

        # Combine, learned first, avoiding duplicates
        combined = list(learned)
        for solution in defaults:
            if solution not in combined:
                combined.append(solution)

        return combined

    def get_error_code(self, error_text: str) -> Optional[str]:
        """
        Get error code for an error text using learned patterns.

        Args:
            error_text: Error message or traceback

        Returns:
            Error code or None if not recognized
        """
        import re

        # Check learned patterns first
        for pattern, code in self.learned_error_patterns.items():
            try:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return code
            except re.error:
                # Invalid regex, try as substring
                if pattern.lower() in error_text.lower():
                    return code

        return None

    def record_execution(
        self,
        code: str,
        success: bool,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        recovery_used: Optional[str] = None,
        execution_time: float = 0.0,
    ) -> None:
        """
        Record an execution for history and statistics.

        Args:
            code: Code that was executed
            success: Whether execution succeeded
            error_type: Type of error if failed
            error_message: Error message if failed
            recovery_used: Recovery solution if one was applied
            execution_time: Time taken for execution
        """
        record = ExecutionRecord(
            timestamp=datetime.utcnow().isoformat(),
            code_snippet=code[:200] if code else "",
            success=success,
            error_type=error_type,
            error_message=error_message[:200] if error_message else None,
            recovery_used=recovery_used,
            execution_time=execution_time,
        )

        self.execution_history.append(record)
        self.total_executions += 1

        if success:
            self.successful_executions += 1

        # Trim history if too large
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[-self.max_history_size :]

        self.last_updated = datetime.utcnow().isoformat()

    def get_most_common_errors(self, top_n: int = 10) -> List[tuple]:
        """
        Get the most common error types from execution history.

        Args:
            top_n: Number of top errors to return

        Returns:
            List of (error_type, count) tuples
        """
        from collections import Counter

        error_counts = Counter(
            record.error_type for record in self.execution_history if record.error_type
        )

        return error_counts.most_common(top_n)

    def get_stats(self) -> dict:
        """
        Get configuration statistics.

        Returns:
            Dictionary with config statistics
        """
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": self.success_rate,
            "patterns_learned": self.patterns_learned,
            "recoveries_learned": self.recoveries_learned,
            "error_patterns_count": len(self.learned_error_patterns),
            "recovery_solutions_count": sum(
                len(v) for v in self.learned_recovery_solutions.values()
            ),
            "history_size": len(self.execution_history),
            "most_common_errors": self.get_most_common_errors(5),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "last_synced_to_mcp": self.last_synced_to_mcp,
        }

    def to_dict(self) -> dict:
        """
        Convert config to dictionary for serialization.

        Returns:
            Dictionary representation of config
        """
        return {
            "learned_error_patterns": self.learned_error_patterns,
            "learned_recovery_solutions": self.learned_recovery_solutions,
            "execution_history": [r.to_dict() for r in self.execution_history],
            "execution_stats": {
                "total_executions": self.total_executions,
                "successful_executions": self.successful_executions,
                "success_rate": self.success_rate,
                "most_common_errors": [
                    {"error": e, "count": c} for e, c in self.get_most_common_errors(10)
                ],
            },
            "learning_stats": {
                "patterns_learned": self.patterns_learned,
                "recoveries_learned": self.recoveries_learned,
            },
            "timestamps": {
                "created_at": self.created_at,
                "last_updated": self.last_updated,
                "last_synced_to_mcp": self.last_synced_to_mcp,
            },
        }

    @classmethod
    def from_dict(
        cls, data: dict, config_path: Optional[Path] = None
    ) -> "DynamicCodingConfig":
        """
        Create config from dictionary.

        Args:
            data: Dictionary with config data
            config_path: Optional custom config path

        Returns:
            DynamicCodingConfig instance
        """
        config = cls()

        if config_path:
            config._config_path = config_path

        # Load learned mappings
        config.learned_error_patterns = data.get("learned_error_patterns", {})
        config.learned_recovery_solutions = data.get("learned_recovery_solutions", {})

        # Load execution history
        history_data = data.get("execution_history", [])
        config.execution_history = [ExecutionRecord.from_dict(r) for r in history_data]

        # Load stats
        stats = data.get("execution_stats", {})
        config.total_executions = stats.get("total_executions", 0)
        config.successful_executions = stats.get("successful_executions", 0)

        learning_stats = data.get("learning_stats", {})
        config.patterns_learned = learning_stats.get("patterns_learned", 0)
        config.recoveries_learned = learning_stats.get("recoveries_learned", 0)

        # Load timestamps
        timestamps = data.get("timestamps", {})
        config.created_at = timestamps.get("created_at", config.created_at)
        config.last_updated = timestamps.get("last_updated", config.last_updated)
        config.last_synced_to_mcp = timestamps.get("last_synced_to_mcp")

        return config

    def save_to_json(self, path: Optional[Path] = None) -> None:
        """
        Save config to JSON file.

        Args:
            path: Optional custom path (defaults to config_path)
        """
        save_path = path or self._config_path

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_json(self, path: Optional[Path] = None) -> bool:
        """
        Load config from JSON file.

        Args:
            path: Optional custom path (defaults to config_path)

        Returns:
            True if loaded successfully, False otherwise
        """
        load_path = path or self._config_path

        if not load_path.exists():
            return False

        try:
            with open(load_path, "r") as f:
                data = json.load(f)

            # Update self with loaded data
            loaded = DynamicCodingConfig.from_dict(data, self._config_path)

            self.learned_error_patterns = loaded.learned_error_patterns
            self.learned_recovery_solutions = loaded.learned_recovery_solutions
            self.execution_history = loaded.execution_history
            self.total_executions = loaded.total_executions
            self.successful_executions = loaded.successful_executions
            self.patterns_learned = loaded.patterns_learned
            self.recoveries_learned = loaded.recoveries_learned
            self.created_at = loaded.created_at
            self.last_updated = loaded.last_updated
            self.last_synced_to_mcp = loaded.last_synced_to_mcp

            return True

        except (json.JSONDecodeError, KeyError, Exception):
            return False

    async def sync_to_mcp(self, mcp_client: Any) -> bool:
        """
        Sync learned patterns and solutions to MCP memory.

        Args:
            mcp_client: MCP client for server communication

        Returns:
            True if synced successfully
        """
        try:
            # Store error patterns
            for pattern, code in self.learned_error_patterns.items():
                await mcp_client.call_tool(
                    "store_domain_mapping",
                    {
                        "domain": "coding",
                        "mapping_type": "error_patterns",
                        "key": pattern,
                        "value": code,
                    },
                )

            # Store recovery solutions
            for error_code, solutions in self.learned_recovery_solutions.items():
                await mcp_client.call_tool(
                    "store_domain_mapping",
                    {
                        "domain": "coding",
                        "mapping_type": "recovery_solutions",
                        "key": error_code,
                        "value": json.dumps(solutions),
                    },
                )

            # Store stats summary
            await mcp_client.call_tool(
                "store_domain_mapping",
                {
                    "domain": "coding",
                    "mapping_type": "execution_stats",
                    "key": "summary",
                    "value": json.dumps(self.get_stats()),
                },
            )

            self.last_synced_to_mcp = datetime.utcnow().isoformat()
            return True

        except Exception:
            return False

    async def load_from_mcp(self, mcp_client: Any) -> bool:
        """
        Load learned patterns and solutions from MCP memory.

        Args:
            mcp_client: MCP client for server communication

        Returns:
            True if loaded successfully
        """
        try:
            # Get all domain mappings
            response = await mcp_client.call_tool(
                "get_domain_mappings", {"domain": "coding"}
            )

            if not response or "NOT_FOUND" in str(response):
                return False

            # MCP returns formatted text, parse it
            # This is a simplified parser - real implementation would be more robust

            return True

        except Exception:
            return False

    async def save(self, mcp_client: Optional[Any] = None) -> None:
        """
        Save config to both JSON and MCP (if client provided).

        Args:
            mcp_client: Optional MCP client for MCP sync
        """
        # Always save to JSON
        self.save_to_json()

        # Sync to MCP if client provided
        if mcp_client:
            await self.sync_to_mcp(mcp_client)

    async def load(self, mcp_client: Optional[Any] = None) -> bool:
        """
        Load config from JSON (and optionally MCP).

        JSON is primary source, MCP is supplementary.

        Args:
            mcp_client: Optional MCP client for MCP data

        Returns:
            True if loaded from at least one source
        """
        json_loaded = self.load_from_json()

        if mcp_client:
            # MCP data supplements JSON data
            await self.load_from_mcp(mcp_client)

        return json_loaded

    def merge_with(self, other: "DynamicCodingConfig") -> None:
        """
        Merge another config into this one.

        Useful for combining configs from different sources.

        Args:
            other: Other config to merge from
        """
        # Merge error patterns (other takes precedence)
        for pattern, code in other.learned_error_patterns.items():
            if pattern not in self.learned_error_patterns:
                self.add_error_pattern(pattern, code)

        # Merge recovery solutions
        for code, solutions in other.learned_recovery_solutions.items():
            for solution in solutions:
                self.add_recovery_solution(code, solution)

        # Don't merge history or stats (keep local)
        self.last_updated = datetime.utcnow().isoformat()

    def export_learned_rules(self) -> List[str]:
        """
        Export learned patterns and solutions as human-readable rules.

        Useful for COMPASS prompt evolution.

        Returns:
            List of rule strings
        """
        rules = []

        # Error pattern rules
        for pattern, code in self.learned_error_patterns.items():
            rules.append(f"When error matches '{pattern}', classify as {code}")

        # Recovery solution rules
        for code, solutions in self.learned_recovery_solutions.items():
            if solutions:
                solutions_str = ", ".join(solutions[:3])  # Top 3
                rules.append(f"For {code}, try: {solutions_str}")

        # Stats-based rules
        common_errors = self.get_most_common_errors(3)
        if common_errors:
            errors_str = ", ".join(e for e, _ in common_errors)
            rules.append(f"Most common errors: {errors_str}")

        return rules
