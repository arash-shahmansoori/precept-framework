"""
Execution Feedback Processor for PRECEPT Framework.

Parses execution output to extract errors, warnings, and logs.
Uses LLM to auto-categorize unknown error types for dynamic learning.

Features:
- Pattern-based error extraction from stderr/traceback
- LLM-powered auto-categorization of unknown errors
- Warning extraction and classification
- Structured feedback for config updates

Usage:
    from precept.execution_feedback_processor import ExecutionFeedbackProcessor
    from precept.code_executor import ExecutionResult

    processor = ExecutionFeedbackProcessor()
    feedback = await processor.process_result(execution_result)

    if feedback.has_new_error_pattern:
        print(f"New error discovered: {feedback.error_category}")
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Import ExecutionResult type
from .code_executor import ExecutionResult


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"  # Crashes, segfaults, system errors
    ERROR = "error"  # Standard Python errors
    WARNING = "warning"  # Deprecation warnings, etc.
    INFO = "info"  # Informational messages


class ErrorCategory(Enum):
    """
    Categories for error classification.

    These map to the error codes in CodingDomainConfig.
    """

    # Dependency errors
    IMPORT_ERROR = "IMPORT-ERROR"
    MODULE_NOT_FOUND = "MODULE-NOT-FOUND"
    DEPENDENCY_CONFLICT = "DEPENDENCY-CONFLICT"

    # Syntax and type errors
    SYNTAX_ERROR = "SYNTAX-ERROR"
    TYPE_ERROR = "TYPE-ERROR"
    NAME_ERROR = "NAME-ERROR"
    ATTRIBUTE_ERROR = "ATTRIBUTE-ERROR"

    # Runtime errors
    VALUE_ERROR = "VALUE-ERROR"
    INDEX_ERROR = "INDEX-ERROR"
    KEY_ERROR = "KEY-ERROR"
    RUNTIME_ERROR = "RUNTIME-ERROR"

    # System errors
    MEMORY_ERROR = "MEMORY-ERROR"
    TIMEOUT_ERROR = "TIMEOUT-ERROR"
    OS_ERROR = "OS-ERROR"
    FILE_NOT_FOUND = "FILE-NOT-FOUND"
    PERMISSION_ERROR = "PERMISSION-ERROR"

    # Critical errors
    SEGFAULT = "SEGFAULT-000"
    BUS_ERROR = "BUS-ERROR"

    # Concurrency errors
    RACE_CONDITION = "RACE-COND-409"
    DEADLOCK = "DEADLOCK-ERROR"

    # Unknown (for LLM categorization)
    UNKNOWN = "UNKNOWN-ERROR"


@dataclass
class Warning:
    """Represents a warning from code execution."""

    message: str
    category: str  # DeprecationWarning, UserWarning, etc.
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ProcessedFeedback:
    """
    Structured feedback from execution result processing.

    Contains all extracted information for learning and config updates.
    """

    # Original result
    original_result: ExecutionResult

    # Error information
    error_category: Optional[ErrorCategory] = None
    error_severity: ErrorSeverity = ErrorSeverity.INFO
    error_message: str = ""
    error_line: Optional[int] = None

    # Pattern information (for learning)
    error_pattern: Optional[str] = None  # Regex pattern that matches this error
    is_known_pattern: bool = False  # Whether this matched an existing pattern
    has_new_error_pattern: bool = False  # Whether this is a new pattern to learn

    # Suggested recovery
    suggested_recovery: Optional[str] = None
    recovery_confidence: float = 0.0

    # Warnings
    warnings: List[Warning] = field(default_factory=list)

    # LLM categorization (if used)
    llm_categorization: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "error_category": self.error_category.value
            if self.error_category
            else None,
            "error_severity": self.error_severity.value,
            "error_message": self.error_message,
            "error_line": self.error_line,
            "error_pattern": self.error_pattern,
            "is_known_pattern": self.is_known_pattern,
            "has_new_error_pattern": self.has_new_error_pattern,
            "suggested_recovery": self.suggested_recovery,
            "recovery_confidence": self.recovery_confidence,
            "warnings": [
                {"message": w.message, "category": w.category} for w in self.warnings
            ],
            "llm_categorization": self.llm_categorization,
        }


# Known error patterns and their categories
ERROR_PATTERNS: Dict[str, ErrorCategory] = {
    # Import/Module errors
    r"ModuleNotFoundError: No module named": ErrorCategory.MODULE_NOT_FOUND,
    r"ImportError: cannot import name": ErrorCategory.IMPORT_ERROR,
    r"ImportError:": ErrorCategory.IMPORT_ERROR,
    # Syntax errors
    r"SyntaxError:": ErrorCategory.SYNTAX_ERROR,
    r"IndentationError:": ErrorCategory.SYNTAX_ERROR,
    r"TabError:": ErrorCategory.SYNTAX_ERROR,
    # Type errors
    r"TypeError:": ErrorCategory.TYPE_ERROR,
    r"AttributeError:": ErrorCategory.ATTRIBUTE_ERROR,
    r"NameError:": ErrorCategory.NAME_ERROR,
    # Value/Index errors
    r"ValueError:": ErrorCategory.VALUE_ERROR,
    r"IndexError:": ErrorCategory.INDEX_ERROR,
    r"KeyError:": ErrorCategory.KEY_ERROR,
    # Runtime errors
    r"RuntimeError:": ErrorCategory.RUNTIME_ERROR,
    r"RecursionError:": ErrorCategory.RUNTIME_ERROR,
    r"StopIteration:": ErrorCategory.RUNTIME_ERROR,
    # System errors
    r"MemoryError:": ErrorCategory.MEMORY_ERROR,
    r"OSError:": ErrorCategory.OS_ERROR,
    r"FileNotFoundError:": ErrorCategory.FILE_NOT_FOUND,
    r"PermissionError:": ErrorCategory.PERMISSION_ERROR,
    r"TimeoutError:": ErrorCategory.TIMEOUT_ERROR,
    # Critical errors
    r"Segmentation fault": ErrorCategory.SEGFAULT,
    r"SIGSEGV": ErrorCategory.SEGFAULT,
    r"Bus error": ErrorCategory.BUS_ERROR,
    # Database/Concurrency
    r"IntegrityError:.*duplicate key": ErrorCategory.RACE_CONDITION,
    r"deadlock detected": ErrorCategory.DEADLOCK,
    r"Lock wait timeout": ErrorCategory.RACE_CONDITION,
}

# Recovery suggestions for each error category
RECOVERY_SUGGESTIONS: Dict[ErrorCategory, List[str]] = {
    ErrorCategory.MODULE_NOT_FOUND: [
        "pip_install",
        "conda_install",
        "check_spelling",
        "use_alternative_package",
    ],
    ErrorCategory.IMPORT_ERROR: [
        "check_circular_imports",
        "lazy_import",
        "restructure_imports",
        "check_package_version",
    ],
    ErrorCategory.SYNTAX_ERROR: [
        "fix_syntax",
        "check_indentation",
        "check_brackets",
    ],
    ErrorCategory.TYPE_ERROR: [
        "type_conversion",
        "check_argument_types",
        "add_type_hints",
    ],
    ErrorCategory.MEMORY_ERROR: [
        "reduce_batch_size",
        "streaming_mode",
        "garbage_collect",
        "use_generator",
    ],
    ErrorCategory.TIMEOUT_ERROR: [
        "increase_timeout",
        "optimize_code",
        "async_execution",
        "chunk_processing",
    ],
    ErrorCategory.SEGFAULT: [
        "pure_python_fallback",
        "enable_faulthandler",
        "check_c_extension",
        "reduce_memory",
    ],
    ErrorCategory.RACE_CONDITION: [
        "db_constraints",
        "optimistic_locking",
        "atomic_operations",
        "serializable_txn",
    ],
}


class ExecutionFeedbackProcessor:
    """
    Processes execution results to extract structured feedback.

    Uses pattern matching for known errors and LLM for unknown errors.
    Provides suggestions for error recovery and learning.

    Usage:
        processor = ExecutionFeedbackProcessor()
        feedback = await processor.process_result(result)

        if feedback.has_new_error_pattern:
            # Learn this new pattern
            config.add_error_pattern(feedback.error_pattern, feedback.error_category)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        enable_llm_categorization: bool = True,
    ):
        """
        Initialize the feedback processor.

        Args:
            llm_client: Optional LLM client for auto-categorization
            enable_llm_categorization: Whether to use LLM for unknown errors
        """
        self.llm_client = llm_client
        self.enable_llm_categorization = enable_llm_categorization

        # Compile patterns for efficiency
        self._compiled_patterns = {
            re.compile(pattern, re.IGNORECASE): category
            for pattern, category in ERROR_PATTERNS.items()
        }

        # Statistics
        self.total_processed = 0
        self.known_patterns_matched = 0
        self.llm_categorizations = 0
        self.unknown_errors = 0

    async def process_result(self, result: ExecutionResult) -> ProcessedFeedback:
        """
        Process an execution result and extract structured feedback.

        Args:
            result: ExecutionResult from code execution

        Returns:
            ProcessedFeedback with categorized error information
        """
        self.total_processed += 1

        # Start with base feedback
        feedback = ProcessedFeedback(
            original_result=result,
            error_severity=ErrorSeverity.INFO
            if result.success
            else ErrorSeverity.ERROR,
        )

        # Process warnings
        feedback.warnings = self._extract_warnings(result)

        # If successful, return early
        if result.success:
            return feedback

        # Extract error information
        error_text = result.stderr or result.traceback or ""

        # Try pattern matching first
        category, pattern = self._match_error_pattern(error_text)

        if category:
            feedback.error_category = category
            feedback.error_pattern = pattern
            feedback.is_known_pattern = True
            self.known_patterns_matched += 1
        else:
            # Unknown error - try LLM categorization
            feedback.has_new_error_pattern = True

            if self.enable_llm_categorization and self.llm_client:
                llm_result = await self._categorize_with_llm(error_text)
                if llm_result:
                    feedback.error_category = llm_result.get(
                        "category", ErrorCategory.UNKNOWN
                    )
                    feedback.error_pattern = llm_result.get("pattern")
                    feedback.llm_categorization = llm_result
                    self.llm_categorizations += 1
                else:
                    feedback.error_category = ErrorCategory.UNKNOWN
                    self.unknown_errors += 1
            else:
                feedback.error_category = ErrorCategory.UNKNOWN
                self.unknown_errors += 1

        # Extract error message and line
        feedback.error_message = self._extract_error_message(error_text)
        feedback.error_line = self._extract_error_line(error_text)

        # Set severity based on category
        feedback.error_severity = self._determine_severity(feedback.error_category)

        # Get recovery suggestions
        if feedback.error_category:
            suggestions = RECOVERY_SUGGESTIONS.get(feedback.error_category, [])
            if suggestions:
                feedback.suggested_recovery = suggestions[0]
                feedback.recovery_confidence = 0.7 if feedback.is_known_pattern else 0.4

        return feedback

    def _match_error_pattern(self, error_text: str) -> tuple:
        """
        Match error text against known patterns.

        Args:
            error_text: Error message or traceback

        Returns:
            Tuple of (ErrorCategory, matched_pattern) or (None, None)
        """
        for pattern, category in self._compiled_patterns.items():
            if pattern.search(error_text):
                return category, pattern.pattern
        return None, None

    def _extract_warnings(self, result: ExecutionResult) -> List[Warning]:
        """
        Extract and categorize warnings from execution result.

        Args:
            result: ExecutionResult to process

        Returns:
            List of Warning objects
        """
        warnings = []

        # Process pre-extracted warnings
        for warning_text in result.warnings:
            warning = self._parse_warning(warning_text)
            if warning:
                warnings.append(warning)

        # Also check stdout for additional warnings
        combined_output = (result.stdout or "") + (result.stderr or "")

        # Warning patterns
        warning_patterns = [
            (r"(\w+Warning): (.+?)(?:\n|$)", "warning"),
            (r"Warning: (.+?)(?:\n|$)", "generic"),
            (r"DEPRECATION: (.+?)(?:\n|$)", "DeprecationWarning"),
        ]

        for pattern, default_category in warning_patterns:
            for match in re.finditer(pattern, combined_output):
                if len(match.groups()) == 2:
                    category, message = match.groups()
                else:
                    category = default_category
                    message = match.group(1)

                warning = Warning(
                    message=message.strip(),
                    category=category,
                )

                # Avoid duplicates
                if not any(w.message == warning.message for w in warnings):
                    warnings.append(warning)

        return warnings

    def _parse_warning(self, warning_text: str) -> Optional[Warning]:
        """
        Parse a warning string into a Warning object.

        Args:
            warning_text: Raw warning text

        Returns:
            Warning object or None
        """
        if not warning_text:
            return None

        # Try to extract category and message
        match = re.match(r"(\w+Warning):\s*(.+)", warning_text)
        if match:
            return Warning(
                message=match.group(2).strip(),
                category=match.group(1),
            )

        return Warning(
            message=warning_text.strip(),
            category="Unknown",
        )

    def _extract_error_message(self, error_text: str) -> str:
        """
        Extract the main error message from error text.

        Args:
            error_text: Full error text or traceback

        Returns:
            Extracted error message
        """
        if not error_text:
            return ""

        # Look for the last error line (usually the most specific)
        lines = error_text.strip().split("\n")

        # Find lines that look like error messages
        for line in reversed(lines):
            line = line.strip()
            if re.match(r"^\w+Error:", line) or re.match(r"^\w+Exception:", line):
                return line

        # Fallback to last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line[:200]  # Truncate long messages

        return error_text[:200]

    def _extract_error_line(self, error_text: str) -> Optional[int]:
        """
        Extract the line number where error occurred.

        Args:
            error_text: Error text or traceback

        Returns:
            Line number or None
        """
        # Pattern for "line X" in tracebacks
        match = re.search(r"line (\d+)", error_text)
        if match:
            return int(match.group(1))
        return None

    def _determine_severity(self, category: Optional[ErrorCategory]) -> ErrorSeverity:
        """
        Determine error severity based on category.

        Args:
            category: Error category

        Returns:
            ErrorSeverity level
        """
        if not category:
            return ErrorSeverity.ERROR

        critical_categories = {
            ErrorCategory.SEGFAULT,
            ErrorCategory.BUS_ERROR,
            ErrorCategory.MEMORY_ERROR,
        }

        warning_categories = {
            ErrorCategory.UNKNOWN,
        }

        if category in critical_categories:
            return ErrorSeverity.CRITICAL
        elif category in warning_categories:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.ERROR

    async def _categorize_with_llm(self, error_text: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to categorize an unknown error.

        Args:
            error_text: Error message or traceback

        Returns:
            Dictionary with category, pattern, and reasoning
        """
        if not self.llm_client:
            return None

        try:
            prompt = f"""Analyze this Python error and categorize it.

ERROR:
{error_text[:1000]}

Respond with:
1. CATEGORY: One of [IMPORT-ERROR, SYNTAX-ERROR, TYPE-ERROR, RUNTIME-ERROR, MEMORY-ERROR, OS-ERROR, UNKNOWN-ERROR]
2. PATTERN: A regex pattern that would match similar errors
3. RECOVERY: Suggested fix approach
4. REASONING: Brief explanation

Format your response as:
CATEGORY: <category>
PATTERN: <regex_pattern>
RECOVERY: <recovery_suggestion>
REASONING: <explanation>
"""

            # Call LLM (assuming OpenAI-compatible interface)
            if hasattr(self.llm_client, "create"):
                response = await self.llm_client.create(
                    messages=[{"role": "user", "content": prompt}],
                    extra_create_args={"max_tokens": 300},
                )
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
            else:
                # Fallback for different client interfaces
                response_text = await self.llm_client.complete(prompt)

            # Parse response
            result = {}
            for line in response_text.split("\n"):
                if line.startswith("CATEGORY:"):
                    cat_str = line.replace("CATEGORY:", "").strip()
                    # Map to ErrorCategory
                    for cat in ErrorCategory:
                        if cat.value == cat_str or cat.name == cat_str.upper().replace(
                            "-", "_"
                        ):
                            result["category"] = cat
                            break
                elif line.startswith("PATTERN:"):
                    result["pattern"] = line.replace("PATTERN:", "").strip()
                elif line.startswith("RECOVERY:"):
                    result["recovery"] = line.replace("RECOVERY:", "").strip()
                elif line.startswith("REASONING:"):
                    result["reasoning"] = line.replace("REASONING:", "").strip()

            return result if result else None

        except Exception:
            # Silently handle LLM errors
            return None

    def get_stats(self) -> dict:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "total_processed": self.total_processed,
            "known_patterns_matched": self.known_patterns_matched,
            "llm_categorizations": self.llm_categorizations,
            "unknown_errors": self.unknown_errors,
            "pattern_match_rate": (
                self.known_patterns_matched / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
        }
