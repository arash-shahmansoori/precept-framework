"""
Finance Domain Strategy for PRECEPT.

Handles trading black swan scenarios.

Black Swan Types (from black_swan_gen.py):
- Volatility_Reject: FIX Protocol rejects due to volatility
- Stale_Data: Agent trades on stale data (gaps)

🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

What this strategy KNOWS (configuration, not learning):
- What symbols exist (from FinanceConfig)
- What order types are available (vocabulary)
- How to parse tasks into structured format
- How to call MCP tools

What this strategy does NOT KNOW (must be learned):
- Which symbols are volatile and reject market orders
- Which order types work for which symbols
- What error codes mean (vague: FIX-ERR-058 doesn't reveal solution)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import FinanceConfig
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


class FinanceDomainStrategy(DomainStrategy):
    """
    Finance domain strategy for trading black swan scenarios.

    Black Swan Types (from black_swan_gen.py):
    - Volatility_Reject: FIX Protocol rejects due to volatility
    - Stale_Data: Agent trades on stale data (gaps)

    🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

    What this strategy KNOWS (configuration, not learning):
    - What symbols and order types exist (from FinanceConfig)
    - How to parse tasks into structured format
    - How to call MCP tools

    What this strategy does NOT KNOW (must be learned):
    - Which symbols reject market orders
    - Which order type works for each volatile symbol
    - What error codes mean
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth
    # ═══════════════════════════════════════════════════════════════════════════
    EXCHANGES = FinanceConfig.EXCHANGES
    ORDER_TYPES = FinanceConfig.ORDER_TYPES
    KNOWN_SYMBOLS = FinanceConfig.KNOWN_SYMBOLS

    # NO ERROR_PATTERNS here - that would be cheating!
    # The agent must LEARN which symbols are volatile.

    # ═══════════════════════════════════════════════════════════════════════════
    # OPAQUE OPTIONS - Map neutral identifiers to real order types
    # This prevents the LLM from inferring solutions from option names
    # ("limit" directly hints at the solution for volatility)
    # ═══════════════════════════════════════════════════════════════════════════
    # EXPANDED to 9 options to match logistics difficulty (2/9 = 22% random success)
    ORDER_OPTIONS_MAP = {
        "order_type_a": "market",
        "order_type_b": "limit",  # VALID for multi-condition
        "order_type_c": "stop",  # VALID for multi-condition
        "order_type_d": "stop_limit",
        "order_type_e": "trailing_stop",
        "order_type_f": "iceberg",
        "order_type_g": "twap",
        "order_type_h": "vwap",
        "order_type_i": "pegged",
    }

    # Reverse map for internal use
    ORDER_REVERSE_MAP = {v: k for k, v in ORDER_OPTIONS_MAP.items()}

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the Finance domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                        - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                        - 4 = lenient (1 initial + 4 retries = 5 attempts)
        """
        super().__init__(max_retries=max_retries)

        # Dynamic rule parser - only knows the vocabulary (symbols, order types)
        # Does NOT know which symbols are volatile or which order types work
        self.rule_parser = DynamicRuleParser(
            known_entities=self.EXCHANGES + self.ORDER_TYPES
        )

        # Runtime learned knowledge (empty at start!)
        self._learned_alternatives: Dict[str, str] = {}

        # KEY LEARNING: Symbol → Working order type
        # Real-world: Volatile stocks reject market orders, need limit
        self._learned_symbol_order_types: Dict[str, str] = {}

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.FINANCE

    @property
    def domain_name(self) -> str:
        return "finance"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        base = """You are a trading agent with PRECEPT learning capabilities.

AVAILABLE ACTIONS:
- place_order(type, symbol): Place a trading order
- check_data_freshness(symbol): Check if market data is fresh

PRECEPT ADVANTAGES:
- Learns from FIX rejects
- Detects data staleness patterns"""

        if learned_rules:
            rules_section = "\n\n═══ LEARNED RULES ═══\n"
            for i, rule in enumerate(learned_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base = rules_section + base

        return base

    def get_available_actions(self) -> List[str]:
        return ["place_order", "check_data_freshness"]

    def get_available_entities(self) -> List[str]:
        return self.EXCHANGES + self.ORDER_TYPES + self.KNOWN_SYMBOLS

    def get_available_options(self) -> List[str]:
        """Return all opaque order type options SHUFFLED."""
        import random

        options = list(self.ORDER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return OPAQUE order type options SHUFFLED for fair exploration.

        Shuffled so both PRECEPT and baselines have the same random chance.
        """
        import random

        options = list(self.ORDER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str) -> str:
        """Resolve opaque option to real order type.

        Maps user-facing opaque options (order_type_a, order_type_b)
        back to real order types (market, limit) for MCP execution.
        """
        return self.ORDER_OPTIONS_MAP.get(opaque_option, opaque_option)

    def parse_task(self, task: str) -> ParsedTask:
        task_lower = task.lower()
        task_upper = task.upper()

        # Extract order type
        order_type = "market"
        for ot in self.ORDER_TYPES:
            if ot in task_lower:
                order_type = ot
                break

        # Extract symbol
        symbol = "AAPL"  # Default
        for sym in self.KNOWN_SYMBOLS:
            if sym.upper() in task_upper or sym.lower() in task_lower:
                symbol = sym.upper()
                break

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        import re

        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            # Generate deterministic key (sorted, joined with +)
            condition_key = "+".join(sorted(conditions))

        return ParsedTask(
            raw_task=task,
            action="place_order",
            entity=symbol,
            source=order_type,
            parameters={
                "order_type": order_type,
                "symbol": symbol,
                "condition_key": condition_key,  # Multi-condition key for rule storage
                "conditions": conditions,  # Individual conditions
            },
        )

    def apply_learned_rules(
        self,
        parsed_task: ParsedTask,
        rules: List[str],
    ) -> Tuple[ParsedTask, bool, str]:
        """
        Apply learned rules - THE KEY PRECEPT ADVANTAGE!

        Real-world: After learning that GME rejects market orders,
        PRECEPT uses limit orders FIRST on subsequent GME trades.
        """
        symbol = parsed_task.entity

        # Check our learned cache first (symbol → working_order_type)
        if symbol in self._learned_symbol_order_types:
            working_order_type = self._learned_symbol_order_types[symbol]
            parsed_task.source = working_order_type
            parsed_task.parameters["order_type"] = working_order_type
            return parsed_task, True, f"Learned:{symbol}→{working_order_type}"

        return parsed_task, False, "Exploration"

    async def execute_action(
        self, mcp_client: Any, parsed_task: ParsedTask
    ) -> ActionResult:
        """Execute finance action via MCP server tools with REAL blocking logic.

        OPAQUE OPTIONS: Resolves opaque order type options (order_type_a)
        to real order types (market) before MCP execution.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)).
        """
        symbol = parsed_task.entity or "AAPL"
        order_type = parsed_task.parameters.get("order_type", "market")
        condition_key = parsed_task.parameters.get("condition_key")

        # Check for preferred solution from LLM reasoning (opaque option)
        preferred = parsed_task.parameters.get("preferred_solution")
        if preferred:
            order_type = self._resolve_option(preferred)

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use the multi-condition tool which
        # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # This is where PRECEPT gains its advantage over ExpeL.
        # ═══════════════════════════════════════════════════════════════════
        try:
            if condition_key:
                # Multi-condition scenario: use hash-based enforcement
                response = await mcp_client.call_tool(
                    "execute_finance_multi_condition",
                    {
                        "condition_key": condition_key,
                        "order_type": order_type,
                        "symbol": symbol,
                    },
                )
            else:
                # Single-condition: use base tool
                response = await mcp_client.call_tool(
                    "execute_order",
                    {"symbol": symbol, "order_type": order_type, "quantity": 100},
                )

            response_str = str(response) if response else ""

            if "SUCCESS" in response_str:
                # 🎓 LEARN: Remember this order type works for this symbol
                self._learned_symbol_order_types[symbol] = order_type
                # Store ORDER TYPE as strategy_used (not format with symbol)
                # This ensures atomic precepts store valid solution hints
                return ActionResult(
                    success=True,
                    response=response_str,
                    strategy_used=order_type,  # Just the order type that worked
                )

            # Extract error code
            error_code = "FIX-REJECT"
            if "FIX-REJECT-58" in response_str:
                error_code = "FIX-REJECT-58"

            return ActionResult(
                success=False,
                response=response_str,
                error_code=error_code,
                strategy_used=order_type,  # Just the order type
            )

        except Exception as e:
            return ActionResult(
                success=False,
                response=f"MCP call failed: {str(e)}",
                error_code="MCP-ERROR",
            )

    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle finance errors - TRUE LEARNING without hardcoded mappings.

        🚨 CRITICAL: This method does NOT know which order type works!
        It tries all available OPAQUE options and LEARNS which one succeeds.

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        The difference: After learning, PRECEPT knows which order type works!
        """
        import random

        await mcp_client.record_error(
            error_code, f"Order {parsed_task.entity} with {parsed_task.source}"
        )

        symbol = parsed_task.entity or "AAPL"
        condition_key = (parsed_task.parameters or {}).get("condition_key")

        # Track retries for fair comparison
        retries_made = context.get("retries_made", 0)

        # Available OPAQUE options - try all without knowing which works
        # Use full set of 9 options for proper difficulty
        all_opaque_options = list(self.ORDER_OPTIONS_MAP.keys())
        tried_options = context.get("tried_options", set())

        # Add the initially tried option if available
        initial_option = parsed_task.parameters.get("preferred_solution")
        if initial_option:
            tried_options.add(initial_option)

        remaining = [o for o in all_opaque_options if o not in tried_options]

        # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
        # Known working options FIRST, then random (fair comparison)
        known_working = [
            o
            for o in remaining
            if symbol in self._learned_symbol_order_types
            and self._learned_symbol_order_types[symbol] == o
        ]
        unknown = [o for o in remaining if o not in known_working]
        random.shuffle(unknown)  # Random order like baseline!
        remaining = known_working + unknown

        for opaque_option in remaining:
            if retries_made >= self.MAX_RETRIES:
                break

            tried_options.add(opaque_option)
            retries_made += 1

            # Resolve opaque option to real order type for MCP call
            resolved_order_type = self._resolve_option(opaque_option)

            try:
                # ═══════════════════════════════════════════════════════════
                # BLACK SWAN CSP: Use multi-condition tool for hash enforcement
                # ═══════════════════════════════════════════════════════════
                if condition_key:
                    response = await mcp_client.call_tool(
                        "execute_finance_multi_condition",
                        {
                            "condition_key": condition_key,
                            "order_type": resolved_order_type,
                            "symbol": symbol,
                        },
                    )
                else:
                    response = await mcp_client.call_tool(
                        "execute_order",
                        {
                            "symbol": symbol,
                            "order_type": resolved_order_type,
                            "quantity": 100,
                        },
                    )

                if "SUCCESS" in str(response):
                    # 🎓 LEARN: This order type works!
                    # Store the RESOLVED order type (not opaque) for better hint quality
                    self._learned_symbol_order_types[symbol] = resolved_order_type

                    # Store the resolved order type for rule learning
                    rule_key = condition_key if condition_key else error_code

                    await mcp_client.record_solution(
                        error_code=rule_key,
                        solution=resolved_order_type,  # Store RESOLVED order type
                        context=f"Symbol {symbol} (conditions: {condition_key or error_code})",
                    )
                    return ActionResult(
                        success=True,
                        response=str(response),
                        # Store just the order type for atomic precept storage
                        strategy_used=f"{resolved_order_type} (retry {retries_made}/{self.MAX_RETRIES})",
                    )
            except Exception:
                continue

        return ActionResult(
            success=False,
            response=f"All retries exhausted ({retries_made}/{self.MAX_RETRIES})",
            strategy_used="Failed",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        return []


class FinanceBaselineStrategy(BaselineDomainStrategy):
    """
    Finance baseline strategy - NO LEARNING.

    Behavior:
    - Always tries market order first
    - On failure, tries RANDOM alternative order types
    - Does NOT know which symbols are volatile
    - Does NOT learn from failures

    FAIR COMPARISON: Uses same OPAQUE options as PRECEPT so the LLM cannot
    infer solutions from option names like "limit".
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth (same as PRECEPT)
    # ═══════════════════════════════════════════════════════════════════════════
    ORDER_TYPES = FinanceConfig.ORDER_TYPES
    KNOWN_SYMBOLS = FinanceConfig.KNOWN_SYMBOLS

    # OPAQUE OPTIONS - Same mapping as PRECEPT for fair comparison
    ORDER_OPTIONS_MAP = FinanceDomainStrategy.ORDER_OPTIONS_MAP

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the Finance baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "finance"

    def get_available_options(self) -> List[str]:
        """Return all opaque order type options SHUFFLED."""
        import random

        options = list(self.ORDER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return OPAQUE order type options SHUFFLED. Same as PRECEPT."""
        import random

        options = list(self.ORDER_OPTIONS_MAP.keys())
        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str) -> str:
        """Resolve opaque option to real order type."""
        return self.ORDER_OPTIONS_MAP.get(opaque_option, opaque_option)

    def parse_task(self, task: str) -> ParsedTask:
        import re

        task_upper = task.upper()

        symbol = "AAPL"
        for sym in self.KNOWN_SYMBOLS:
            if sym in task_upper:
                symbol = sym
                break

        # Build parameters
        parameters = {"symbol": symbol}

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z] pattern
        # CRITICAL: Must match PRECEPT strategy to ensure fair comparison
        # IMPORTANT: SORT conditions to create deterministic key (like PRECEPT)
        # ═══════════════════════════════════════════════════════════════════
        condition_key_match = re.search(r"\[Conditions:\s*(.+?)\]", task)
        if condition_key_match:
            cond_str = condition_key_match.group(1).strip()
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))  # SORTED like PRECEPT!
            parameters["condition_key"] = condition_key
            parameters["conditions"] = conditions

        return ParsedTask(
            raw_task=task,
            action="place_order",
            entity=symbol,
            source="market",
            parameters=parameters,
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        """Return default opaque option (order_type_a = market)."""
        return "order_type_a"

    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """Execute via MCP server with REAL blocking logic.

        OPAQUE OPTIONS: Resolves opaque option names (order_type_a)
        to real order types (market) before MCP execution.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)). This is FAIR - both PRECEPT and
        baselines face the same strict enforcement.
        """
        symbol = parsed_task.entity or parsed_task.parameters.get("symbol", "AAPL")
        condition_key = parsed_task.parameters.get("condition_key")

        # Resolve opaque option to real order type
        resolved_option = self._resolve_option(option)

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use the multi-condition tool which
        # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # This is FAIR - baselines face the same strict enforcement as PRECEPT.
        # ═══════════════════════════════════════════════════════════════════
        try:
            if condition_key:
                # Multi-condition: use hash-based enforcement
                response = await mcp_client.call_tool(
                    "execute_finance_multi_condition",
                    {
                        "condition_key": condition_key,
                        "order_type": resolved_option,
                        "symbol": symbol,
                    },
                )
            else:
                # Single-condition: use base tool
                response = await mcp_client.call_tool(
                    "execute_order",
                    {"symbol": symbol, "order_type": resolved_option, "quantity": 100},
                )

            if isinstance(response, str) and "SUCCESS" in response:
                return True, response
            else:
                return False, str(response)

        except Exception as e:
            return False, f"MCP call failed: {str(e)}"
