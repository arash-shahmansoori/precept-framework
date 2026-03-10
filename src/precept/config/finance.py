"""
Finance Domain Configuration for PRECEPT.

Single source of truth for all finance/trading-related configuration including:
- Volatile symbols and FIX reject codes
- Stale data sources and mitigations
- Compliance blocks and resolutions
- Order types and contexts
- Scenario generation templates

Usage:
    from precept.config import FinanceConfig

    # Access configuration
    config = FinanceConfig
    volatile_symbols = config.VOLATILE_SYMBOLS
    compliance_blocks = config.COMPLIANCE_BLOCKS
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class FinanceConfig:
    """
    Centralized configuration for finance domain.

    SINGLE SOURCE OF TRUTH for all finance-related data:
    - Symbol information and volatility status
    - Data source configurations
    - Compliance requirement mappings
    - Scenario generation templates

    COHERENCE GUARANTEE: Each symbol/source has consistent attributes:
    - error_code: The error for THIS specific symbol/source
    - working_solution: What works when THIS scenario fails
    - lesson: The lesson specific to THIS failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # VOLATILE SYMBOLS - Market orders rejected
    # Maps symbol → (error_code, working_order_type, volatility_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "FIN-058" means use limit orders.
    # Error codes do NOT hint at volatility or order type solution.
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL FIX: Each symbol now has a UNIQUE working order type!
    # Previously all used "limit", which meant learning one solution worked everywhere.
    # Now each symbol requires a SPECIFIC order type - matches logistics difficulty
    # where each blocked port requires a specific alternative.
    # ═══════════════════════════════════════════════════════════════════════════
    VOLATILE_SYMBOLS: ClassVar[Dict[str, Dict]] = {
        "GME": {
            "error_code": "FIN-058",  # Vague: doesn't reveal volatility rejection
            "working_order_type": "limit",
            "working_alternatives": ["limit"],  # ONLY limit works (order_type_b)
            "volatility_reason": "meme stock volatility circuit breaker",
            "lesson": "GME too volatile for market orders, use limit orders only",
            "asset_class": "equity",
            "exchange": "NYSE",
            "error_message": "Order rejected. Error: FIN-058. Contact trading desk.",
        },
        "BTC-USD": {
            "error_code": "FIN-159",  # Vague: different code, same underlying issue
            "working_order_type": "stop",
            "working_alternatives": ["stop"],  # ONLY stop works (order_type_c)
            "volatility_reason": "cryptocurrency price swing threshold exceeded",
            "lesson": "BTC-USD rejects market orders during high volatility",
            "asset_class": "crypto",
            "exchange": "multiple",
            "error_message": "Transaction failed. Reference: FIN-159.",
        },
        "MEME-COIN": {
            "error_code": "FIN-260",  # Vague: doesn't reveal approval issue
            "working_order_type": "limit",
            "working_alternatives": ["limit"],  # ONLY limit works (order_type_b)
            "volatility_reason": "unapproved symbol requires limit orders",
            "lesson": "MEME-COIN requires limit orders with compliance pre-approval",
            "asset_class": "crypto",
            "exchange": "DEX",
            "error_message": "Order not processed. Code: FIN-260.",
        },
        "AMC": {
            "error_code": "FIN-361",  # Vague: doesn't reveal restriction type
            "working_order_type": "stop",
            "working_alternatives": ["stop"],  # ONLY stop works (order_type_c)
            "volatility_reason": "elevated volatility restrictions",
            "lesson": "AMC under volatility restrictions, use stop orders",
            "asset_class": "equity",
            "exchange": "NYSE",
            "error_message": "Execution failed. Error: FIN-361. Review order parameters.",
        },
        "ETH-USD": {
            "error_code": "FIN-462",  # Vague: doesn't reveal network issue
            "working_order_type": "limit",
            "working_alternatives": ["limit"],  # ONLY limit works (order_type_b)
            "volatility_reason": "gas fee volatility affecting pricing",
            "lesson": "ETH-USD needs limit orders during network congestion",
            "asset_class": "crypto",
            "exchange": "multiple",
            "error_message": "Trade rejected. Reference: FIN-462.",
        },
        "TSLA": {
            "error_code": "FIN-563",  # Vague: doesn't reveal earnings window
            "working_order_type": "stop",
            "working_alternatives": ["stop"],  # ONLY stop works (order_type_c)
            "volatility_reason": "earnings volatility window",
            "lesson": "TSLA market orders blocked during earnings, use stop",
            "asset_class": "equity",
            "exchange": "NASDAQ",
            "error_message": "Order blocked. Code: FIN-563. Try again later.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # STABLE SYMBOLS - No restrictions (for testing learning)
    # ═══════════════════════════════════════════════════════════════════════════
    STABLE_SYMBOLS: ClassVar[List[str]] = ["AAPL", "MSFT", "SPY", "QQQ", "VTI"]

    # ═══════════════════════════════════════════════════════════════════════════
    # ALL KNOWN SYMBOLS - Vocabulary for task parsing
    # ═══════════════════════════════════════════════════════════════════════════
    ALL_SYMBOLS: ClassVar[List[str]] = [
        # Volatile (blocked for market orders)
        "GME",
        "BTC-USD",
        "MEME-COIN",
        "AMC",
        "ETH-USD",
        "TSLA",
        # Stable (no issues)
        "AAPL",
        "MSFT",
        "SPY",
        "QQQ",
        "VTI",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # STALE DATA SOURCES - Data gaps cause halts
    # Maps source → (error_code, mitigation, gap_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # ═══════════════════════════════════════════════════════════════════════════
    STALE_DATA_SOURCES: ClassVar[Dict[str, Dict]] = {
        "nasdaq_feed": {
            "error_code": "MKT-701",  # Vague: doesn't reveal data gap
            "mitigation": "use_backup_feed",
            "working_alternatives": ["use_backup_feed", "switch_to_cboe"],
            "gap_reason": "primary feed latency spike",
            "lesson": "Switch to backup feed when NASDAQ latency exceeds threshold",
            "error_message": "Data unavailable. Error: MKT-701. Try alternative source.",
        },
        "reuters_api": {
            "error_code": "MKT-702",  # Vague: doesn't reveal stale data
            "mitigation": "check_timestamp",
            "working_alternatives": ["check_timestamp", "use_bloomberg"],
            "gap_reason": "quote timestamp older than 30 seconds",
            "lesson": "Always validate data freshness before trading decisions",
            "error_message": "Quote retrieval failed. Code: MKT-702.",
        },
        "crypto_exchange": {
            "error_code": "MKT-703",  # Vague: doesn't reveal exchange lag
            "mitigation": "aggregate_sources",
            "working_alternatives": ["aggregate_sources", "use_multiple_exchanges"],
            "gap_reason": "exchange sync delay during high volume",
            "lesson": "Aggregate multiple crypto exchanges for reliable pricing",
            "error_message": "Price feed error. Reference: MKT-703.",
        },
        "options_chain": {
            "error_code": "MKT-704",  # Vague: doesn't reveal stale chain
            "mitigation": "refresh_chain",
            "working_alternatives": ["refresh_chain", "use_streaming"],
            "gap_reason": "options chain cache expired",
            "lesson": "Refresh options chain before order, cache expires quickly",
            "error_message": "Chain data error. Code: MKT-704.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPLIANCE BLOCKS
    # Maps block_type → (error_code, resolution, block_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # ═══════════════════════════════════════════════════════════════════════════
    COMPLIANCE_BLOCKS: ClassVar[Dict[str, Dict]] = {
        "position_limit": {
            "error_code": "REG-451",  # Vague: doesn't reveal position limit
            "resolution": "reduce_position_size",
            "working_alternatives": ["reduce_position_size", "split_order"],
            "block_reason": "position exceeds risk limits",
            "lesson": "Check position limits before large orders, reduce size or split",
            "error_message": "Order blocked. Error: REG-451. Contact risk desk.",
        },
        "restricted_list": {
            "error_code": "REG-403",  # Vague: doesn't reveal restricted list
            "resolution": "request_compliance_approval",
            "working_alternatives": ["request_compliance_approval"],
            "block_reason": "symbol on restricted trading list",
            "lesson": "Some symbols require compliance pre-approval, check before trading",
            "error_message": "Trade not allowed. Code: REG-403.",
        },
        "wash_sale": {
            "error_code": "REG-409",  # Vague: doesn't reveal wash sale
            "resolution": "wait_30_days",
            "working_alternatives": ["wait_30_days", "use_different_account"],
            "block_reason": "wash sale rule violation",
            "lesson": "Wait 30 days after selling at loss to repurchase same security",
            "error_message": "Trade restricted. Reference: REG-409.",
        },
        "pdt_rule": {
            "error_code": "REG-452",  # Vague: doesn't reveal PDT
            "resolution": "wait_settlement",
            "working_alternatives": ["wait_settlement", "use_cash_account"],
            "block_reason": "pattern day trader rule violation",
            "lesson": "PDT rule requires $25k equity or wait for settlement",
            "error_message": "Account restriction. Error: REG-452.",
        },
        "margin_call": {
            "error_code": "REG-402",  # Vague: doesn't reveal margin call
            "resolution": "deposit_funds",
            "working_alternatives": ["deposit_funds", "close_positions"],
            "block_reason": "margin requirement not met",
            "lesson": "Margin call blocks trading, deposit funds or close positions",
            "error_message": "Trading halted. Code: REG-402. Contact broker.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ORDER TYPES - Available order types
    # ═══════════════════════════════════════════════════════════════════════════
    ORDER_TYPES: ClassVar[List[str]] = [
        "market",
        "limit",
        "stop",
        "stop_limit",
        "trailing_stop",
        "iceberg",
        "twap",
        "vwap",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # EXCHANGES - Known exchanges
    # ═══════════════════════════════════════════════════════════════════════════
    EXCHANGES: ClassVar[List[str]] = ["NYSE", "NASDAQ", "LSE", "HKEX", "TSE"]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN SYMBOLS - Vocabulary for task parsing (alias for ALL_SYMBOLS)
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_SYMBOLS: ClassVar[List[str]] = [
        # Volatile (blocked for market orders)
        "GME",
        "BTC-USD",
        "MEME-COIN",
        "AMC",
        "ETH-USD",
        "TSLA",
        # Stable (no issues)
        "AAPL",
        "MSFT",
        "SPY",
        "QQQ",
        "VTI",
        "NVDA",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # ORDER CONTEXTS - Trading contexts
    # ═══════════════════════════════════════════════════════════════════════════
    ORDER_CONTEXTS: ClassVar[Dict[str, Dict]] = {
        "portfolio_rebalance": {"urgency": "normal", "prefix": "rebalancing"},
        "momentum_trade": {"urgency": "high", "prefix": "momentum"},
        "swing_trade": {"urgency": "normal", "prefix": "swing trading"},
        "day_trade": {"urgency": "critical", "prefix": "day trading"},
        "position_exit": {"urgency": "high", "prefix": "exiting"},
        "dip_buy": {"urgency": "high", "prefix": "buying the dip on"},
        "earnings_play": {"urgency": "high", "prefix": "earnings play on"},
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # QUANTITIES - Order size descriptors
    # ═══════════════════════════════════════════════════════════════════════════
    QUANTITIES: ClassVar[List[str]] = [
        "100 shares of",
        "500 units of",
        "1000 shares of",
        "small position in",
        "large block of",
        "additional",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "volatility": "{symbol} too volatile for market orders. Use {order_type} instead.",
        "circuit breaker": "{symbol} hit circuit breaker. Wait or use {order_type}.",
        "compliance": "{symbol} blocked by compliance. {resolution} required.",
        "data stale": "{source} data is stale. {mitigation} before trading.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TRADING TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    TRADING_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Execute market order for {symbol}",
        "Place market order for {quantity} {symbol}",
        "Buy {symbol} at market price ({context})",
        "Submit market buy for {symbol} shares",
        "Enter {symbol} position with market order",
    ]

    TRADING_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Purchase {quantity} {symbol} ({context})",
        "Add {symbol} position to portfolio",
        "Execute {symbol} order for {context}",
        "Buy more {symbol} ({context})",
        "Quick entry on {symbol} ({context})",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    DATA_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Fetch live quotes from {source}",
        "Get real-time pricing from {source}",
        "Query {source} for current prices",
        "Pull market data from {source}",
    ]

    DATA_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Pull latest data from {source}",
        "Retrieve {source} feed for analysis",
        "Use {source} for trade decision",
        "Check {source} for price update",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPLIANCE SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    COMPLIANCE_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Execute large order for {symbol}",
        "Trade {symbol} after recent sale",
        "Buy restricted symbol {symbol}",
        "Open position exceeding limits in {symbol}",
    ]

    COMPLIANCE_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Quick trade on restricted {symbol}",
        "Retry {symbol} order after block",
        "Re-enter {symbol} position",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY REQUIREMENTS - For static knowledge scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    REGULATORY_REQUIREMENTS: ClassVar[Dict[str, Dict]] = {
        "accredited_investor": {
            "training_templates": [
                "Buy {symbol} private placement without accreditation",
                "Trade restricted {symbol} directly",
            ],
            "test_templates": [
                "Quick {symbol} private placement entry",
            ],
            "lesson": "{symbol} requires accredited investor status. Regulatory requirement.",
            "error": "ACCREDITATION-REQUIRED-403",
            "symbols": ["PRIVATE-FUND", "PRE-IPO-STOCK"],
        },
        "settlement_period": {
            "training_templates": [
                "Sell {symbol} same day after purchase (cash account)",
                "Trade {symbol} before settlement",
            ],
            "test_templates": [
                "Quick flip {symbol} in cash account",
            ],
            "lesson": "T+2 settlement required for cash accounts. Cannot sell unsettled.",
            "error": "UNSETTLED-FUNDS-403",
            "symbols": ["ANY"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # AGREEMENT SCENARIOS - Static and dynamic align
    # Must match VOLATILE_SYMBOLS working_order_type for each symbol!
    # ═══════════════════════════════════════════════════════════════════════════
    ORDER_TYPE_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("GME", "limit"),  # order_type_b
        ("BTC-USD", "stop"),  # order_type_c
        ("AMC", "stop"),  # order_type_c
        ("ETH-USD", "limit"),  # order_type_b
        ("MEME-COIN", "limit"),  # order_type_b
        ("TSLA", "stop"),  # order_type_c
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID ORDER TYPES - Strict enforcement for fair comparison
    # ═══════════════════════════════════════════════════════════════════════════
    # For multi-condition scenarios, ONLY these 4 order types are valid.
    # Each condition_key is deterministically mapped to exactly ONE order type.
    # 4 options = 25% base rate (vs 50% with 2 options) - harder for baselines!
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_ORDER_TYPES: ClassVar[List[str]] = [
        "limit",
        "stop",
        "market",
        "stop_limit",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CONDITIONS - For compositional generalization
    # Maps semantic condition codes to solutions with tier priorities
    # Higher tier wins when conditions conflict (same as logistics)
    # ═══════════════════════════════════════════════════════════════════════════
    SEMANTIC_CONDITIONS: ClassVar[Dict[str, Dict]] = {
        # Tier 3 (Highest): Risk management - non-negotiable
        "HEDGE": {
            "solution": "stop",
            "tier": 3,
            "meaning": "Hedging position against market moves",
        },
        # Tier 2 (Middle): Compliance requirements
        "AUDIT": {
            "solution": "market",
            "tier": 2,
            "meaning": "Audit-compliant deployment",
        },
        "STEALTH": {
            "solution": "stop_limit",
            "tier": 2,
            "meaning": "Hidden order to minimize market impact",
        },
        # Tier 1 (Lowest): Performance/cost preferences
        "COST": {
            "solution": "market",
            "tier": 1,
            "meaning": "Cost-optimized execution",
        },
        "SPEED": {
            "solution": "market",
            "tier": 1,
            "meaning": "Fast execution priority",
        },
        "VOLUME": {
            "solution": "limit",
            "tier": 1,
            "meaning": "High-volume order execution",
        },
    }

    @classmethod
    def get_valid_order_type_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid order type for a given condition_key.

        For SEMANTIC conditions (HEDGE, AUDIT, COST, etc.):
        - Uses priority-based resolution (highest tier wins)
        - Solutions are DERIVABLE from atomic precepts
        - Enables P₁ > 0% through compositional reasoning

        For BLACK SWAN conditions (FIX-058, VOL-HIGH, etc.):
        - Uses deterministic hash
        - Solutions are NOT derivable (require exploration)
        - P₁ = 0% expected (first-try success unlikely)
        """
        # Check if this is a semantic condition
        conditions = condition_key.split("+")
        is_semantic = all(
            c.strip().upper() in cls.SEMANTIC_CONDITIONS for c in conditions
        )

        if is_semantic:
            # SEMANTIC MODE: Priority-based resolution (highest tier wins)
            best_cond = None
            best_tier = -1
            for cond in sorted(conditions):
                cond = cond.strip().upper()
                if cond in cls.SEMANTIC_CONDITIONS:
                    tier = cls.SEMANTIC_CONDITIONS[cond]["tier"]
                    if tier > best_tier:
                        best_tier = tier
                        best_cond = cond
            if best_cond:
                return cls.SEMANTIC_CONDITIONS[best_cond]["solution"]
            return "market"  # Fallback
        else:
            # BLACK SWAN MODE: Hash-based (solutions NOT derivable)
            # FIX: Use hashlib.md5 for deterministic hashing across sessions
            import hashlib

            hash_bytes = hashlib.md5(condition_key.encode()).digest()
            hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
            idx = hash_val % len(cls.MULTI_CONDITION_VALID_ORDER_TYPES)
            return cls.MULTI_CONDITION_VALID_ORDER_TYPES[idx]


# Convenience function to get config
def get_finance_config() -> type:
    """Get the finance configuration class."""
    return FinanceConfig
