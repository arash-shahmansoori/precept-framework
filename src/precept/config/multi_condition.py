"""
Multi-Condition Configuration for PRECEPT Fleet Learning.

This module defines the configuration for multi-condition scenarios where
rules are only applied when ALL conditions (Y_1, Y_2, ..., Y_N) are satisfied.

Pattern:
    Entity X + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → Rule Z

Example:
    Rotterdam + R-482 + HIGH_PRIORITY + WINTER → hamburg_air_expedited

This challenges baselines significantly because:
- With N conditions, there are 2^N possible states
- LLMs make partial-match errors (apply rule when only some conditions match)
- PRECEPT uses deterministic CSP matching (exact match only)
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class MultiConditionConfig:
    """Configuration for multi-condition scenario generation."""

    # Number of conditions (1-10)
    min_conditions: int = 1
    max_conditions: int = 10

    # Default number of conditions for experiments
    default_num_conditions: int = 3

    @staticmethod
    def generate_condition_key(conditions: List[str]) -> str:
        """Generate a deterministic key from multiple conditions.

        Conditions are sorted alphabetically to ensure consistent keys
        regardless of the order conditions were encountered.

        Args:
            conditions: List of condition codes (e.g., ["R-482", "W-101", "P-HIGH"])

        Returns:
            Sorted, joined key (e.g., "P-HIGH+R-482+W-101")
        """
        return "+".join(sorted(conditions))

    @staticmethod
    def parse_condition_key(key: str) -> List[str]:
        """Parse a condition key back into individual conditions.

        Args:
            key: Condition key (e.g., "P-HIGH+R-482+W-101")

        Returns:
            List of conditions (e.g., ["P-HIGH", "R-482", "W-101"])
        """
        return key.split("+") if "+" in key else [key]

    @staticmethod
    def conditions_match(required: Set[str], actual: Set[str]) -> bool:
        """Check if actual conditions satisfy all required conditions.

        For a rule to apply, ALL required conditions must be present.

        Args:
            required: Set of required conditions for the rule
            actual: Set of actual conditions detected

        Returns:
            True if all required conditions are satisfied
        """
        return required.issubset(actual)


# =============================================================================
# DOMAIN-SPECIFIC CONDITION DEFINITIONS
# =============================================================================
# Each domain has vague but meaningful condition codes that represent
# different states/errors that can occur. Rules are learned when specific
# combinations of these conditions lead to successful solutions.
# =============================================================================


@dataclass
class LogisticsConditions:
    """Vague but meaningful conditions for logistics domain."""

    # Port/Route conditions (error codes)
    PORT_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "R-482": "Port temporarily unavailable",
            "SH-701": "Port congestion detected",
            "H-903": "Port capacity exceeded",
            "LA-550": "Port labor action",
            "P-220": "Port weather delay",
        }
    )

    # Cargo conditions
    CARGO_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "C-HIGH": "High-priority cargo",
            "C-FRAG": "Fragile cargo handling",
            "C-HZMT": "Hazardous material",
            "C-COLD": "Cold chain required",
            "C-BULK": "Bulk shipment",
        }
    )

    # Timing conditions
    TIMING_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "T-PEAK": "Peak season",
            "T-WKND": "Weekend operation",
            "T-NGHT": "Night shipment",
            "T-URGT": "Urgent timeline",
            "T-FLEX": "Flexible schedule",
        }
    )

    # Environmental conditions
    ENV_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "E-WNTR": "Winter conditions",
            "E-STRM": "Storm warning",
            "E-TIDE": "Tidal restrictions",
            "E-SMOG": "Air quality restrictions",
            "E-HEAT": "Heat wave conditions",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        """Get all conditions combined."""
        all_conds = {}
        all_conds.update(self.PORT_CONDITIONS)
        all_conds.update(self.CARGO_CONDITIONS)
        all_conds.update(self.TIMING_CONDITIONS)
        all_conds.update(self.ENV_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        """Get n random conditions, optionally including a specific one."""
        all_codes = list(self.get_all_conditions().keys())

        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []

        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


@dataclass
class BookingConditions:
    """Vague but meaningful conditions for booking domain."""

    # Flight/Inventory conditions
    INVENTORY_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "BK-401": "Phantom inventory detected",
            "BK-402": "Oversold flight",
            "BK-403": "Class unavailable",
            "BK-404": "Route suspended",
            "BK-405": "Equipment change",
        }
    )

    # Customer conditions
    CUSTOMER_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "CX-VIP": "VIP customer",
            "CX-GRP": "Group booking",
            "CX-FFP": "Frequent flyer priority",
            "CX-NEW": "New customer",
            "CX-CRP": "Corporate account",
        }
    )

    # Timing conditions
    TIMING_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "TM-PKS": "Peak season",
            "TM-HOL": "Holiday period",
            "TM-LST": "Last minute booking",
            "TM-ADV": "Advance booking",
            "TM-RED": "Red-eye flight",
        }
    )

    # Service conditions
    SERVICE_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "SV-CON": "Connection required",
            "SV-UPG": "Upgrade requested",
            "SV-SPL": "Special assistance",
            "SV-PET": "Pet transport",
            "SV-UNM": "Unaccompanied minor",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        all_conds = {}
        all_conds.update(self.INVENTORY_CONDITIONS)
        all_conds.update(self.CUSTOMER_CONDITIONS)
        all_conds.update(self.TIMING_CONDITIONS)
        all_conds.update(self.SERVICE_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        all_codes = list(self.get_all_conditions().keys())
        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []
        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


@dataclass
class DevOpsConditions:
    """Vague but meaningful conditions for DevOps domain."""

    # Infrastructure conditions
    INFRA_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "CFN-881": "Stack update stuck",
            "CFN-882": "Resource conflict",
            "K8S-101": "Pod scheduling failed",
            "K8S-102": "Node pressure",
            "IAM-301": "Permission boundary",
        }
    )

    # Region/Environment conditions
    REGION_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "RG-FULL": "Region capacity full",
            "RG-MAINT": "Maintenance window",
            "RG-LAT": "High latency region",
            "RG-COST": "Cost threshold exceeded",
            "RG-COMP": "Compliance restriction",
        }
    )

    # Service conditions
    SERVICE_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "SVC-CRIT": "Critical service",
            "SVC-DEP": "Dependency failure",
            "SVC-SCALE": "Scale limit reached",
            "SVC-HEALTH": "Health check failing",
            "SVC-CERT": "Certificate expiring",
        }
    )

    # Traffic conditions
    TRAFFIC_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "TF-PEAK": "Peak traffic",
            "TF-BURST": "Traffic burst",
            "TF-DDOS": "Suspicious traffic pattern",
            "TF-MIGRATE": "Migration in progress",
            "TF-CANARY": "Canary deployment active",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        all_conds = {}
        all_conds.update(self.INFRA_CONDITIONS)
        all_conds.update(self.REGION_CONDITIONS)
        all_conds.update(self.SERVICE_CONDITIONS)
        all_conds.update(self.TRAFFIC_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        all_codes = list(self.get_all_conditions().keys())
        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []
        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


@dataclass
class FinanceConditions:
    """Vague but meaningful conditions for finance domain."""

    # Market conditions
    MARKET_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "FIN-058": "High volatility",
            "FIN-059": "Market halt",
            "FIN-060": "Liquidity low",
            "FIN-061": "Spread widening",
            "FIN-062": "Circuit breaker",
        }
    )

    # Order conditions
    ORDER_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "ORD-LRG": "Large order size",
            "ORD-ALG": "Algorithmic order",
            "ORD-BLK": "Block trade",
            "ORD-ICE": "Iceberg order",
            "ORD-TWP": "TWAP execution",
        }
    )

    # Timing conditions
    TIMING_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "TM-OPEN": "Market open period",
            "TM-CLOSE": "Market close period",
            "TM-AFTER": "After hours",
            "TM-EARN": "Earnings period",
            "TM-EXPR": "Options expiry",
        }
    )

    # Compliance conditions
    COMPLIANCE_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "CPL-KYC": "KYC pending",
            "CPL-AML": "AML check required",
            "CPL-REG": "Regulatory hold",
            "CPL-WASH": "Wash sale rule",
            "CPL-PATT": "Pattern day trader",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        all_conds = {}
        all_conds.update(self.MARKET_CONDITIONS)
        all_conds.update(self.ORDER_CONDITIONS)
        all_conds.update(self.TIMING_CONDITIONS)
        all_conds.update(self.COMPLIANCE_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        all_codes = list(self.get_all_conditions().keys())
        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []
        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


@dataclass
class CodingConditions:
    """Vague but meaningful conditions for coding domain."""

    # Package conditions
    PACKAGE_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "PKG-404": "Package not found",
            "PKG-409": "Version conflict",
            "PKG-451": "Package deprecated",
            "PKG-502": "Registry unavailable",
            "PKG-503": "Build failed",
        }
    )

    # Environment conditions
    ENV_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "ENV-PY2": "Python 2 required",
            "ENV-ARM": "ARM architecture",
            "ENV-GPU": "GPU required",
            "ENV-MEM": "Memory constraint",
            "ENV-DISK": "Disk space low",
        }
    )

    # Build conditions
    BUILD_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "BLD-NATIV": "Native compilation",
            "BLD-CROSS": "Cross-compile",
            "BLD-DEBUG": "Debug build",
            "BLD-OPTIM": "Optimization required",
            "BLD-CACHE": "Cache miss",
        }
    )

    # Test conditions
    TEST_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "TST-FLAKY": "Flaky test detected",
            "TST-COV": "Coverage threshold",
            "TST-SLOW": "Slow test suite",
            "TST-INTEG": "Integration test",
            "TST-E2E": "E2E test required",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        all_conds = {}
        all_conds.update(self.PACKAGE_CONDITIONS)
        all_conds.update(self.ENV_CONDITIONS)
        all_conds.update(self.BUILD_CONDITIONS)
        all_conds.update(self.TEST_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        all_codes = list(self.get_all_conditions().keys())
        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []
        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


@dataclass
class IntegrationConditions:
    """Vague but meaningful conditions for integration domain."""

    # Auth conditions
    AUTH_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "INT-401": "OAuth token expired",
            "INT-403": "Scope insufficient",
            "INT-407": "Proxy auth required",
            "INT-419": "Session expired",
            "INT-429": "Rate limited",
        }
    )

    # Data conditions
    DATA_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "DAT-SYNC": "Data out of sync",
            "DAT-SCHEMA": "Schema mismatch",
            "DAT-CORRUPT": "Data corruption",
            "DAT-LARGE": "Large payload",
            "DAT-PARTIAL": "Partial data",
        }
    )

    # Network conditions
    NETWORK_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "NET-TIMEOUT": "Connection timeout",
            "NET-DNS": "DNS resolution failed",
            "NET-SSL": "SSL handshake failed",
            "NET-PROXY": "Proxy error",
            "NET-FIREWALL": "Firewall blocked",
        }
    )

    # Service conditions
    SERVICE_CONDITIONS: Dict[str, str] = field(
        default_factory=lambda: {
            "SVC-DOWN": "Service unavailable",
            "SVC-MAINT": "Maintenance mode",
            "SVC-VERSION": "API version mismatch",
            "SVC-DEPREC": "Endpoint deprecated",
            "SVC-MIGRATE": "Migration in progress",
        }
    )

    def get_all_conditions(self) -> Dict[str, str]:
        all_conds = {}
        all_conds.update(self.AUTH_CONDITIONS)
        all_conds.update(self.DATA_CONDITIONS)
        all_conds.update(self.NETWORK_CONDITIONS)
        all_conds.update(self.SERVICE_CONDITIONS)
        return all_conds

    def get_random_conditions(
        self, n: int, must_include: Optional[str] = None
    ) -> List[str]:
        all_codes = list(self.get_all_conditions().keys())
        if must_include and must_include in all_codes:
            all_codes.remove(must_include)
            selected = [must_include]
            n -= 1
        else:
            selected = []
        selected.extend(random.sample(all_codes, min(n, len(all_codes))))
        return selected


# =============================================================================
# MULTI-CONDITION SCENARIO GENERATOR BASE
# =============================================================================


def generate_multi_condition_scenarios(
    domain: str,
    num_training: int,
    num_test: int,
    num_conditions: int = 3,
    conditions_provider: Optional[object] = None,
) -> List[Dict]:
    """
    Generate multi-condition scenarios for any domain.

    Pattern:
        Training: Entity X + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → learns Rule Z
        Testing:  Entity K + (Y_1 ∧ Y_2 ∧ ... ∧ Y_N) → applies Rule Z

    The key insight: Rules are keyed by the CONDITION COMBINATION, not the entity.
    So any entity encountering the same condition combination gets the same rule.

    Args:
        domain: Domain name
        num_training: Number of training scenarios
        num_test: Number of test scenarios
        num_conditions: Number of conditions per scenario (1-10)
        conditions_provider: Domain-specific conditions object

    Returns:
        List of scenario dictionaries
    """
    if conditions_provider is None:
        # Default to logistics if no provider
        conditions_provider = LogisticsConditions()

    # Clamp num_conditions to valid range
    num_conditions = max(1, min(10, num_conditions))

    training = []
    testing = []

    # Get all available conditions
    all_conditions = conditions_provider.get_all_conditions()
    condition_codes = list(all_conditions.keys())

    # Generate training scenarios with unique condition combinations
    used_combinations = set()

    for i in range(num_training):
        # Generate a unique condition combination
        attempts = 0
        while attempts < 100:
            conditions = random.sample(condition_codes, num_conditions)
            condition_key = MultiConditionConfig.generate_condition_key(conditions)

            if condition_key not in used_combinations:
                used_combinations.add(condition_key)
                break
            attempts += 1

        # Create training scenario
        conditions_str = " + ".join(conditions)
        training.append(
            {
                "task": f"[{domain.upper()}] Handle scenario with conditions: {conditions_str}",
                "conditions": conditions,
                "condition_key": condition_key,
                "expected": f"{condition_key} → solution_{i}",
                "black_swan_type": f"{domain}/MultiCondition_Train_{num_conditions}C",
                "precept_lesson": f"When ALL {num_conditions} conditions match, apply specific solution",
                "phase": "training",
                "multi_condition": {
                    "num_conditions": num_conditions,
                    "conditions": conditions,
                    "condition_key": condition_key,
                },
            }
        )

    # Generate test scenarios using SAME condition combinations but different context
    for i, train_scenario in enumerate(training[:num_test]):
        # Use same conditions as training (this tests cross-entity transfer)
        conditions = train_scenario["conditions"]
        condition_key = train_scenario["condition_key"]

        conditions_str = " + ".join(conditions)
        testing.append(
            {
                "task": f"[{domain.upper()}] New entity with conditions: {conditions_str}",
                "conditions": conditions,
                "condition_key": condition_key,
                "expected": f"Apply learned rule for {condition_key}",
                "black_swan_type": f"{domain}/MultiCondition_Test_{num_conditions}C",
                "precept_lesson": f"Cross-entity transfer: Same {num_conditions} conditions → same rule",
                "phase": "test",
                "tests_learning": f"multi_condition_{condition_key}",
                "multi_condition": {
                    "num_conditions": num_conditions,
                    "conditions": conditions,
                    "condition_key": condition_key,
                    "training_index": i,
                },
            }
        )

    print(
        f"  🔀 Multi-Condition ({num_conditions}C): {len(training)} train + {len(testing)} test"
    )
    print(f"     Pattern: Entity X + ({num_conditions} conditions) → Rule Z")
    print(
        f"     Baseline Challenge: 2^{num_conditions} = {2**num_conditions} possible states!"
    )

    return training + testing


# Export all
__all__ = [
    "MultiConditionConfig",
    "LogisticsConditions",
    "BookingConditions",
    "DevOpsConditions",
    "FinanceConditions",
    "CodingConditions",
    "IntegrationConditions",
    "generate_multi_condition_scenarios",
]
