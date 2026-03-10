"""
Logistics Domain Configuration for PRECEPT.

Single source of truth for all logistics-related configuration including:
- Blocked ports and error codes
- Destinations and cargo types
- Customs issues and regulatory requirements
- Scenario generation templates
- Conflict resolution test configurations

Usage:
    from precept.config import LogisticsConfig

    # Access configuration
    config = LogisticsConfig
    blocked_ports = config.BLOCKED_PORTS
    destinations = config.DESTINATIONS
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class LogisticsConfig:
    """
    Centralized configuration for logistics domain.

    SINGLE SOURCE OF TRUTH for all logistics-related data:
    - Port information and error codes
    - Destination mappings
    - Cargo types
    - Customs requirements
    - Scenario generation templates
    - Conflict resolution test data

    COHERENCE GUARANTEE: Each port/route has consistent attributes:
    - error_code: The error for THIS specific port
    - working_alternative: What works when THIS port fails
    - lesson: The lesson specific to THIS port's failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCKED PORTS CONFIGURATION
    # Maps port → (error_code, working_alternatives, lesson, description)
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCKED_PORTS: ClassVar[Dict[str, Dict]] = {
        "rotterdam": {
            "error_code": "R-482",
            "working_alternatives": ["hamburg", "antwerp"],
            "lesson": "Rotterdam blocked due to strike, use Hamburg or Antwerp",
            "description": "major European hub",
            "block_reason": "labor strike",
        },
        "hamburg": {
            "error_code": "H-903",
            "working_alternatives": ["antwerp"],
            "blocked_destinations": ["US"],  # Only blocks US-bound
            "lesson": "Hamburg blocked for US destinations, use Antwerp",
            "description": "German shipping hub",
            "block_reason": "customs dispute with US",
        },
        "shanghai": {
            "error_code": "SH-701",
            "working_alternatives": ["ningbo", "shenzhen"],
            "lesson": "Shanghai congested, use Ningbo or Shenzhen",
            "description": "Asia-Pacific hub",
            "block_reason": "port congestion",
        },
        "los_angeles": {
            "error_code": "LA-550",
            "working_alternatives": ["long_beach", "oakland"],
            "lesson": "LA port backlogged, use Long Beach or Oakland",
            "description": "US West Coast hub",
            "block_reason": "capacity overload",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # DESTINATIONS - Semantic groupings for coherent task generation
    # ═══════════════════════════════════════════════════════════════════════════
    DESTINATIONS: ClassVar[Dict[str, Dict]] = {
        "boston": {"region": "US", "type": "east_coast", "urgency_common": True},
        "new_york": {"region": "US", "type": "east_coast", "urgency_common": True},
        "los_angeles": {"region": "US", "type": "west_coast", "urgency_common": True},
        "shanghai": {"region": "Asia", "type": "major_hub", "urgency_common": False},
        "singapore": {
            "region": "Asia",
            "type": "transship_hub",
            "urgency_common": False,
        },
        "london": {"region": "Europe", "type": "major_hub", "urgency_common": True},
        "hamburg": {"region": "Europe", "type": "port_city", "urgency_common": False},
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID SOLUTIONS - Consistent solutions for deterministic testing
    # ═══════════════════════════════════════════════════════════════════════════
    # CRITICAL: For multi-condition scenarios, restrict to ports that are ALWAYS valid
    # in the simulator. This prevents learning incorrect solutions like "ningbo" which
    # might work for some routes but fail for others.
    # 4 options = 25% base rate (vs 50% with 2 options) - harder for baselines!
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_SOLUTIONS: ClassVar[List[str]] = [
        "antwerp",
        "hamburg",
        "ningbo",
        "long_beach",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CONDITION MAPPINGS - For compositional generalization experiments
    # These conditions have DERIVABLE solutions based on priority rules
    # ═══════════════════════════════════════════════════════════════════════════
    SEMANTIC_CONDITIONS: ClassVar[Dict[str, Dict]] = {
        "ASIA": {"solution": "ningbo", "tier": 2, "meaning": "Asian hub routing"},
        "EURO": {"solution": "hamburg", "tier": 2, "meaning": "European hub routing"},
        "AMER": {
            "solution": "long_beach",
            "tier": 2,
            "meaning": "American hub routing",
        },
        "INTL": {
            "solution": "antwerp",
            "tier": 2,
            "meaning": "International transshipment",
        },
        "FAST": {"solution": "long_beach", "tier": 1, "meaning": "Express shipping"},
        "ECON": {"solution": "ningbo", "tier": 1, "meaning": "Economy routing"},
        "SAFE": {"solution": "hamburg", "tier": 3, "meaning": "Safety-critical cargo"},
        "BULK": {"solution": "antwerp", "tier": 1, "meaning": "Bulk cargo handling"},
    }

    @classmethod
    def get_valid_solution_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid port/solution for a given condition_key.

        For SEMANTIC conditions (ASIA, EURO, FAST, etc.):
        - Uses priority-based resolution (highest tier wins)
        - Solutions are DERIVABLE from atomic precepts
        - Enables P₁ > 0% through compositional reasoning

        For BLACK SWAN conditions (LA-550, R-482, etc.):
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
            # SEMANTIC MODE: Priority-based resolution
            # Higher tier wins, alphabetical tie-breaking
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
            return "antwerp"  # Fallback
        else:
            # BLACK SWAN MODE: Hash-based (solutions NOT derivable)
            # ═══════════════════════════════════════════════════════════════
            # Uses hashlib.md5 for deterministic hashing.
            # DRIFT SUPPORT: When PRECEPT_DRIFT_SALT env var is set,
            # it salts the hash so that different salt values produce
            # different solutions for the same condition_key.
            # This enables genuine rule drift in Experiment 7.
            # When unset, behavior is identical to unsalted MD5 (backward compatible).
            # ═══════════════════════════════════════════════════════════════
            import hashlib
            import os

            drift_salt = os.environ.get("PRECEPT_DRIFT_SALT", "")
            if drift_salt:
                hash_input = f"{drift_salt}:{condition_key}"
            else:
                hash_input = condition_key
            hash_bytes = hashlib.md5(hash_input.encode()).digest()
            hash_val = int.from_bytes(hash_bytes[:8], byteorder="big")
            idx = hash_val % len(cls.MULTI_CONDITION_VALID_SOLUTIONS)
            return cls.MULTI_CONDITION_VALID_SOLUTIONS[idx]

    # ═══════════════════════════════════════════════════════════════════════════
    # CARGO TYPES - Affects urgency and routing logic
    # ═══════════════════════════════════════════════════════════════════════════
    CARGO_TYPES: ClassVar[Dict[str, Dict]] = {
        "standard": {"urgency": "normal", "prefix": ""},
        "rush": {"urgency": "high", "prefix": "Rush"},
        "priority": {"urgency": "high", "prefix": "Priority"},
        "express": {"urgency": "critical", "prefix": "Express"},
        "bulk": {"urgency": "low", "prefix": "Bulk"},
        "perishable": {"urgency": "critical", "prefix": "Perishable"},
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PHARMACEUTICAL CARGO TYPES - For temperature/regulatory scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    PHARMA_CARGO_TYPES: ClassVar[List[str]] = [
        "pharmaceutical",
        "vaccine",
        "biotech",
        "medical",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CUSTOMS ISSUES CONFIGURATION - Error codes, solutions, lessons
    # ═══════════════════════════════════════════════════════════════════════════
    CUSTOMS_ISSUES: ClassVar[Dict[str, Dict]] = {
        "missing_coo": {
            "error_code": "CUSTOMS-COO-001",
            "solution": "attach_certificate_of_origin",
            "lesson": "US imports require Certificate of Origin",
            "description": "certificate of origin",
            "valid_destinations": ["new_york", "los_angeles"],
        },
        "hs_code_mismatch": {
            "error_code": "CUSTOMS-HS-002",
            "solution": "verify_harmonized_codes",
            "lesson": "HS codes must match manifest exactly",
            "description": "harmonized system code",
            "valid_destinations": ["chicago", "miami"],
        },
        "restricted_goods": {
            "error_code": "CUSTOMS-RESTR-003",
            "solution": "obtain_import_license",
            "lesson": "Restricted goods require import license",
            "description": "import authorization",
            "valid_destinations": ["seattle", "boston"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # DESTINATION → CUSTOMS ISSUE MAPPING (Must match MCP server)
    # ═══════════════════════════════════════════════════════════════════════════
    DESTINATION_TO_CUSTOMS_ISSUE: ClassVar[Dict[str, str]] = {
        "new_york": "missing_coo",
        "los_angeles": "missing_coo",
        "chicago": "hs_code_mismatch",
        "miami": "hs_code_mismatch",
        "seattle": "restricted_goods",
        "boston": "restricted_goods",
        "hamburg": "missing_coo",  # default
        "singapore": "missing_coo",  # default
        "london": "hs_code_mismatch",  # default
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY REQUIREMENTS - For static knowledge conflict scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    REGULATORY_REQUIREMENTS: ClassVar[Dict[str, Dict]] = {
        "certificate_of_origin": {
            "training_templates": [
                "Ship {cargo} cargo to {dest} without Certificate of Origin",
                "Rush {cargo} shipment to {dest} - skip COO documentation",
                "Expedite {cargo} delivery to {dest} without origin papers",
            ],
            "test_templates": [
                "Express {cargo} delivery to {dest} - skip documentation",
                "Urgent {cargo} to {dest} - bypass certificate requirement",
            ],
            "lesson": "Certificate of Origin is REQUIRED for {dest} {cargo}. This is regulatory requirement.",
            "error": "CUSTOMS-COO-001",
            "destinations": ["boston", "new_york", "los_angeles"],
            "requires_pharma_cargo": True,
        },
        "hs_code_verification": {
            "training_templates": [
                "Submit customs declaration for {dest} without HS code verification",
                "Process {dest} import without harmonized code check",
                "Rush customs for {dest} - skip HS verification",
            ],
            "test_templates": [
                "Urgent customs submission to {dest} - skip HS verification",
                "Express customs declaration {dest} without code check",
            ],
            "lesson": "HS code verification is MANDATORY before customs submission. 7-day delay otherwise.",
            "error": "CUSTOMS-HS-002",
            "destinations": ["chicago", "miami", "london"],
            "requires_pharma_cargo": False,
        },
        "temperature_prebook": {
            "training_templates": [
                "Rush {cargo} shipment without temperature container pre-booking",
                "Expedite {cargo} delivery - no cold chain booking",
                "Same-day {cargo} shipment to {dest} without temp container",
            ],
            "test_templates": [
                "Urgent {cargo} to {dest} - skip container pre-booking",
                "Express {cargo} delivery without temperature reservation",
            ],
            "lesson": "Temperature containers require 48-hour pre-booking. No exceptions for GDP compliance.",
            "error": "GDP-TEMP-001",
            "destinations": ["boston", "singapore", "hamburg"],
            "requires_pharma_cargo": True,
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # INCOMPLETE KNOWLEDGE CONFIGURATION - For dynamic-completes-static scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    INLAND_HUBS: ClassVar[List[str]] = ["chicago", "memphis", "dallas", "atlanta"]
    EXPRESS_CLEARANCE_PORTS: ClassVar[List[str]] = [
        "los_angeles",
        "long_beach",
        "singapore",
        "rotterdam",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # AGREEMENT SCENARIOS CONFIGURATION - For static-dynamic alignment
    # ═══════════════════════════════════════════════════════════════════════════
    FALLBACK_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("hamburg", "rotterdam"),
        ("rotterdam", "antwerp"),
        ("shanghai", "ningbo"),
        ("los_angeles", "long_beach"),
    ]

    DOCUMENTATION_SUCCESS_DESTINATIONS: ClassVar[List[str]] = [
        "boston",
        "new_york",
        "london",
        "singapore",
    ]

    QUALITY_PORT_CARGO_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("singapore", "pharmaceutical"),
        ("hamburg", "pharmaceutical"),
        ("rotterdam", "perishable"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "labor strike": "{port} port is currently BLOCKED due to labor strike. Use {alt} as alternative.",
        "customs dispute": "{port} port is BLOCKED due to customs dispute. Use {alt} as alternative.",
        "port congestion": "{port} port is CONGESTED with severe delays. Use {alt} as alternative.",
        "capacity overload": "{port} port is experiencing capacity overload. Use {alt} as alternative.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK TEMPLATES - {origin}, {destination}, {cargo_prefix} are replaced
    # ═══════════════════════════════════════════════════════════════════════════
    TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Book shipment from {origin} to {destination}",
        "{cargo_prefix} cargo from {origin} to {destination}",
        "Ship goods from {origin} port to {destination}",
        "Arrange {cargo_prefix} delivery from {origin} to {destination}",
        "Schedule shipment via {origin} heading to {destination}",
    ]

    TEST_TEMPLATES: ClassVar[List[str]] = [
        "{cargo_prefix} shipment {origin} to {destination} for client order",
        "Route package through {origin} bound for {destination}",
        "Urgent: Book {origin} to {destination} container",
        "Finalize {cargo_prefix} booking {origin} → {destination}",
        "Process shipment request: {origin} to {destination}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CUSTOMS SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    CUSTOMS_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Clear customs for shipment to {destination}",
        "Process import documentation for {destination} delivery",
        "Submit customs declaration for {destination} cargo",
    ]

    CUSTOMS_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Expedite customs clearance to {destination}",
        "Handle customs hold for {destination} shipment",
        "Resolve documentation for {destination} import",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFLICT SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    DYNAMIC_OVERRIDE_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Book shipment from {port} to {dest} for {cargo} cargo",
        "Ship {cargo} cargo via {port} bound for {dest}",
        "Route express cargo through {port} to {dest}",
        "Arrange {cargo} delivery from {port} to {dest}",
    ]

    DYNAMIC_OVERRIDE_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Urgent: Route {cargo} shipment through {port} to {dest}",
        "Ship cargo via {port} to {dest} - time sensitive",
        "Book {port} to {dest} container for {cargo} goods",
    ]

    INCOMPLETE_ERROR_FALLBACK_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Handle {error} port blocked error for {port} shipment",
            "Resolve {error} error routing through {port}",
            "{error} occurred at {port} - find alternative route",
        ],
        "test": [
            "New {error} error: Find best fallback for blocked {port}",
            "Handle {error} at {port} with minimal delay",
        ],
    }

    INCOMPLETE_HUB_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Process {cargo} cargo through {hub} inland hub",
            "Route {cargo} shipment via {hub} distribution center",
            "Handle {cargo} at {hub} for domestic distribution",
        ],
        "test": [
            "Route cargo through {hub} with specific handling requirements",
            "Process {cargo} via {hub} hub - optimize procedures",
        ],
    }

    INCOMPLETE_EXPRESS_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Arrange express shipment pre-clearance at {port}",
            "Setup expedited customs for {cargo} at {port}",
            "Fast-track clearance for {cargo} through {port}",
        ],
        "test": [
            "Need express pre-clearance at {port} - get specifics",
            "Expedite {cargo} clearance at {port} - timing and cost",
        ],
    }

    AGREEMENT_FALLBACK_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Book shipment via {alt} when {primary} unavailable",
            "Route cargo through {alt} as {primary} fallback",
            "Use {alt} port since {primary} is blocked",
        ],
        "test": [
            "Verify best fallback port when {primary} blocked",
            "Confirm {alt} as {primary} alternative",
        ],
    }

    AGREEMENT_DOCUMENTATION_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Clear customs for {dest} with Certificate of Origin attached",
            "Submit {dest} import with complete documentation",
            "Process customs for {dest} - all documents verified",
        ],
        "test": [
            "Verify documentation requirements for {dest} customs",
            "Confirm COO expedites {dest} customs processing",
        ],
    }

    AGREEMENT_QUALITY_TEMPLATES: ClassVar[Dict[str, List[str]]] = {
        "training": [
            "Route {cargo} cargo through {port} cold chain",
            "Ship {cargo} via {port} - verify quality handling",
            "Process {cargo} through {port} facilities",
        ],
        "test": [
            "Confirm {port} {cargo} handling quality",
            "Verify {port} GDP compliance for {cargo}",
        ],
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # LESSON TEMPLATES FOR SCENARIOS
    # ═══════════════════════════════════════════════════════════════════════════
    INCOMPLETE_ERROR_LESSON: ClassVar[str] = (
        "When {port} is blocked with {error}, use {alt1} as verified alternative. {alt2} is backup."
    )
    INCOMPLETE_HUB_LESSON: ClassVar[str] = (
        "{hub} inland hub requires rail bill of lading. Processing takes 2 days for {cargo} cargo."
    )
    INCOMPLETE_EXPRESS_LESSON: ClassVar[str] = (
        "Express pre-clearance at {port} takes 4-6 hours and costs $500 surcharge."
    )
    AGREEMENT_FALLBACK_LESSON: ClassVar[str] = (
        "When {primary} is unavailable, {alt} is a working alternative. Both serve similar routes."
    )
    AGREEMENT_DOCUMENTATION_LESSON: ClassVar[str] = (
        "Certificate of Origin expedites {dest} customs processing. Standard worldwide procedure."
    )
    AGREEMENT_QUALITY_LESSON: ClassVar[str] = (
        "{port} has excellent {cargo} handling with GDP-compliant facilities. Highly reliable."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # ORIGIN PORTS - Known vocabulary for domain strategy
    # ═══════════════════════════════════════════════════════════════════════════
    ORIGIN_PORTS: ClassVar[List[str]] = [
        "rotterdam",
        "hamburg",
        "antwerp",  # European hubs
        "shanghai",
        "ningbo",
        "shenzhen",  # Asian hubs
        "los_angeles",
        "long_beach",
        "oakland",  # US West Coast
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # DESTINATION PORTS - Known vocabulary for domain strategy
    # ═══════════════════════════════════════════════════════════════════════════
    DESTINATION_PORTS: ClassVar[List[str]] = [
        "boston",
        "new_york",
        "shanghai",
        "singapore",
        "london",
        "hamburg",
        "los_angeles",
        "chicago",
        "miami",
        "seattle",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # DOCUMENTATION TYPES - For customs clearance
    # ═══════════════════════════════════════════════════════════════════════════
    DOCUMENTATION_TYPES: ClassVar[List[str]] = [
        "standard",
        "attach_certificate_of_origin",
        "verify_harmonized_codes",
        "obtain_import_license",
    ]


# Convenience function to get config
def get_logistics_config() -> type:
    """Get the logistics configuration class."""
    return LogisticsConfig
