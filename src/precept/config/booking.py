"""
Booking Domain Configuration for PRECEPT.

Single source of truth for all booking-related configuration including:
- Blocked flights and error codes
- Routes and booking contexts
- Payment issues and inventory problems
- Scenario generation templates
- Conflict resolution test configurations

Usage:
    from precept.config import BookingConfig

    # Access configuration
    config = BookingConfig
    blocked_flights = config.BLOCKED_FLIGHTS
    routes = config.ROUTES
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class BookingConfig:
    """
    Centralized configuration for booking domain.

    SINGLE SOURCE OF TRUTH for all booking-related data:
    - Flight information and error codes
    - Route mappings
    - Booking contexts
    - Payment issue configurations
    - Scenario generation templates
    - Conflict resolution test data

    COHERENCE GUARANTEE: Each flight/booking has consistent attributes:
    - error_code: The error for THIS specific flight
    - working_alternative: What works when THIS flight fails
    - lesson: The lesson specific to THIS flight's failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCKED FLIGHTS CONFIGURATION (85% blocked = 17/20 for challenging baselines)
    # Maps flight → (error_code, working_alternatives, block_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN what "BK-401" means (phantom inventory).
    # Error codes do NOT hint at the solution (e.g., "use DL-123").
    #
    # With 85% blocked, Reflexion's probability of success with 4 retries:
    # P(success) = 1 - (17/20)(16/19)(15/18)(14/17) ≈ 50.9%
    # This makes baselines struggle while PRECEPT's learned rules shine!
    # ═══════════════════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════════════════
    # BALANCED DISTRIBUTION: Each blocked flight has ONLY ONE working alternative
    # This ensures learning one solution doesn't work for ALL scenarios
    # Distribution: ~50% DL-123, ~50% UA-200 (8/9 split for 17 flights)
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCKED_FLIGHTS: ClassVar[Dict[str, Dict]] = {
        # DL-123 works (9 flights)
        "AA-999": {
            "error_code": "BK-401",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "phantom inventory (GDS shows available but physically full)",
            "lesson": "AA-999 has phantom inventory, use DL-123",
            "airline": "American Airlines",
            "error_message": "Booking failed. Error code: BK-401. Please contact support.",
        },
        "WN-456": {
            "error_code": "BK-303",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "system maintenance window active",
            "lesson": "WN-456 under maintenance, use DL-123 as fallback",
            "airline": "Southwest",
            "error_message": "Service temporarily unavailable. Code: BK-303.",
        },
        "AF-201": {
            "error_code": "BK-411",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "international carrier API timeout",
            "lesson": "AF-201 API issues, use domestic DL-123",
            "airline": "Air France",
            "error_message": "Request timeout. Error: BK-411. Try again later.",
        },
        "EK-405": {
            "error_code": "BK-433",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "premium cabin oversold",
            "lesson": "EK-405 oversold, use DL-123",
            "airline": "Emirates",
            "error_message": "Capacity issue. Code: BK-433. Contact support.",
        },
        "CX-703": {
            "error_code": "BK-466",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "route temporarily suspended",
            "lesson": "CX-703 suspended, use DL-123",
            "airline": "Cathay Pacific",
            "error_message": "Route unavailable. Code: BK-466.",
        },
        "TK-106": {
            "error_code": "BK-499",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "payment gateway regional block",
            "lesson": "TK-106 payment blocked, use DL-123",
            "airline": "Turkish Airlines",
            "error_message": "Transaction failed. Code: BK-499.",
        },
        "QF-207": {
            "error_code": "BK-511",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "aircraft substitution pending",
            "lesson": "QF-207 equipment change, fallback DL-123",
            "airline": "Qantas",
            "error_message": "Schedule change. Error: BK-511.",
        },
        "AC-409": {
            "error_code": "BK-533",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "weather-related capacity reduction",
            "lesson": "AC-409 weather hold, use DL-123",
            "airline": "Air Canada",
            "error_message": "Service disruption. Code: BK-533.",
        },
        "NH-510": {
            "error_code": "BK-544",
            "working_alternatives": ["DL-123"],  # ONLY DL-123 works
            "block_reason": "slot allocation exceeded",
            "lesson": "NH-510 slot full, switch to DL-123",
            "airline": "ANA",
            "error_message": "Capacity exceeded. Error: BK-544.",
        },
        # UA-200 works (8 flights)
        "UA-666": {
            "error_code": "BK-502",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "GDS synchronization failure",
            "lesson": "UA-666 GDS desync, switch to UA-200",
            "airline": "United Airlines",
            "error_message": "Unable to complete reservation. Reference: BK-502.",
        },
        "BA-100": {
            "error_code": "BK-710",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "economy fare class sold out",
            "lesson": "BA-100 economy closed, only UA-200 has availability",
            "airline": "British Airways",
            "error_message": "Requested option not available. Error: BK-710.",
        },
        "LH-302": {
            "error_code": "BK-422",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "EU passenger data regulation hold",
            "lesson": "LH-302 GDPR hold, switch to UA-200",
            "airline": "Lufthansa",
            "error_message": "Unable to process. Reference: BK-422.",
        },
        "QR-501": {
            "error_code": "BK-444",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "codeshare agreement conflict",
            "lesson": "QR-501 codeshare blocked, fallback UA-200",
            "airline": "Qatar Airways",
            "error_message": "Booking unavailable. Error: BK-444.",
        },
        "SQ-602": {
            "error_code": "BK-455",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "frequent flyer integration error",
            "lesson": "SQ-602 FFP error, use UA-200",
            "airline": "Singapore Airlines",
            "error_message": "System error. Reference: BK-455.",
        },
        "JL-804": {
            "error_code": "BK-477",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "visa verification pending",
            "lesson": "JL-804 visa hold, use UA-200",
            "airline": "Japan Airlines",
            "error_message": "Processing delay. Error: BK-477.",
        },
        "KE-905": {
            "error_code": "BK-488",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "seat map synchronization failure",
            "lesson": "KE-905 seat sync issue, switch UA-200",
            "airline": "Korean Air",
            "error_message": "Inventory error. Reference: BK-488.",
        },
        "VS-308": {
            "error_code": "BK-522",
            "working_alternatives": ["UA-200"],  # ONLY UA-200 works
            "block_reason": "crew scheduling conflict",
            "lesson": "VS-308 crew issue, alternative UA-200",
            "airline": "Virgin Atlantic",
            "error_message": "Operational issue. Reference: BK-522.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # WORKING FLIGHTS - Guaranteed to succeed (3 working = 15% of total)
    # ═══════════════════════════════════════════════════════════════════════════
    WORKING_FLIGHTS: ClassVar[List[str]] = ["DL-123", "UA-200", "AA-200"]

    # ═══════════════════════════════════════════════════════════════════════════
    # DEFAULT FLIGHT - Used when task doesn't specify a flight
    # CRITICAL: Must be a BLOCKED flight to avoid giving unfair advantage!
    # The agent should NOT get lucky by defaulting to a working flight.
    # ═══════════════════════════════════════════════════════════════════════════
    DEFAULT_FLIGHT: ClassVar[str] = "AA-999"  # Blocked (phantom inventory)

    # ═══════════════════════════════════════════════════════════════════════════
    # ALL KNOWN FLIGHTS - Vocabulary for task parsing (20 total: 17 blocked + 3 working)
    # ═══════════════════════════════════════════════════════════════════════════
    ALL_FLIGHTS: ClassVar[List[str]] = [
        # Blocked (17)
        "AA-999",
        "UA-666",
        "WN-456",
        "BA-100",
        "AF-201",
        "LH-302",
        "EK-405",
        "QR-501",
        "SQ-602",
        "CX-703",
        "JL-804",
        "KE-905",
        "TK-106",
        "QF-207",
        "VS-308",
        "AC-409",
        "NH-510",
        # Working (3)
        "DL-123",
        "UA-200",
        "AA-200",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # ROUTES - Semantic groupings for coherent task generation
    # ═══════════════════════════════════════════════════════════════════════════
    ROUTES: ClassVar[Dict[str, Dict]] = {
        "JFK-LAX": {
            "origin": "New York JFK",
            "destination": "Los Angeles",
            "type": "domestic",
            "duration": "6h",
        },
        "JFK-SFO": {
            "origin": "New York JFK",
            "destination": "San Francisco",
            "type": "domestic",
            "duration": "5h30m",
        },
        "LAX-ORD": {
            "origin": "Los Angeles",
            "destination": "Chicago",
            "type": "domestic",
            "duration": "4h",
        },
        "JFK-LHR": {
            "origin": "New York JFK",
            "destination": "London",
            "type": "international",
            "duration": "7h30m",
        },
        "LAX-NRT": {
            "origin": "Los Angeles",
            "destination": "Tokyo",
            "type": "international",
            "duration": "12h",
        },
        "SFO-CDG": {
            "origin": "San Francisco",
            "destination": "Paris",
            "type": "international",
            "duration": "10h30m",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # BOOKING CONTEXTS - Affects urgency and business logic
    # ═══════════════════════════════════════════════════════════════════════════
    BOOKING_CONTEXTS: ClassVar[Dict[str, Dict]] = {
        "business": {"urgency": "high", "prefix": "business trip"},
        "leisure": {"urgency": "normal", "prefix": "vacation"},
        "urgent": {"urgency": "critical", "prefix": "urgent travel"},
        "weekend": {"urgency": "normal", "prefix": "weekend getaway"},
        "conference": {"urgency": "high", "prefix": "conference attendance"},
        "family": {"urgency": "normal", "prefix": "family visit"},
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PAYMENT ISSUES CONFIGURATION
    # Maps issue → (error_code, solution, lesson, description)
    # ═══════════════════════════════════════════════════════════════════════════
    PAYMENT_ISSUES: ClassVar[Dict[str, Dict]] = {
        "gateway_timeout": {
            "error_code": "GATEWAY-TIMEOUT-504",
            "solution": "use_idempotency_key",
            "lesson": "Gateway timeout requires idempotency key on retry to prevent double charge",
            "description": "payment provider timeout",
            "valid_gateways": ["stripe", "paypal"],
        },
        "partial_charge": {
            "error_code": "PARTIAL-CHARGE-206",
            "solution": "implement_saga_pattern",
            "lesson": "Partial charge requires saga pattern for multi-step payment rollback",
            "description": "partial payment processed",
            "valid_gateways": ["stripe", "square"],
        },
        "fraud_hold": {
            "error_code": "FRAUD-HOLD-451",
            "solution": "trigger_manual_review",
            "lesson": "High-value bookings trigger fraud review, escalate to manual approval",
            "description": "fraud prevention hold triggered",
            "valid_gateways": ["stripe", "braintree"],
        },
        "3ds_failure": {
            "error_code": "3DS-AUTH-FAIL",
            "solution": "retry_with_3ds_exemption",
            "lesson": "3DS authentication failure, retry with SCA exemption for low-risk",
            "description": "3D Secure authentication failed",
            "valid_gateways": ["stripe", "adyen"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # INVENTORY ISSUES CONFIGURATION
    # Maps issue → (error_code, solution, lesson, description)
    # ═══════════════════════════════════════════════════════════════════════════
    INVENTORY_ISSUES: ClassVar[Dict[str, Dict]] = {
        "overbooking": {
            "error_code": "OVERBOOK-409",
            "solution": "optimistic_locking",
            "lesson": "Overbooking requires optimistic locking with version check",
            "description": "seat oversold due to race condition",
        },
        "connecting_mismatch": {
            "error_code": "CONNECT-UNAVAIL-410",
            "solution": "atomic_booking",
            "lesson": "Connected itineraries require atomic booking transaction",
            "description": "connection unavailable after first leg booked",
        },
        "rate_drift": {
            "error_code": "RATE-DRIFT-422",
            "solution": "rate_lock_token",
            "lesson": "Price changed during checkout, use rate lock token from search",
            "description": "price increased during checkout flow",
        },
        "seat_map_stale": {
            "error_code": "SEATMAP-STALE-409",
            "solution": "refresh_before_select",
            "lesson": "Seat map is stale, refresh seat map before selection",
            "description": "seat already taken when selected",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # PAYMENT GATEWAYS - Vocabulary for domain strategy
    # ═══════════════════════════════════════════════════════════════════════════
    PAYMENT_GATEWAYS: ClassVar[List[str]] = [
        "stripe",
        "paypal",
        "square",
        "braintree",
        "adyen",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # BOOKING SYSTEMS - GDS providers
    # ═══════════════════════════════════════════════════════════════════════════
    BOOKING_SYSTEMS: ClassVar[List[str]] = ["amadeus", "sabre", "travelport", "galileo"]

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "phantom inventory": "{flight} has phantom inventory (shows available but full). Use {alt} instead.",
        "gds": "{flight} has GDS sync issues. Switch to {alt} for reliable booking.",
        "maintenance": "{flight} system under maintenance. Use {alt} as fallback.",
        "fare class": "{flight} fare class closed. Only {alt} has availability.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # TASK TEMPLATES - {flight}, {route}, {context} are replaced
    # ═══════════════════════════════════════════════════════════════════════════
    TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Book flight {flight} for {context}",
        "Reserve seat on {flight} ({route})",
        "Book {flight} tickets for {context}",
        "Purchase {flight} for {route} travel",
        "Secure booking on {flight} for {context}",
    ]

    TEST_TEMPLATES: ClassVar[List[str]] = [
        "Book {flight} for another passenger ({context})",
        "Reserve {flight} {route} urgently",
        "Finalize {flight} booking for {context}",
        "Process {flight} reservation ({route})",
        "Complete {flight} purchase for {context}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # PAYMENT SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    PAYMENT_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Process payment via {gateway} for booking {booking_id}",
        "Charge {gateway} for reservation {booking_id}",
        "Complete {gateway} purchase for order {booking_id}",
    ]

    PAYMENT_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Retry {gateway} payment for {booking_id}",
        "Finalize {gateway} charge for booking {booking_id}",
        "Process {gateway} refund for {booking_id}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # INVENTORY SCENARIO TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    INVENTORY_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Book last available seat for {route}",
        "Reserve final seat on {route} flight",
        "Secure remaining inventory for {route}",
    ]

    INVENTORY_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Grab last seat on {route}",
        "Book remaining seat for {route}",
        "Finalize last-minute booking for {route}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFLICT RESOLUTION TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    DYNAMIC_OVERRIDE_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Book {flight} for {context} via {route}",
        "Reserve {flight} seats for {context}",
        "Process {flight} booking ({context})",
    ]

    DYNAMIC_OVERRIDE_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Urgent: Book {flight} for {context}",
        "Rush {flight} reservation ({route})",
        "Express booking for {flight}",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY REQUIREMENTS - For static knowledge scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    REGULATORY_REQUIREMENTS: ClassVar[Dict[str, Dict]] = {
        "visa_documentation": {
            "training_templates": [
                "Book international {flight} to {dest} without visa check",
                "Rush {flight} to {dest} - skip documentation",
            ],
            "test_templates": [
                "Express {flight} to {dest} - bypass visa requirement",
            ],
            "lesson": "International travel to {dest} requires visa verification. Cannot bypass.",
            "error": "VISA-REQUIRED-403",
            "destinations": ["London", "Tokyo", "Paris"],
        },
        "passport_validity": {
            "training_templates": [
                "Book {flight} to {dest} with expiring passport",
                "Process {flight} international booking - passport expires in 2 months",
            ],
            "test_templates": [
                "Rush {flight} to {dest} with near-expiry passport",
            ],
            "lesson": "Passport must be valid 6 months beyond travel date for {dest}.",
            "error": "PASSPORT-INVALID-403",
            "destinations": ["Tokyo", "Singapore", "Dubai"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # AGREEMENT SCENARIOS - Static and dynamic align
    # ═══════════════════════════════════════════════════════════════════════════
    FALLBACK_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("AA-999", "DL-123"),
        ("UA-666", "DL-123"),
        ("WN-456", "DL-123"),
        ("BA-100", "DL-123"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # DESTINATION → CUSTOMS/VISA MAPPING (For consistency)
    # ═══════════════════════════════════════════════════════════════════════════
    DESTINATION_REQUIREMENTS: ClassVar[Dict[str, str]] = {
        "London": "visa_documentation",
        "Tokyo": "passport_validity",
        "Paris": "visa_documentation",
        "Singapore": "passport_validity",
        "Dubai": "passport_validity",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID SOLUTIONS
    # For multi-condition scenarios, ONLY these 2 flights are valid.
    # Each condition_key is deterministically mapped to exactly ONE flight.
    # This matches logistics domain difficulty where only antwerp/hamburg work.
    #
    # Difficulty Analysis:
    # - Total flights available: 20
    # - Valid for multi-condition: 2 (hash-mapped to 1 per condition_key)
    # - Random success rate: 1/20 = 5% (stricter than logistics!)
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_FLIGHTS: ClassVar[List[str]] = [
        "DL-123",  # Delta working flight
        "UA-200",  # United working flight
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CONDITIONS - For compositional generalization
    # Maps semantic condition codes to solutions with tier priorities
    # Higher tier wins when conditions conflict (same as logistics)
    # ═══════════════════════════════════════════════════════════════════════════
    SEMANTIC_CONDITIONS: ClassVar[Dict[str, Dict]] = {
        # Tier 3 (Highest): Protection - non-negotiable
        "CANCEL": {"solution": "DL-123", "tier": 3, "meaning": "Free cancellation required"},
        # Tier 2 (Middle): Flexibility requirements
        "REFUND": {"solution": "DL-123", "tier": 2, "meaning": "Fully refundable ticket"},
        "CHANGE": {"solution": "UA-200", "tier": 2, "meaning": "Free date change allowed"},
        "BUSI": {"solution": "UA-200", "tier": 2, "meaning": "Business travel requirements"},
        # Tier 1 (Lowest): Cost/convenience preferences
        "CHEAP": {"solution": "UA-200", "tier": 1, "meaning": "Budget-conscious booking"},
        "FAST": {"solution": "DL-123", "tier": 1, "meaning": "Fastest route preferred"},
        "NIGHT": {"solution": "UA-200", "tier": 1, "meaning": "Overnight travel acceptable"},
        "CONN": {"solution": "DL-123", "tier": 1, "meaning": "Connections acceptable for savings"},
    }

    @classmethod
    def get_valid_solution_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid flight for a given condition_key.

        For SEMANTIC conditions (CANCEL, BUSI, FAST, etc.):
        - Uses priority-based resolution (highest tier wins)
        - Solutions are DERIVABLE from atomic precepts
        - Enables P₁ > 0% through compositional reasoning

        For BLACK SWAN conditions (BK-401, etc.):
        - Uses deterministic hash
        - Solutions are NOT derivable (require exploration)
        - P₁ = 0% expected (first-try success unlikely)
        """
        # Check if this is a semantic condition
        conditions = condition_key.split("+")
        is_semantic = all(c.strip().upper() in cls.SEMANTIC_CONDITIONS for c in conditions)

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
            return "DL-123"  # Fallback
        else:
            # BLACK SWAN MODE: Hash-based (solutions NOT derivable)
            import hashlib
            hash_val = int(
                hashlib.sha256(condition_key.encode()).hexdigest()[:8], 16
            )
            idx = abs(hash_val) % len(cls.MULTI_CONDITION_VALID_FLIGHTS)
            return cls.MULTI_CONDITION_VALID_FLIGHTS[idx]


# Convenience function to get config
def get_booking_config() -> type:
    """Get the booking configuration class."""
    return BookingConfig
