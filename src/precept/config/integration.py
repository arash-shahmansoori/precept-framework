"""
Integration Domain Configuration for PRECEPT.

Single source of truth for all integration/API-related configuration including:
- OAuth sources and token issues
- Gateway endpoints and failure modes
- Webhook issues and solutions
- API error codes and recovery strategies
- Scenario generation templates

Usage:
    from precept.config import IntegrationConfig

    # Access configuration
    config = IntegrationConfig
    oauth_sources = config.OAUTH_SOURCES
    gateway_endpoints = config.GATEWAY_ENDPOINTS
"""

from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple


@dataclass(frozen=True)
class IntegrationConfig:
    """
    Centralized configuration for integration domain.

    SINGLE SOURCE OF TRUTH for all integration-related data:
    - OAuth source configurations and error codes
    - Gateway endpoint mappings
    - Webhook issue patterns
    - Scenario generation templates

    COHERENCE GUARANTEE: Each source/endpoint has consistent attributes:
    - error_code: The error for THIS specific source/endpoint
    - recovery_action: What works when THIS integration fails
    - lesson: The lesson specific to THIS failure mode
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID SOURCES - Consistent sources for multi-condition tests
    # These are the ONLY sources that will succeed in the simulator for
    # multi-condition scenarios. This ensures agents learn valid alternatives.
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_SOURCES: ClassVar[List[str]] = [
        "salesforce-backup",
        "hubspot-v2",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # OAUTH SOURCES - Token expiration and auth issues
    # Maps source → (error_code, recovery_action, failure_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "INT-401" means re-authenticate.
    # Error codes do NOT hint at OAuth, token, or scope issues.
    # ═══════════════════════════════════════════════════════════════════════════
    OAUTH_SOURCES: ClassVar[Dict[str, Dict]] = {
        "salesforce": {
            "error_code": "INT-401",  # Vague: doesn't reveal OAuth issue
            "recovery_action": "re-authenticate",
            "working_alternatives": ["re-authenticate", "refresh_token"],
            "failure_reason": "token expired silently (no refresh possible)",
            "lesson": "Salesforce 401 = re-authenticate completely (not just retry)",
            "data_type": "CRM data",
            "error_message": "Connection failed. Error: INT-401. Retry or contact admin.",
        },
        "hubspot": {
            "error_code": "INT-402",  # Vague: different code, similar issue
            "recovery_action": "re-authenticate",
            "working_alternatives": ["re-authenticate", "check_scopes"],
            "failure_reason": "refresh token invalid or revoked",
            "lesson": "HubSpot 401 with invalid refresh = full re-auth required",
            "data_type": "contacts",
            "error_message": "Request denied. Code: INT-402.",
        },
        "zendesk": {
            "error_code": "INT-429",  # Vague: doesn't reveal rate limit
            "recovery_action": "exponential-backoff",
            "working_alternatives": ["exponential-backoff", "rate_limit_queue"],
            "failure_reason": "rate limit exceeded (too many requests)",
            "lesson": "Zendesk 429 = exponential backoff (not immediate retry)",
            "data_type": "tickets",
            "error_message": "Service temporarily unavailable. Reference: INT-429.",
        },
        "stripe": {
            "error_code": "INT-511",  # Vague: doesn't reveal signature issue
            "recovery_action": "rotate-keys",
            "working_alternatives": ["rotate-keys", "verify_endpoint"],
            "failure_reason": "webhook signature mismatch (key rotated)",
            "lesson": "Stripe signature error = rotate webhook signing keys",
            "data_type": "payment webhooks",
            "error_message": "Verification failed. Error: INT-511.",
        },
        "google_workspace": {
            "error_code": "INT-403",  # Vague: doesn't reveal scope issue
            "recovery_action": "request-scopes",
            "working_alternatives": ["request-scopes", "admin_consent"],
            "failure_reason": "missing required OAuth scopes",
            "lesson": "Google 403 = check OAuth scopes in API console",
            "data_type": "calendar events",
            "error_message": "Access denied. Code: INT-403. Check permissions.",
        },
        "microsoft_graph": {
            "error_code": "INT-404",  # Vague: doesn't reveal consent requirement
            "recovery_action": "admin-consent",
            "working_alternatives": ["admin-consent", "incremental_consent"],
            "failure_reason": "admin consent required for tenant",
            "lesson": "Microsoft Graph 403 = need admin consent for enterprise scope",
            "data_type": "user profiles",
            "error_message": "Operation blocked. Reference: INT-404.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # GATEWAY ENDPOINTS - Various failure modes
    # Maps endpoint → (error_code, recovery_action, failure_reason, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # The agent must LEARN that "GW-502" means escalate to infra.
    # ═══════════════════════════════════════════════════════════════════════════
    GATEWAY_ENDPOINTS: ClassVar[Dict[str, Dict]] = {
        "legacy-erp": {
            "error_code": "GW-502",  # Vague: doesn't reveal upstream issue
            "recovery_action": "escalate-to-infra",
            "working_alternatives": ["escalate-to-infra", "check_health_endpoint"],
            "failure_reason": "upstream service unavailable",
            "lesson": "Legacy ERP 502 = escalate to infra team (not retryable)",
            "error_message": "Request failed. Error: GW-502. Try again later.",
        },
        "partner-api": {
            "error_code": "GW-603",  # Vague: doesn't reveal WAF block
            "recovery_action": "whitelist-ip",
            "working_alternatives": ["whitelist-ip", "use_api_gateway"],
            "failure_reason": "WAF blocking request (IP not whitelisted)",
            "lesson": "Partner API WAF 403 = request IP whitelist from partner",
            "error_message": "Access denied. Code: GW-603.",
        },
        "payment-gateway": {
            "error_code": "GW-704",  # Vague: doesn't reveal SSL issue
            "recovery_action": "rotate-certificates",
            "working_alternatives": ["rotate-certificates", "check_cert_chain"],
            "failure_reason": "SSL certificate expired or invalid",
            "lesson": "Payment gateway SSL error = rotate certificates immediately",
            "error_message": "Connection error. Reference: GW-704.",
        },
        "analytics-api": {
            "error_code": "GW-805",  # Vague: doesn't reveal CORS issue
            "recovery_action": "use-backend-proxy",
            "working_alternatives": ["use-backend-proxy", "configure_cors"],
            "failure_reason": "CORS policy blocking frontend request",
            "lesson": "Analytics CORS error = use backend proxy (not browser direct)",
            "error_message": "Request blocked. Error: GW-805.",
        },
        "inventory-service": {
            "error_code": "GW-906",  # Vague: doesn't reveal mesh issue
            "recovery_action": "check-sidecar",
            "working_alternatives": ["check-sidecar", "restart_envoy"],
            "failure_reason": "service mesh sidecar unhealthy",
            "lesson": "Service mesh timeout = check Envoy/Istio sidecar health",
            "error_message": "Service timeout. Code: GW-906.",
        },
        "notification-service": {
            "error_code": "GW-107",  # Vague: doesn't reveal circuit breaker
            "recovery_action": "wait-circuit-reset",
            "working_alternatives": ["wait-circuit-reset", "use_fallback"],
            "failure_reason": "circuit breaker open due to failures",
            "lesson": "Circuit open = wait for reset or use fallback channel",
            "error_message": "Service unavailable. Reference: GW-107.",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # WEBHOOK ISSUES
    # Maps issue → (error_code, solution, issue_description, lesson)
    #
    # CRITICAL: Error codes are VAGUE and non-revealing!
    # ═══════════════════════════════════════════════════════════════════════════
    WEBHOOK_ISSUES: ClassVar[Dict[str, Dict]] = {
        "duplicate_delivery": {
            "error_code": "WH-409",  # Vague: doesn't reveal duplicate
            "solution": "idempotency_check",
            "working_alternatives": ["idempotency_check", "event_deduplication"],
            "issue_description": "duplicate webhook delivery",
            "lesson": "Make webhook handlers idempotent with event ID tracking",
            "error_message": "Event processing error. Code: WH-409.",
        },
        "out_of_order": {
            "error_code": "WH-400",  # Vague: doesn't reveal ordering issue
            "solution": "event_timestamp",
            "working_alternatives": ["event_timestamp", "sequence_number"],
            "issue_description": "events delivered out of order",
            "lesson": "Use event timestamps for ordering, not delivery order",
            "error_message": "Sequence error. Reference: WH-400.",
        },
        "payload_mismatch": {
            "error_code": "WH-422",  # Vague: doesn't reveal schema mismatch
            "solution": "version_check",
            "working_alternatives": ["version_check", "schema_migration"],
            "issue_description": "schema version mismatch",
            "lesson": "Check webhook API version in headers before processing",
            "error_message": "Payload error. Error: WH-422.",
        },
        "signature_expired": {
            "error_code": "WH-511",  # Vague: doesn't reveal signature expiry
            "solution": "clock_sync",
            "working_alternatives": ["clock_sync", "increase_tolerance"],
            "issue_description": "webhook signature timestamp expired",
            "error_message": "Verification failed. Code: WH-511.",
            "lesson": "Sync server clock with NTP, signature has 5-min window",
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # AUTH PROVIDERS - Vocabulary for domain strategy
    # ═══════════════════════════════════════════════════════════════════════════
    AUTH_PROVIDERS: ClassVar[List[str]] = ["oauth", "saml", "jwt", "api_key", "mtls"]

    # ═══════════════════════════════════════════════════════════════════════════
    # API ENDPOINTS - Vocabulary for domain strategy
    # ═══════════════════════════════════════════════════════════════════════════
    API_ENDPOINT_TYPES: ClassVar[List[str]] = [
        "rest",
        "graphql",
        "grpc",
        "soap",
        "websocket",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN SOURCES - All integration sources
    # 9 sources to match logistics difficulty (1/9 = 11% random success)
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_SOURCES: ClassVar[List[str]] = [
        "salesforce",
        "hubspot",
        "zendesk",
        "stripe",
        "google_workspace",
        "microsoft_graph",
        "slack_enterprise",  # Added for difficulty parity
        "jira_cloud",  # Added for difficulty parity
        "confluence_server",  # Added for difficulty parity
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # KNOWN ENDPOINTS - All gateway endpoints
    # 9 endpoints to match logistics difficulty (1/9 = 11% random success)
    # ═══════════════════════════════════════════════════════════════════════════
    KNOWN_ENDPOINTS: ClassVar[List[str]] = [
        "legacy-erp",
        "partner-api",
        "payment-gateway",
        "analytics-api",
        "inventory-service",
        "notification-service",
        "audit-logger",  # Added for difficulty parity
        "data-warehouse",  # Added for difficulty parity
        "compliance-gateway",  # Added for difficulty parity
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SYNC CONTEXTS
    # ═══════════════════════════════════════════════════════════════════════════
    SYNC_CONTEXTS: ClassVar[Dict[str, Dict]] = {
        "crm_sync": {"action": "sync", "target": "CRM"},
        "data_pull": {"action": "pull", "target": "database"},
        "analytics_export": {"action": "export", "target": "analytics"},
        "backup_restore": {"action": "restore", "target": "backup"},
        "migration": {"action": "migrate", "target": "new system"},
        "real_time_stream": {"action": "stream", "target": "event bus"},
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # DATA OPERATIONS - Types of data operations
    # ═══════════════════════════════════════════════════════════════════════════
    DATA_OPERATIONS: ClassVar[List[str]] = [
        "leads",
        "contacts",
        "deals",
        "accounts",
        "opportunities",
        "tickets",
        "events",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK REASON → LESSON TEMPLATE MAPPING
    # ═══════════════════════════════════════════════════════════════════════════
    BLOCK_REASON_TEMPLATES: ClassVar[Dict[str, str]] = {
        "token expired": "{source} token expired. {action} required.",
        "rate limit": "{source} rate limited. Use {action}.",
        "waf blocked": "{endpoint} blocked by WAF. {action} needed.",
        "circuit open": "{endpoint} circuit breaker open. {action}.",
        "webhook": "Webhook {issue}. Implement {solution}.",
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # OAUTH TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    OAUTH_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Sync {source} {data_type} with local system",
        "Pull {source} {data_type} into database",
        "Fetch {data_type} from {source} API",
        "Connect to {source} for {data_type}",
    ]

    OAUTH_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Sync {source} {data_op} for marketing campaign",
        "Pull {source} {data_op} into reporting dashboard",
        "Export {source} {data_op} for analysis",
        "Refresh {source} {data_op} cache",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # GATEWAY TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    GATEWAY_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Call {endpoint} API for data retrieval",
        "Connect to {endpoint} endpoint",
        "Fetch data from {endpoint}",
        "Query {endpoint} for records",
    ]

    GATEWAY_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Retry {endpoint} connection",
        "Resume {endpoint} data sync",
        "Reconnect to {endpoint} service",
        "Refresh {endpoint} integration",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # WEBHOOK TASK TEMPLATES
    # ═══════════════════════════════════════════════════════════════════════════
    WEBHOOK_TRAINING_TEMPLATES: ClassVar[List[str]] = [
        "Process incoming {source} webhook",
        "Handle {source} event notification",
        "Receive {source} callback",
        "Process {source} webhook payload",
    ]

    WEBHOOK_TEST_TEMPLATES: ClassVar[List[str]] = [
        "Handle retried {source} webhook",
        "Process duplicate {source} event",
        "Validate {source} webhook signature",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # REGULATORY REQUIREMENTS - For static knowledge scenarios
    # ═══════════════════════════════════════════════════════════════════════════
    REGULATORY_REQUIREMENTS: ClassVar[Dict[str, Dict]] = {
        "data_residency": {
            "training_templates": [
                "Sync {source} EU customer data to US region",
                "Transfer {source} GDPR data cross-border",
            ],
            "test_templates": [
                "Quick sync {source} EU data to US",
            ],
            "lesson": "GDPR requires EU data residency. Cannot transfer to US without SCCs.",
            "error": "GDPR-VIOLATION-403",
            "sources": ["salesforce", "hubspot"],
        },
        "pci_compliance": {
            "training_templates": [
                "Store {source} card data locally",
                "Cache {source} payment details",
            ],
            "test_templates": [
                "Save {source} payment info for retry",
            ],
            "lesson": "PCI-DSS prohibits storing card data. Use tokenization.",
            "error": "PCI-VIOLATION-403",
            "sources": ["stripe", "payment-gateway"],
        },
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # AGREEMENT SCENARIOS - Static and dynamic align
    # ═══════════════════════════════════════════════════════════════════════════
    RECOVERY_PAIRS: ClassVar[List[Tuple[str, str]]] = [
        ("salesforce", "re-authenticate"),
        ("hubspot", "re-authenticate"),
        ("zendesk", "exponential-backoff"),
        ("stripe", "rotate-keys"),
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # MULTI-CONDITION VALID SOURCES/ENDPOINTS - Strict enforcement for fair comparison
    # ═══════════════════════════════════════════════════════════════════════════
    # For multi-condition scenarios, ONLY these sources/endpoints are valid.
    # Each condition_key is deterministically mapped to exactly ONE solution.
    # This matches logistics domain difficulty where only antwerp/hamburg work.
    # ═══════════════════════════════════════════════════════════════════════════
    MULTI_CONDITION_VALID_SOLUTIONS: ClassVar[List[str]] = [
        "salesforce-backup",
        "hubspot-v2",
    ]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CONDITIONS - For compositional generalization
    # Maps semantic condition codes to solutions with tier priorities
    # Higher tier wins when conditions conflict (same as logistics)
    # ═══════════════════════════════════════════════════════════════════════════
    SEMANTIC_CONDITIONS: ClassVar[Dict[str, Dict]] = {
        # Tier 3 (Highest): Security/compliance - non-negotiable
        "SECURE": {
            "solution": "salesforce-backup",
            "tier": 3,
            "meaning": "Security-hardened integration",
        },
        # Tier 2 (Middle): Reliability requirements
        "VERIFY": {
            "solution": "hubspot-v2",
            "tier": 2,
            "meaning": "Request verification required",
        },
        "AUDIT": {
            "solution": "salesforce-backup",
            "tier": 2,
            "meaning": "Audit trail required",
        },
        # Tier 1 (Lowest): Performance/convenience preferences
        "BATCH": {
            "solution": "salesforce-backup",
            "tier": 1,
            "meaning": "Batch processing for efficiency",
        },
        "STREAM": {
            "solution": "hubspot-v2",
            "tier": 1,
            "meaning": "Real-time streaming data",
        },
        "QUERY": {
            "solution": "hubspot-v2",
            "tier": 1,
            "meaning": "Flexible query patterns",
        },
    }

    @classmethod
    def get_valid_solution_for_conditions(cls, condition_key: str) -> str:
        """
        Get the ONLY valid solution for a given condition_key.

        For SEMANTIC conditions (BATCH, VERIFY, STREAM, etc.):
        - Uses priority-based resolution (highest tier wins)
        - Solutions are DERIVABLE from atomic precepts
        - Enables P₁ > 0% through compositional reasoning

        For BLACK SWAN conditions (INT-401, GW-502, etc.):
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
            return "salesforce-backup"  # Fallback
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


# Convenience function to get config
def get_integration_config() -> type:
    """Get the integration configuration class."""
    return IntegrationConfig
