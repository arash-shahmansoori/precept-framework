"""
Mock Data for PRECEPT Tests.

This module provides sample data for testing various components.
"""

from typing import Any, Dict, List

# =============================================================================
# SAMPLE TASKS
# =============================================================================

SAMPLE_TASKS: List[Dict[str, Any]] = [
    {
        "task": "Book shipment from Rotterdam to Boston",
        "expected_error": "R-482",
        "expected_solution": "Antwerp",
        "domain": "logistics",
    },
    {
        "task": "Book shipment from Hamburg to New York",
        "expected_error": "H-903",
        "expected_solution": "Antwerp",
        "domain": "logistics",
    },
    {
        "task": "Clear customs for Shanghai to New York",
        "expected_error": "CUSTOMS-COO-001",
        "expected_solution": "Obtain COO",
        "domain": "logistics",
    },
    {
        "task": "Book shipment from Shanghai to London",
        "expected_error": "SH-701",
        "expected_solution": "Ningbo",
        "domain": "logistics",
    },
]

# =============================================================================
# SAMPLE ERRORS
# =============================================================================

SAMPLE_ERRORS: Dict[str, Dict[str, str]] = {
    "R-482": {
        "code": "R-482",
        "message": "Rotterdam port is currently blocked for all traffic",
        "solution": "Use Antwerp as alternative",
        "severity": "hard",
    },
    "H-903": {
        "code": "H-903",
        "message": "Hamburg to US destinations blocked due to regulations",
        "solution": "Use Antwerp or Rotterdam for US routes",
        "severity": "hard",
    },
    "SH-701": {
        "code": "SH-701",
        "message": "Shanghai port congestion - delays expected",
        "solution": "Use Ningbo or Shenzhen as alternative",
        "severity": "soft",
    },
    "CUSTOMS-COO-001": {
        "code": "CUSTOMS-COO-001",
        "message": "Certificate of Origin missing for customs clearance",
        "solution": "Obtain COO from manufacturer",
        "severity": "hard",
    },
    "LA-550": {
        "code": "LA-550",
        "message": "Los Angeles port strike - operations suspended",
        "solution": "Use Long Beach or Oakland",
        "severity": "transient",
    },
}

# =============================================================================
# SAMPLE RULES
# =============================================================================

SAMPLE_RULES: Dict[str, str] = {
    "R-482": "Rotterdam blocked → Use Antwerp",
    "H-903": "Hamburg to US blocked → Use Antwerp/Rotterdam",
    "SH-701": "Shanghai congested → Use Ningbo/Shenzhen",
    "CUSTOMS-COO-001": "Missing COO → Obtain from manufacturer",
    "LA-550": "LA port strike → Use Long Beach/Oakland",
}

# =============================================================================
# SAMPLE EXPERIENCES
# =============================================================================

SAMPLE_EXPERIENCES: List[Dict[str, Any]] = [
    {
        "task": "Book shipment from Rotterdam to Boston",
        "outcome": "success",
        "strategy": "pivot_to_antwerp",
        "lessons": "Rotterdam R-482 requires Antwerp fallback",
        "domain": "logistics",
    },
    {
        "task": "Book shipment from Hamburg to NYC",
        "outcome": "success",
        "strategy": "pivot_to_antwerp",
        "lessons": "Hamburg H-903 blocks US destinations",
        "domain": "logistics",
    },
    {
        "task": "Clear customs Shanghai to New York",
        "outcome": "success",
        "strategy": "obtain_coo",
        "lessons": "New York customs always requires COO",
        "domain": "logistics",
    },
]

# =============================================================================
# SAMPLE SCENARIOS
# =============================================================================

SAMPLE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "task": "Book shipment from Rotterdam to Boston",
        "expected": "R-482",
        "tests_learning": "port_closure",
        "phase": "training",
    },
    {
        "task": "Book shipment from Hamburg to New York",
        "expected": "H-903",
        "tests_learning": "destination_restriction",
        "phase": "training",
    },
    {
        "task": "Clear customs for Shanghai shipment to New York",
        "expected": "CUSTOMS-COO-001",
        "tests_learning": "customs_documentation",
        "phase": "training",
    },
    {
        "task": "Book shipment from Rotterdam to NYC",
        "expected": "R-482",
        "tests_learning": "learned_port_closure",
        "phase": "test",
    },
]

# =============================================================================
# LLM RESPONSE SAMPLES
# =============================================================================

LLM_RESPONSE_SAMPLES: Dict[str, str] = {
    "valid_solution": """
SOLUTION: Antwerp
REASONING: Rotterdam is blocked due to R-482, Antwerp is the nearest alternative port
CONFIDENCE: high
""",
    "explore": """
SOLUTION: EXPLORE
REASONING: Need more information about available alternatives
CONFIDENCE: low
""",
    "exhausted": """
SOLUTION: EXHAUSTED
REASONING: All available options have been tried and failed
CONFIDENCE: low
""",
    "reflexion": """
REFLECTION: The previous attempt with Rotterdam failed because of the R-482 port
closure. This is a hard constraint that cannot be bypassed.
LESSON: For any Rotterdam shipment, immediately pivot to Antwerp without trying
Rotterdam first.
SOLUTION: Antwerp
REASONING: Applying learned lesson from previous failure
CONFIDENCE: high
""",
    "malformed": """
The best option would be to use Antwerp since Rotterdam is blocked.
I think this should work.
""",
}

# =============================================================================
# CONSTRAINT SAMPLES
# =============================================================================

CONSTRAINT_SAMPLES: List[Dict[str, Any]] = [
    {
        "solution": "rotterdam",
        "error_code": "R-482",
        "error_message": "Port blocked",
        "expected_type": "hard",
    },
    {
        "solution": "hamburg",
        "error_code": "H-903",
        "error_message": "Destination blocked",
        "expected_type": "hard",
    },
    {
        "solution": "shanghai",
        "error_code": "SH-701",
        "error_message": "Port congested",
        "expected_type": "soft",
    },
]
