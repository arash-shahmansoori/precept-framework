"""
Strategy Registry for Domain Strategies.

This module provides a registry and factory functions for domain strategies.

Usage:
    from precept import get_domain_strategy, get_baseline_strategy

    # Get a learning strategy
    strategy = get_domain_strategy("logistics")

    # Get a baseline strategy (no learning)
    baseline = get_baseline_strategy("logistics")

    # List all available domains
    domains = list_available_domains()
"""

from typing import Dict, List, Type

from .domain_strategies.base import BaselineDomainStrategy, DomainStrategy
from .domain_strategies.logistics import (
    LogisticsBaselineStrategy,
    LogisticsDomainStrategy,
)
from .domain_strategies.coding import (
    CodingBaselineStrategy,
    CodingDomainStrategy,
)
from .domain_strategies.devops import (
    DevOpsBaselineStrategy,
    DevOpsDomainStrategy,
)
from .domain_strategies.finance import (
    FinanceBaselineStrategy,
    FinanceDomainStrategy,
)
from .domain_strategies.booking import (
    BookingBaselineStrategy,
    BookingDomainStrategy,
)
from .domain_strategies.integration import (
    IntegrationBaselineStrategy,
    IntegrationDomainStrategy,
)


# Learning strategies (PRECEPT - with DynamicRuleParser, NO hardcoded knowledge)
DOMAIN_STRATEGIES: Dict[str, Type[DomainStrategy]] = {
    "logistics": LogisticsDomainStrategy,
    "coding": CodingDomainStrategy,
    "devops": DevOpsDomainStrategy,
    "finance": FinanceDomainStrategy,
    "booking": BookingDomainStrategy,
    "integration": IntegrationDomainStrategy,
}

# Baseline strategies (NO learning - random fallback only)
BASELINE_STRATEGIES: Dict[str, Type[BaselineDomainStrategy]] = {
    "logistics": LogisticsBaselineStrategy,
    "coding": CodingBaselineStrategy,
    "devops": DevOpsBaselineStrategy,
    "finance": FinanceBaselineStrategy,
    "booking": BookingBaselineStrategy,
    "integration": IntegrationBaselineStrategy,
}


def get_domain_strategy(domain: str, max_retries: int = None) -> DomainStrategy:
    """
    Factory function to get a domain strategy by name.

    ALL strategies use DynamicRuleParser - NO hardcoded knowledge.

    Args:
        domain: One of "logistics", "coding", "devops", "finance", "booking", "integration"
        max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES (2).
                    - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                    - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                    - 4 = lenient (1 initial + 4 retries = 5 attempts)

    Returns:
        Instantiated DomainStrategy for that domain

    Raises:
        ValueError: If domain is not recognized

    Example:
        strategy = get_domain_strategy("logistics")
        parsed_task = strategy.parse_task("Book shipment from Rotterdam to Boston")

        # With custom max_retries
        strict_strategy = get_domain_strategy("logistics", max_retries=1)
    """
    if domain not in DOMAIN_STRATEGIES:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_STRATEGIES.keys())}")
    return DOMAIN_STRATEGIES[domain](max_retries=max_retries)


def get_baseline_strategy(domain: str, max_retries: int = None) -> BaselineDomainStrategy:
    """
    Factory function to get a baseline strategy by name.

    ALL baseline strategies have NO learning - random fallback only.

    Args:
        domain: One of "logistics", "coding", "devops", "finance", "booking", "integration"
        max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES (2).
                    - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                    - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                    - 4 = lenient (1 initial + 4 retries = 5 attempts)

    Returns:
        Instantiated BaselineDomainStrategy for that domain

    Raises:
        ValueError: If domain is not recognized

    Example:
        baseline = get_baseline_strategy("logistics")
        parsed_task = baseline.parse_task("Book shipment from Rotterdam to Boston")

        # With custom max_retries
        strict_baseline = get_baseline_strategy("logistics", max_retries=1)
    """
    if domain not in BASELINE_STRATEGIES:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(BASELINE_STRATEGIES.keys())}")
    return BASELINE_STRATEGIES[domain](max_retries=max_retries)


def list_available_domains() -> List[str]:
    """
    List all available domain names.

    Returns:
        List of domain names
    """
    return list(DOMAIN_STRATEGIES.keys())


def get_domain_strategy_class(domain: str) -> Type[DomainStrategy]:
    """
    Get the domain strategy class (not instantiated).

    Useful for subclassing or inspection.

    Args:
        domain: Domain name

    Returns:
        DomainStrategy class
    """
    if domain not in DOMAIN_STRATEGIES:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_STRATEGIES.keys())}")
    return DOMAIN_STRATEGIES[domain]


def get_baseline_strategy_class(domain: str) -> Type[BaselineDomainStrategy]:
    """
    Get the baseline strategy class (not instantiated).

    Useful for subclassing or inspection.

    Args:
        domain: Domain name

    Returns:
        BaselineDomainStrategy class
    """
    if domain not in BASELINE_STRATEGIES:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(BASELINE_STRATEGIES.keys())}")
    return BASELINE_STRATEGIES[domain]
