"""
Domain Strategy Pattern for PRECEPT Black Swan Handling.

This module provides a pluggable strategy pattern for handling different
black swan categories (Logistics, Coding, DevOps, Finance, Booking, Integration).

Usage:
    from precept.domain_strategies import (
        # Base classes
        DomainStrategy,
        BaselineDomainStrategy,
        ParsedTask,
        ActionResult,
        BlackSwanCategory,

        # Concrete strategies
        LogisticsDomainStrategy,
        CodingDomainStrategy,
        DevOpsDomainStrategy,
        FinanceDomainStrategy,
        BookingDomainStrategy,
        IntegrationDomainStrategy,

        # Baseline strategies
        LogisticsBaselineStrategy,
        CodingBaselineStrategy,
        DevOpsBaselineStrategy,
        FinanceBaselineStrategy,
        BookingBaselineStrategy,
        IntegrationBaselineStrategy,
    )
"""

from .base import (
    ActionResult,
    BaselineDomainStrategy,
    BlackSwanCategory,
    DomainStrategy,
    ParsedTask,
    # COMPASS Epistemic Probe Protocol
    ProbeResult,
    ProbeSpec,
)
from .logistics import LogisticsDomainStrategy, LogisticsBaselineStrategy
from .coding import CodingDomainStrategy, CodingBaselineStrategy
from .devops import DevOpsDomainStrategy, DevOpsBaselineStrategy
from .finance import FinanceDomainStrategy, FinanceBaselineStrategy
from .booking import BookingDomainStrategy, BookingBaselineStrategy
from .integration import IntegrationDomainStrategy, IntegrationBaselineStrategy

__all__ = [
    # Base classes
    "BlackSwanCategory",
    "ParsedTask",
    "ActionResult",
    "DomainStrategy",
    "BaselineDomainStrategy",
    # COMPASS Epistemic Probe Protocol
    "ProbeSpec",
    "ProbeResult",
    # Concrete strategies (with learning)
    "LogisticsDomainStrategy",
    "CodingDomainStrategy",
    "DevOpsDomainStrategy",
    "FinanceDomainStrategy",
    "BookingDomainStrategy",
    "IntegrationDomainStrategy",
    # Baseline strategies (NO learning)
    "LogisticsBaselineStrategy",
    "CodingBaselineStrategy",
    "DevOpsBaselineStrategy",
    "FinanceBaselineStrategy",
    "BookingBaselineStrategy",
    "IntegrationBaselineStrategy",
]
