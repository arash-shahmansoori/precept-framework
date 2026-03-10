"""
Scenario Generators for PRECEPT Testing.

This module provides scenario generators for all black swan categories,
matching the domain strategies in domain_strategies/.

Available Generators:
- LogisticsScenarioGenerator: Port closures, customs, EDI rejections
- CodingScenarioGenerator: Dependencies, crashes, concurrency, imports
- DevOpsScenarioGenerator: CloudFormation, IAM, Kubernetes
- FinanceScenarioGenerator: Trading, market data, compliance, risk
- BookingScenarioGenerator: Reservations, payments, inventory, pricing
- IntegrationScenarioGenerator: OAuth, gateways, throttling, webhooks

Usage:
    from precept.scenario_generators import (
        LogisticsScenarioGenerator,
        CodingScenarioGenerator,
        DevOpsScenarioGenerator,
        FinanceScenarioGenerator,
        BookingScenarioGenerator,
        IntegrationScenarioGenerator,
        # Or use convenience functions
        generate_logistics_scenarios,
        generate_coding_scenarios,
        generate_devops_scenarios,
        generate_finance_scenarios,
        generate_booking_scenarios,
        generate_integration_scenarios,
    )

    # Using class
    generator = CodingScenarioGenerator()
    scenarios = generator.generate_all()

    # Using convenience function
    scenarios = generate_coding_scenarios()
"""

# Logistics
# Booking
from .booking import BookingScenarioGenerator, generate_booking_scenarios

# Coding
from .coding import CodingScenarioGenerator, generate_coding_scenarios

# DevOps
from .devops import DevOpsScenarioGenerator, generate_devops_scenarios

# Finance
from .finance import FinanceScenarioGenerator, generate_finance_scenarios

# Integration
from .integration import IntegrationScenarioGenerator, generate_integration_scenarios
from .logistics import LogisticsScenarioGenerator, generate_logistics_scenarios

__all__ = [
    # Logistics
    "LogisticsScenarioGenerator",
    "generate_logistics_scenarios",
    # Coding
    "CodingScenarioGenerator",
    "generate_coding_scenarios",
    # DevOps
    "DevOpsScenarioGenerator",
    "generate_devops_scenarios",
    # Finance
    "FinanceScenarioGenerator",
    "generate_finance_scenarios",
    # Booking
    "BookingScenarioGenerator",
    "generate_booking_scenarios",
    # Integration
    "IntegrationScenarioGenerator",
    "generate_integration_scenarios",
]
