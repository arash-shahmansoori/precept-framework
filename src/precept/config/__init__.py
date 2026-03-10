"""
PRECEPT Configuration Package.

Centralized configuration management for PRECEPT agents and baselines.

This package provides a clean, modular configuration structure:
- paths.py: File and directory path configurations
- llm.py: LLM client configurations
- agent.py: PRECEPT agent behavior configurations
- baseline.py: Baseline agent configurations
- constraints.py: Constraint classification configurations
- prompts.py: Prompt template configurations
- logging.py: Production-ready logging configuration
- unified.py: Unified PreceptConfig combining all configs

Domain-specific configurations (Single Source of Truth):
- logistics.py: Logistics domain configuration (ports, customs, scenarios)
- booking.py: Booking domain configuration (flights, payments, inventory)
- coding.py: Coding domain configuration (packages, crashes, concurrency)
- devops.py: DevOps domain configuration (stacks, IAM, Kubernetes)
- finance.py: Finance domain configuration (symbols, data, compliance)
- integration.py: Integration domain configuration (OAuth, gateways, webhooks)

Usage:
    # Import specific configs
    from precept.config import PreceptConfig, AgentConfig, BaselineConfig

    # Get default configuration
    from precept.config import get_default_config
    config = get_default_config()

    # Setup logging
    from precept.config import setup_logging, get_logger
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    # Domain configs (Single Source of Truth)
    from precept.config import LogisticsConfig, BookingConfig, CodingConfig
    blocked_ports = LogisticsConfig.BLOCKED_PORTS
    blocked_flights = BookingConfig.BLOCKED_FLIGHTS
    blocked_packages = CodingConfig.BLOCKED_PACKAGES
"""

from .agent import AgentConfig
from .baseline import BaselineConfig
from .constraints import ConstraintConfig
from .factual_extraction import (
    FactualExtractionConfig,
    configure_factual_extraction,
    get_factual_extraction_config,
    reset_factual_extraction_config,
)
from .llm import LLMConfig
from .logging import (
    LogConfig,
    get_agent_logger,
    get_baseline_logger,
    get_experiment_logger,
    get_logger,
    get_mcp_logger,
    setup_experiment_logging,
    setup_logging,
    setup_mcp_server_logging,
)
from .booking import BookingConfig, get_booking_config
from .coding import CodingConfig, get_coding_config
from .devops import DevOpsConfig, get_devops_config
from .finance import FinanceConfig, get_finance_config
from .integration import IntegrationConfig, get_integration_config
from .logistics import LogisticsConfig, get_logistics_config
from .paths import (
    DataPaths,
    get_data_dir,
    get_project_root,
    get_server_script,
)
from .prompts import PromptTemplates
from .multi_condition import (
    MultiConditionConfig,
    LogisticsConditions,
    BookingConditions,
    DevOpsConditions,
    FinanceConditions,
    CodingConditions,
    IntegrationConditions,
    generate_multi_condition_scenarios,
)
from .dynamic_static_knowledge import (
    DynamicStaticKnowledgeGenerator,
    generate_dynamic_static_knowledge,
)
from .unified import (
    PreceptConfig,
    create_agent_config,
    create_baseline_config,
    get_config_from_env,
    get_default_config,
)

__all__ = [
    # Path configurations
    "DataPaths",
    "get_project_root",
    "get_data_dir",
    "get_server_script",
    # LLM configurations
    "LLMConfig",
    # Agent configurations
    "AgentConfig",
    # Baseline configurations
    "BaselineConfig",
    # Constraint configurations
    "ConstraintConfig",
    # Factual extraction configurations
    "FactualExtractionConfig",
    "get_factual_extraction_config",
    "configure_factual_extraction",
    "reset_factual_extraction_config",
    # Domain configurations (Single Source of Truth)
    "LogisticsConfig",
    "get_logistics_config",
    "BookingConfig",
    "get_booking_config",
    "CodingConfig",
    "get_coding_config",
    "DevOpsConfig",
    "get_devops_config",
    "FinanceConfig",
    "get_finance_config",
    "IntegrationConfig",
    "get_integration_config",
    # Prompt templates
    "PromptTemplates",
    # Multi-condition configurations
    "MultiConditionConfig",
    "LogisticsConditions",
    "BookingConditions",
    "DevOpsConditions",
    "FinanceConditions",
    "CodingConditions",
    "IntegrationConditions",
    "generate_multi_condition_scenarios",
    # Dynamic static knowledge
    "DynamicStaticKnowledgeGenerator",
    "generate_dynamic_static_knowledge",
    # Logging
    "LogConfig",
    "setup_logging",
    "get_logger",
    "setup_experiment_logging",
    "setup_mcp_server_logging",
    "get_agent_logger",
    "get_baseline_logger",
    "get_mcp_logger",
    "get_experiment_logger",
    # Unified configuration
    "PreceptConfig",
    "get_default_config",
    "get_config_from_env",
    "create_agent_config",
    "create_baseline_config",
]
