"""
PRECEPT: Planning Resilience via Experience, Context Engineering & Probing Trajectories

A unified framework that combines:
- COMPASS (Genetic-Pareto Optimization) for offline prompt optimization ("Compiler")
- Evo-Memory/ReMem for online experience learning ("Runtime")

The framework implements a continuous improvement cycle:
1. Optimization Phase (COMPASS) - Evolve best system prompts
2. Deployment Phase (Evo-Memory) - Operate with ReMem loop
3. Consolidation Phase - Bake frequent memories into prompts

Based on:
- COMPASS/GEPA: "Reflective Prompt Evolution Can Outperform Reinforcement Learning"
- Evo-Memory: "Benchmarking LLM Agent Test-time Learning with Self-Evolving Memory"
  (https://arxiv.org/html/2511.20857v1)
"""

# AutoGen Integration (Scalable Agentic Framework)
from .autogen_integration import (
    # Configuration
    AutoGenPRECEPTConfig,
    # Agent Mixin
    PRECEPTAgentMixin,
    # MCP Adapter
    PRECEPTMCPAdapter,
    # Tool Definitions
    PRECEPTToolDefinitions,
    # Factory Functions
    check_autogen_availability,
    create_precept_autogen_agent,
    create_precept_team,
    get_mcp_server_params,
)
from .baselines import (
    BaselineVectorStore,
    RAGWithToolsBaseline,
    # Baseline Agents
    SimpleRAGBaseline,
    compare_results,
    # Comparison Utilities
    create_baseline_comparison,
)
from .compass_integration import (
    # COMPASS Bridge (merged from compass_bridge.py)
    COMPASSBridge,
    # COMPASS Compilation
    COMPASSCompilationEngine,
    COMPASSDualRetriever,
    COMPASSHardIngestion,
    # COMPASS Hard Ingestion
    COMPASSHardIngestionConfig,
    # COMPASS Dual Retrieval
    DualRetrievalResult,
    # Integrated Agent
    IntegratedPRECEPTAgent,
    create_integrated_agent,
)
from .complexity_analyzer import (
    ComplexityDimension,
    ComplexityEstimate,
    EntityPatternDetector,
    # Multi-Strategy Coordination
    MultiStrategyCoordinator,
    # Complexity Analysis (COMPASS ML generalized)
    PRECEPTComplexityAnalyzer,
    ReasoningPatternDetector,
    RolloutDecision,
    # Smart Rollouts (COMPASS advantage)
    SmartRolloutStrategy,
    # Pattern Detectors
    ToolPatternDetector,
    # Convenience Functions
    analyze_task_complexity,
    decide_rollouts,
    get_complexity_analyzer,
    get_rollout_strategy,
    get_strategy_coordinator,
)
from .context_engineering import (
    # Background Memory Generation (Async Refine)
    BackgroundMemoryWriter,
    CompactionSummary,
    ConflictDetection,
    ConflictType,
    # Master Orchestrator
    ContextEngineeringManager,
    # Irrelevance-based Pruning
    IrrelevancePruner,
    MemoryLoadDecision,
    # Memory Scoping (Application vs User level)
    MemoryScope,
    MemoryScopeManager,
    MemoryWriteJob,
    ProceduralMemoryStore,
    # Procedural Memory (Strategies/Playbooks)
    Procedure,
    # Reactive Retrieval (Memory-as-a-Tool)
    ReactiveRetriever,
    ScopedMemory,
    # Session Compaction (Trajectory Compression)
    SessionCompactor,
    # Smart Consolidation Triggers
    SmartConsolidationTrigger,
    create_context_engineering_manager,
)

# Document Processors (Extensible Hard Ingestion)
from .document_processors import (
    ChunkingStrategy,
    # Base classes
    DocumentProcessor,
    # Chunking strategies
    FixedSizeChunker,
    # LangChain Adapter (RECOMMENDED - uses battle-tested ecosystem)
    LangChainAdapter,
    MarkdownProcessor,
    # Concrete processors (fallback)
    PDFProcessor,
    ProcessingConfig,
    ProcessingResult,
    # Factory and registry (for custom processors)
    ProcessorFactory,
    ProcessorRegistry,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TextProcessor,
    WebScrapingProcessor,
    load_document,
    load_document_sync,
)
from .gepa import (
    GEPAConfig,
    # GEPA Engine
    GEPAEvolutionEngine,
    GEPAMutation,
    GEPAParetoCandidate,
    # GEPA Models
    GEPAReflection,
    create_gepa_engine,
)

# COMPASSBridge merged into compass_integration.py
from .ingestion import (
    DefaultHardIngestionPipeline,
    # Hard Ingestion (Document Stream)
    DocumentChunk,
    # Feedback Ingestion (Training Stream)
    ExecutionTrace,
    FeedbackIngestionManager,
    HardIngestionPipeline,
    IngestionPriority,
    # Ingestion Types
    IngestionType,
    # Unified Coordinator
    PRECEPTIngestionCoordinator,
    SoftIngestionManager,
    SoftIngestionResult,
    # Soft Ingestion (Experience Stream)
    SoftPatch,
)
from .llm_clients import (
    # Availability flags
    ANTHROPIC_AVAILABLE,
    GEMINI_AVAILABLE,
    GROQ_AVAILABLE,
    TOGETHER_AVAILABLE,
    # Availability check
    check_api_availability,
    create_openai_embeddings,
    # LangChain-compatible embedding factory (from models/embedding_models)
    get_openai_embedding_model,
    get_precept_embedding_fn,
    # Factory functions
    get_precept_llm_client,
    # Alternative LLM providers (from models/ directory)
    precept_anthropic_client,
    precept_embed_documents,
    # Embedding functions (OpenAI - REAL API)
    precept_embedding_fn,
    precept_gemini_client,
    precept_groq_client,
    # Primary LLM client (OpenAI - REAL API)
    precept_llm_client,
    precept_llm_client_with_context,
    precept_together_client,
)
from .memory_consolidation import (
    ConsolidationResult,
    FrequencyAnalyzer,
    MemoryConsolidator,
)
from .memory_store import (
    EpisodicMemory,
    Experience,
    ExperienceType,
    MemoryPriority,
    MemoryStore,
    SemanticMemoryIndex,
)
from .pareto_memory import (
    ParetoMemoryManager,
    PromptVersion,
    TaskType,
    TaskTypeRouter,
)
from .precept_orchestrator import (
    PRECEPTConfig,
    PRECEPTOrchestrator,
    PRECEPTPhase,
    create_precept_agent,
)
from .remem_pipeline import (
    ReMem,
    ReMemAction,
    ReMemState,
    ThinkActRefineLoop,
)
from .scoring import (
    GEPAEvaluationResult,
    # GEPA-compliant Scoring
    GEPAObjective,
    GEPARolloutExecutor,
    RolloutResult,
    compute_gepa_scores,
    compute_scores_from_task_results,
    pareto_select,
    update_pareto_front,
)
from .simulation import (
    # Simulation World
    BlackSwanWorld,
    # HARD MODE: Cryptic errors, no hints (realistic testing)
    HardModeBlackSwanWorld,
    RuleType,
    SimulationResult,
    SimulationRule,
    create_hard_mode_tool_executor,
    # Tool Executor Factory
    create_logistics_tool_executor,
    # Extended Scenarios (15 learning + 5 test)
    get_extended_learning_scenarios,
    get_extended_test_scenarios,
    get_hard_black_swan_scenarios,
    # Test Scenarios (RIGOROUS - NO HINTS)
    get_learning_scenarios,
    # Knowledge Base
    get_logistics_knowledge_base,
    get_post_learning_scenarios,
)

# MCP Server/Client (Full Protocol Implementation)
# Try to import MCP components - they require the mcp library
try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

# PRECEPT MCP Client (connects to server via MCP protocol)
# Abstract base class + category-specific clients for all black swan types
try:
    from .precept_mcp_client import (
        # Abstract Base Class (use this for custom domains)
        AbstractPRECEPTMCPClient,
        BookingMCPClient,
        CodingMCPClient,
        DevOpsMCPClient,
        FinanceMCPClient,
        IntegrationMCPClient,
        # Category-specific clients (from black_swan_gen.py categories)
        LogisticsMCPClient,
        # Backward compatibility alias
        PRECEPTMCPClient,  # = LogisticsMCPClient
        call_precept_tool,
        # Utility functions
        get_precept_server_params,
        list_precept_tools,
    )

    PRECEPT_MCP_CLIENT_AVAILABLE = True
except ImportError:
    PRECEPT_MCP_CLIENT_AVAILABLE = False
    AbstractPRECEPTMCPClient = None
    PRECEPTMCPClient = None

# Deep MCP Integration into ReMem (Google Whitepaper: Memory-as-a-Tool)
# AutoGen PRECEPT Agent (Strategy Pattern with LLM Reasoning)
# COMPASS-Enhanced MCP Client
from .compass_mcp_client import PRECEPTMCPClientWithCOMPASS

# COMPASS Controller (System 2 Executive Function)
from .compass_controller import (
    COMPASSAction,
    COMPASSConfig,
    COMPASSController,
    COMPASSDecision,
    create_compass_controller,
)

# Domain Strategy Pattern (Pluggable Black Swan Handling)
from .domain_strategies import (
    ActionResult,
    BaselineDomainStrategy,
    # Base classes
    BlackSwanCategory,
    BookingBaselineStrategy,
    BookingDomainStrategy,
    CodingBaselineStrategy,
    CodingDomainStrategy,
    DevOpsBaselineStrategy,
    DevOpsDomainStrategy,
    DomainStrategy,
    FinanceBaselineStrategy,
    FinanceDomainStrategy,
    IntegrationBaselineStrategy,
    IntegrationDomainStrategy,
    # Baseline strategies (NO learning)
    LogisticsBaselineStrategy,
    # Concrete strategies (with learning)
    LogisticsDomainStrategy,
    ParsedTask,
)
from .mcp_remem_integration import (
    # Drop-in Replacement for ReMem
    MCPReMem,
    # State
    MCPReMemPhase,
    MCPReMemState,
    # Loop (Deep MCP Integration)
    MCPThinkActRefineLoop,
    # Tool Actions
    MCPToolCall,
    # Tool Registry
    MCPToolCategory,
    MCPToolDefinition,
    MCPToolRegistry,
    MCPToolResult,
)
from .precept_agent import PRECEPTAgent

# Dynamic Rule Parser (Domain-Agnostic Learning)
from .rule_parser import (
    DynamicRuleParser,
    ParsedRule,
)

# Strategy Registry (Factory Functions)
from .strategy_registry import (
    BASELINE_STRATEGIES,
    DOMAIN_STRATEGIES,
    get_baseline_strategy,
    get_baseline_strategy_class,
    get_domain_strategy,
    get_domain_strategy_class,
    list_available_domains,
)

# Code Execution Components (Docker-based sandboxed execution)
try:
    from .code_executor import CodeExecutionManager, ExecutionResult
    from .dynamic_coding_config import DynamicCodingConfig, ExecutionRecord
    from .execution_feedback_processor import (
        ErrorCategory,
        ErrorSeverity,
        ExecutionFeedbackProcessor,
        ProcessedFeedback,
    )
    from .execution_feedback_processor import (
        Warning as ExecutionWarning,
    )
    from .multiturn_docker_agent import ConversationTurn, MultiTurnDockerAgent
    from .single_turn_docker_agent import SingleTurnDockerAgent

    CODE_EXECUTION_AVAILABLE = True
except ImportError:
    CODE_EXECUTION_AVAILABLE = False
    CodeExecutionManager = None
    ExecutionResult = None
    ExecutionFeedbackProcessor = None
    ProcessedFeedback = None
    ErrorCategory = None
    ErrorSeverity = None
    ExecutionWarning = None
    DynamicCodingConfig = None
    ExecutionRecord = None
    MultiTurnDockerAgent = None
    ConversationTurn = None
    SingleTurnDockerAgent = None

# Configuration Module (Centralized Settings)
# Agent Functions (Pure Functions for Testability)
from .agent_functions import (
    ContextFetchResult,
    # Data classes
    LLMSuggestion,
    TaskResult,
    build_agent_stats,
    build_baseline_prompt,
    build_full_reflexion_prompt,
    # Prompt building
    build_reasoning_prompt,
    build_reflexion_prompt,
    compute_average,
    # Statistics
    compute_success_rate,
    fetch_context,
    # Async helpers
    parallel_fetch,
    # Parsing functions
    parse_llm_response,
    parse_reflexion_response,
    # Task execution
    record_error_and_add_constraint,
    record_successful_solution,
    store_experience_and_trigger_learning,
)

# Baseline Agents (Fair Comparison)
from .baseline_agents import (
    ExpeL_BaselineAgent,
    FullReflexionBaselineAgent,
    LLMBaselineAgent,
    ReflexionBaselineAgent,
)

# Baseline Functions (Pure Functions for Baseline Agents)
from .baseline_functions import (
    add_reflection,
    # Prompt building
    build_baseline_llm_prompt,
    # Statistics
    build_baseline_stats,
    build_current_episode_context,
    # Context building
    build_error_context,
    build_full_reflexion_llm_prompt,
    build_reflection_section,
    build_reflexion_llm_prompt,
    clear_reflection_memory,
    create_reflection_record,
    format_accumulated_reflections,
    # Reflection memory
    get_reflection_memory,
    # Response parsing
    parse_baseline_llm_response,
    # ExpeL functions
    add_expel_insight,
    build_expel_insight_extraction_prompt,
    build_expel_task_prompt,
    clear_expel_insights,
    extract_conditions_from_task,
    get_expel_insights,
    get_expel_stats,
    parse_expel_insight_response,
    parse_expel_task_response,
    retrieve_expel_insights_by_conditions,
)
from .baseline_functions import (
    get_memory_stats as get_baseline_memory_stats,
)
from .config import (
    AgentConfig,
    BaselineConfig,
    ConstraintConfig,
    DataPaths,
    LLMConfig,
    # Main configuration classes
    PreceptConfig,
    PromptTemplates,
    create_agent_config,
    create_baseline_config,
    get_config_from_env,
    get_data_dir,
    # Factory functions
    get_default_config,
    # Path utilities
    get_project_root,
    get_server_script,
)
from .constraints import (
    Constraint as PruningConstraint,
)

# Constraint Classification (Deterministic Pruning)
from .constraints import (
    # Types
    ConstraintType as PruningConstraintType,
)
from .constraints import (
    RefineInterceptor as PruningRefineInterceptor,
)
from .constraints import (
    # Pure functions
    classify_error,
    create_constraint,
    create_refine_interceptor,
    format_forbidden_injection,
    get_remaining_options,
    suggest_diagnostic_probe,
)

# CSP Constraint Manager (Black Swan CSP Solving)
from .csp_constraint_manager import (
    CONSTRAINT_CLUSTERS,
    CausalChain,
    CausalChainTracker,
    ConflictResolution,
    ConflictResolver,
    Constraint,
    ConstraintTier,
    ConstraintType,
    CSPConstraintManager,
    ExecutionFeedback,
    RefineInterceptor,
    create_csp_scenario,
)

# Scenario Generators (all 6 domains)
from .scenario_generators import (
    # Booking
    BookingScenarioGenerator,
    # Coding
    CodingScenarioGenerator,
    # DevOps
    DevOpsScenarioGenerator,
    # Finance
    FinanceScenarioGenerator,
    # Integration
    IntegrationScenarioGenerator,
    # Logistics
    LogisticsScenarioGenerator,
    generate_booking_scenarios,
    generate_coding_scenarios,
    generate_devops_scenarios,
    generate_finance_scenarios,
    generate_integration_scenarios,
    generate_logistics_scenarios,
)

# Execution Tracing (Detailed Logging for Experiments)
from .execution_tracer import (
    ExecutionLog,
    ExecutionTracer,
    TaskTrace,
    TraceEvent,
    trace_compass_decision,
    trace_error_recovery,
    trace_execute_action,
    trace_fetch_context,
    trace_learning,
    trace_llm_reasoning,
    trace_parse_task,
    trace_probe_execution,
    trace_rule_applied,
)

__all__ = [
    # Memory Store
    "EpisodicMemory",
    "Experience",
    "ExperienceType",
    "MemoryPriority",
    "MemoryStore",
    "SemanticMemoryIndex",
    # ReMem Pipeline
    "ReMem",
    "ReMemAction",
    "ReMemState",
    "ThinkActRefineLoop",
    # Memory Consolidation
    "MemoryConsolidator",
    "ConsolidationResult",
    "FrequencyAnalyzer",
    # Pareto Memory
    "ParetoMemoryManager",
    "PromptVersion",
    "TaskType",
    "TaskTypeRouter",
    # Orchestrator
    "PRECEPTOrchestrator",
    "PRECEPTConfig",
    "PRECEPTPhase",
    "create_precept_agent",
    # COMPASS Bridge
    "COMPASSBridge",
    "create_integrated_agent",
    # Ingestion (Three-Stream Architecture)
    "IngestionType",
    "IngestionPriority",
    # Hard Ingestion (Document Stream)
    "DocumentChunk",
    "HardIngestionPipeline",
    "DefaultHardIngestionPipeline",
    # Soft Ingestion (Experience Stream)
    "SoftPatch",
    "SoftIngestionResult",
    "SoftIngestionManager",
    # Feedback Ingestion (Training Stream)
    "ExecutionTrace",
    "FeedbackIngestionManager",
    # Unified Coordinator
    "PRECEPTIngestionCoordinator",
    # Document Processors (Extensible Hard Ingestion)
    # LangChain Adapter (RECOMMENDED)
    "LangChainAdapter",
    "load_document",
    "load_document_sync",
    # Base classes
    "DocumentProcessor",
    "ProcessingConfig",
    "ProcessingResult",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "SemanticChunker",
    # Concrete processors (fallback)
    "PDFProcessor",
    "TextProcessor",
    "WebScrapingProcessor",
    "MarkdownProcessor",
    # Factory and registry
    "ProcessorFactory",
    "ProcessorRegistry",
    # COMPASS Integration (Production Infrastructure)
    "COMPASSHardIngestionConfig",
    "COMPASSHardIngestion",
    "DualRetrievalResult",
    "COMPASSDualRetriever",
    "COMPASSCompilationEngine",
    "IntegratedPRECEPTAgent",
    # LLM Clients (REAL API - NO MOCKS)
    "precept_llm_client",
    "precept_llm_client_with_context",
    "precept_embedding_fn",
    "precept_embed_documents",
    # LangChain-compatible embedding object (for vector stores)
    "get_openai_embedding_model",
    "get_precept_llm_client",
    "get_precept_embedding_fn",
    "create_openai_embeddings",
    # Alternative LLM Providers (from models/ directory)
    "precept_anthropic_client",
    "precept_gemini_client",
    "precept_groq_client",
    "precept_together_client",
    "ANTHROPIC_AVAILABLE",
    "GEMINI_AVAILABLE",
    "GROQ_AVAILABLE",
    "TOGETHER_AVAILABLE",
    "check_api_availability",
    # GEPA (Genetic-Pareto Evolution)
    "GEPAReflection",
    "GEPAMutation",
    "GEPAParetoCandidate",
    "GEPAConfig",
    "GEPAEvolutionEngine",
    "create_gepa_engine",
    # GEPA-compliant Scoring (from scoring.py)
    "GEPAObjective",
    "RolloutResult",
    "GEPAEvaluationResult",
    "compute_gepa_scores",
    "compute_scores_from_task_results",
    "pareto_select",
    "update_pareto_front",
    "GEPARolloutExecutor",
    # Baselines for Comparison
    "SimpleRAGBaseline",
    "RAGWithToolsBaseline",
    "BaselineVectorStore",
    "create_baseline_comparison",
    "compare_results",
    # Simulation Framework
    "BlackSwanWorld",
    "SimulationRule",
    "SimulationResult",
    "RuleType",
    "create_logistics_tool_executor",
    "get_logistics_knowledge_base",
    "get_learning_scenarios",
    "get_post_learning_scenarios",
    "get_hard_black_swan_scenarios",
    # Complexity Analysis (COMPASS ML Generalized)
    "PRECEPTComplexityAnalyzer",
    "ComplexityEstimate",
    "ComplexityDimension",
    # Smart Rollouts (COMPASS Advantage)
    "SmartRolloutStrategy",
    "RolloutDecision",
    # Multi-Strategy Coordination
    "MultiStrategyCoordinator",
    # Pattern Detectors
    "ToolPatternDetector",
    "ReasoningPatternDetector",
    "EntityPatternDetector",
    # Convenience Functions
    "analyze_task_complexity",
    "decide_rollouts",
    "get_complexity_analyzer",
    "get_rollout_strategy",
    "get_strategy_coordinator",
    # Context Engineering (Google Whitepaper Patterns)
    # Memory Scoping
    "MemoryScope",
    "ScopedMemory",
    "MemoryScopeManager",
    # Procedural Memory
    "Procedure",
    "ProceduralMemoryStore",
    # Session Compaction
    "SessionCompactor",
    "CompactionSummary",
    # Reactive Retrieval
    "ReactiveRetriever",
    "MemoryLoadDecision",
    # Background Memory Writer
    "BackgroundMemoryWriter",
    "MemoryWriteJob",
    # Smart Consolidation Triggers
    "SmartConsolidationTrigger",
    "ConflictType",
    "ConflictDetection",
    # Irrelevance-based Pruning
    "IrrelevancePruner",
    # Master Orchestrator
    "ContextEngineeringManager",
    "create_context_engineering_manager",
    # AutoGen Integration (Scalable Agentic Framework)
    "AutoGenPRECEPTConfig",
    "PRECEPTToolDefinitions",
    "PRECEPTAgentMixin",
    "PRECEPTMCPAdapter",
    "check_autogen_availability",
    "create_precept_autogen_agent",
    "create_precept_team",
    "get_mcp_server_params",
    # MCP Server/Client (Full Protocol)
    "MCP_AVAILABLE",
    "PRECEPT_MCP_CLIENT_AVAILABLE",
    # Abstract base class
    "AbstractPRECEPTMCPClient",
    # Category-specific MCP clients (from black_swan_gen.py)
    "LogisticsMCPClient",
    "CodingMCPClient",
    "DevOpsMCPClient",
    "FinanceMCPClient",
    "BookingMCPClient",
    "IntegrationMCPClient",
    # Backward compatibility
    "PRECEPTMCPClient",
    # Utilities
    "get_precept_server_params",
    "call_precept_tool",
    "list_precept_tools",
    # Deep MCP Integration (Google Whitepaper: Memory-as-a-Tool)
    "MCPToolCategory",
    "MCPToolDefinition",
    "MCPToolRegistry",
    "MCPToolCall",
    "MCPToolResult",
    "MCPReMemPhase",
    "MCPReMemState",
    "MCPThinkActRefineLoop",
    "MCPReMem",
    # Domain Strategy Pattern (Pluggable Black Swan Handling)
    "BlackSwanCategory",
    "ParsedTask",
    "ActionResult",
    "DomainStrategy",
    "BaselineDomainStrategy",
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
    # Dynamic Rule Parser
    "ParsedRule",
    "DynamicRuleParser",
    # Strategy Registry
    "DOMAIN_STRATEGIES",
    "BASELINE_STRATEGIES",
    "get_domain_strategy",
    "get_baseline_strategy",
    "list_available_domains",
    "get_domain_strategy_class",
    "get_baseline_strategy_class",
    # COMPASS-Enhanced MCP Client
    "PRECEPTMCPClientWithCOMPASS",
    # COMPASS Controller (System 2 Executive)
    "COMPASSController",
    "COMPASSConfig",
    "COMPASSAction",
    "COMPASSDecision",
    "create_compass_controller",
    # AutoGen PRECEPT Agent
    "PRECEPTAgent",
    # Baseline Agents
    "LLMBaselineAgent",
    "ReflexionBaselineAgent",
    "FullReflexionBaselineAgent",
    "ExpeL_BaselineAgent",
    # ExpeL Functions
    "add_expel_insight",
    "build_expel_insight_extraction_prompt",
    "build_expel_task_prompt",
    "clear_expel_insights",
    "extract_conditions_from_task",
    "get_expel_insights",
    "get_expel_stats",
    "parse_expel_insight_response",
    "parse_expel_task_response",
    "retrieve_expel_insights_by_conditions",
    # Configuration Module
    "PreceptConfig",
    "AgentConfig",
    "BaselineConfig",
    "LLMConfig",
    "DataPaths",
    "PromptTemplates",
    "ConstraintConfig",
    "get_default_config",
    "get_config_from_env",
    "create_agent_config",
    "create_baseline_config",
    "get_project_root",
    "get_data_dir",
    "get_server_script",
    # Constraint Classification (Pruning)
    "PruningConstraintType",
    "PruningConstraint",
    "PruningRefineInterceptor",
    "classify_error",
    "create_constraint",
    "format_forbidden_injection",
    "get_remaining_options",
    "suggest_diagnostic_probe",
    "create_refine_interceptor",
    # Agent Functions
    "LLMSuggestion",
    "TaskResult",
    "ContextFetchResult",
    "parse_llm_response",
    "parse_reflexion_response",
    "build_reasoning_prompt",
    "build_baseline_prompt",
    "build_reflexion_prompt",
    "build_full_reflexion_prompt",
    "parallel_fetch",
    "fetch_context",
    "record_error_and_add_constraint",
    "record_successful_solution",
    "store_experience_and_trigger_learning",
    "compute_success_rate",
    "compute_average",
    "build_agent_stats",
    # Baseline Functions
    "get_reflection_memory",
    "add_reflection",
    "clear_reflection_memory",
    "get_baseline_memory_stats",
    "parse_baseline_llm_response",
    "build_error_context",
    "build_reflection_section",
    "format_accumulated_reflections",
    "build_current_episode_context",
    "build_baseline_llm_prompt",
    "build_reflexion_llm_prompt",
    "build_full_reflexion_llm_prompt",
    "build_baseline_stats",
    "create_reflection_record",
    # CSP Constraint Manager
    "CSPConstraintManager",
    "Constraint",
    "ConstraintTier",
    "ConstraintType",
    "CausalChainTracker",
    "CausalChain",
    "RefineInterceptor",
    "ExecutionFeedback",
    "ConflictResolver",
    "ConflictResolution",
    "CONSTRAINT_CLUSTERS",
    "create_csp_scenario",
    # Code Execution Components (Docker-based sandboxed execution)
    "CODE_EXECUTION_AVAILABLE",
    "CodeExecutionManager",
    "ExecutionResult",
    "ExecutionFeedbackProcessor",
    "ProcessedFeedback",
    "ErrorCategory",
    "ErrorSeverity",
    "ExecutionWarning",
    "DynamicCodingConfig",
    "ExecutionRecord",
    # Multi-Turn Docker Agent (combines multi-turn + Docker execution + learning)
    "MultiTurnDockerAgent",
    "ConversationTurn",
    # Single-Turn Docker Agent (single-turn + Docker execution + learning)
    "SingleTurnDockerAgent",
    # Scenario Generators (all 6 domains)
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
    # Execution Tracing (Detailed Logging for Experiments)
    "ExecutionTracer",
    "ExecutionLog",
    "TaskTrace",
    "TraceEvent",
    "trace_parse_task",
    "trace_fetch_context",
    "trace_compass_decision",
    "trace_llm_reasoning",
    "trace_execute_action",
    "trace_error_recovery",
    "trace_probe_execution",
    "trace_rule_applied",
    "trace_learning",
]
