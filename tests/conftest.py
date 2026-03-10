"""
Pytest Configuration and Shared Fixtures for PRECEPT Tests.

This module provides:
- Common fixtures for unit and integration tests
- Mock objects for MCP clients and LLM responses
- Test data factories for scenarios and tasks
- Configuration fixtures for different test scenarios
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# =============================================================================
# PATH SETUP - Ensure precept is importable
# =============================================================================

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def src_path(project_root: Path) -> Path:
    """Get src/precept directory."""
    return project_root / "src" / "precept"


@pytest.fixture
def data_path(project_root: Path) -> Path:
    """Get data directory."""
    return project_root / "data"


# =============================================================================
# MOCK PARSED TASK
# =============================================================================


@dataclass
class MockParsedTask:
    """Mock ParsedTask for testing."""

    action: str = "book_shipment"
    entity: str = "shipment"
    source: Optional[str] = "rotterdam"
    target: Optional[str] = "boston"
    parameters: Dict[str, Any] = None
    task_type: str = "booking"

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@pytest.fixture
def mock_parsed_task() -> MockParsedTask:
    """Create a mock parsed task."""
    return MockParsedTask()


@pytest.fixture
def mock_customs_task() -> MockParsedTask:
    """Create a mock customs task."""
    return MockParsedTask(
        action="clear_customs",
        entity="customs",
        source="shanghai",
        target="new_york",
        parameters={"is_customs": True},
        task_type="customs",
    )


# =============================================================================
# MOCK MCP CLIENT
# =============================================================================


@pytest.fixture
def mock_mcp_client() -> MagicMock:
    """Create a mock MCP client with all required methods."""
    client = MagicMock()

    # Async methods
    client.retrieve_memories = AsyncMock(
        return_value="Previous experience: Rotterdam blocked, use Antwerp"
    )
    client.get_procedure = AsyncMock(return_value="No procedure found")
    client.get_learned_rules = AsyncMock(
        return_value="R-482: Rotterdam blocked → Use Antwerp"
    )
    client.record_error = AsyncMock(return_value="Error recorded")
    client.record_solution = AsyncMock(return_value="Solution recorded")
    client.store_experience = AsyncMock(return_value="Experience stored")
    client.trigger_consolidation = AsyncMock(return_value="Consolidation triggered")
    client.trigger_compass_evolution = AsyncMock(return_value="Evolution triggered")
    client.get_evolved_prompt = AsyncMock(return_value="Evolved system prompt...")
    client.update_memory_usefulness = AsyncMock(return_value="Usefulness updated")

    # Sync methods
    client.analyze_complexity = MagicMock(return_value="medium")

    return client


# =============================================================================
# MOCK LLM CLIENT
# =============================================================================


@dataclass
class MockLLMResponse:
    """Mock LLM response object."""

    content: str = (
        "SOLUTION: Antwerp\nREASONING: Rotterdam is blocked\nCONFIDENCE: high"
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Create a mock LLM client."""
    client = MagicMock()
    client.create = AsyncMock(return_value=MockLLMResponse())
    return client


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Create a mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "SOLUTION: Antwerp\nREASONING: Rotterdam is blocked\nCONFIDENCE: high"
                }
            }
        ]
    }


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def agent_config():
    """Create a test AgentConfig."""
    from precept.config import AgentConfig

    return AgentConfig(
        max_attempts=3,
        consolidation_interval=2,
        compass_evolution_interval=2,
        failure_threshold=2,
        enable_llm_reasoning=True,
        force_llm_reasoning=False,
        verbose_llm=False,
        max_internal_workers=2,
        max_pivots=2,
    )


@pytest.fixture
def baseline_config():
    """Create a test BaselineConfig."""
    from precept.config import BaselineConfig

    return BaselineConfig(
        model="gpt-4o-mini",
        max_attempts=3,
        temperature=0.7,
        max_tokens=200,
        verbose=False,
        max_internal_workers=2,
    )


@pytest.fixture
def llm_config():
    """Create a test LLMConfig."""
    from precept.config import LLMConfig

    return LLMConfig(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=500,
    )


@pytest.fixture
def precept_config(agent_config, baseline_config, llm_config):
    """Create a full PreceptConfig."""
    from precept.config import PreceptConfig

    config = PreceptConfig()
    config.agent = agent_config
    config.baseline = baseline_config
    config.llm = llm_config
    return config


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_task() -> str:
    """Sample task string."""
    return "Book shipment from Rotterdam to Boston"


@pytest.fixture
def sample_memories() -> str:
    """Sample retrieved memories."""
    return """
    Memory 1: Rotterdam port closure (R-482) - Use Antwerp instead
    Memory 2: Boston customs requires COO documentation
    Memory 3: Hamburg to US destinations blocked (H-903)
    """


@pytest.fixture
def sample_rules() -> str:
    """Sample learned rules."""
    return """
    R-482: Rotterdam blocked → Use Antwerp
    H-903: Hamburg to US blocked → Use Antwerp/Rotterdam
    CUSTOMS-COO-001: Missing COO → Obtain from manufacturer
    """


@pytest.fixture
def sample_options() -> List[str]:
    """Sample available options."""
    return ["rotterdam", "hamburg", "antwerp", "shanghai", "ningbo"]


@pytest.fixture
def llm_response_with_solution() -> str:
    """Sample LLM response with solution."""
    return """
    SOLUTION: Antwerp
    REASONING: Rotterdam is blocked due to R-482, Antwerp is the best alternative
    CONFIDENCE: high
    """


@pytest.fixture
def llm_response_explore() -> str:
    """Sample LLM response suggesting exploration."""
    return """
    SOLUTION: EXPLORE
    REASONING: Need more information before deciding
    CONFIDENCE: low
    """


@pytest.fixture
def reflexion_response() -> str:
    """Sample Reflexion-style response."""
    return """
    REFLECTION: Rotterdam failed due to port closure. The error code R-482 indicates
    the port is blocked for all traffic.
    LESSON: Always check port status before booking. Alternative ports like Antwerp
    should be tried for Rotterdam routes.
    SOLUTION: Antwerp
    REASONING: Antwerp is the nearest alternative to Rotterdam
    CONFIDENCE: high
    """


# =============================================================================
# MOCK DOMAIN STRATEGY
# =============================================================================


@pytest.fixture
def mock_domain_strategy() -> MagicMock:
    """Create a mock domain strategy."""
    strategy = MagicMock()
    strategy.domain_name = "logistics"
    strategy.category = MagicMock(value="LOGISTICS")

    strategy.parse_task = MagicMock(
        return_value=MockParsedTask(
            action="book_shipment",
            entity="shipment",
            source="rotterdam",
            target="boston",
            parameters={},
        )
    )
    strategy.get_available_actions = MagicMock(
        return_value=["book_shipment", "check_port", "clear_customs"]
    )
    strategy.get_options_for_task = MagicMock(
        return_value=["rotterdam", "hamburg", "antwerp"]
    )
    strategy.get_system_prompt = MagicMock(
        return_value="You are a logistics assistant..."
    )
    strategy.create_autogen_tools = MagicMock(return_value=[])

    # Mock execute_action
    action_result = MagicMock()
    action_result.success = True
    action_result.response = "Shipment booked successfully"
    action_result.strategy_used = "direct"
    action_result.error_code = None
    strategy.execute_action = AsyncMock(return_value=action_result)

    return strategy


@pytest.fixture
def mock_baseline_strategy() -> MagicMock:
    """Create a mock baseline strategy."""
    strategy = MagicMock()
    strategy.domain_name = "logistics"
    strategy.parse_task = MagicMock(
        return_value=MockParsedTask(
            action="book_shipment",
            entity="shipment",
            source="rotterdam",
            target="boston",
        )
    )
    strategy.get_options_for_task = MagicMock(
        return_value=["rotterdam", "hamburg", "antwerp"]
    )
    strategy.execute_action = AsyncMock(return_value=(True, "Success"))

    return strategy


# =============================================================================
# CONSTRAINT FIXTURES
# =============================================================================


@pytest.fixture
def mock_refine_interceptor():
    """Create a mock RefineInterceptor."""
    from precept.constraints import create_refine_interceptor

    return create_refine_interceptor()


# =============================================================================
# ASYNC TEST HELPERS
# =============================================================================


@pytest.fixture
def async_mock():
    """Factory for creating async mocks."""

    def _create_async_mock(return_value=None):
        return AsyncMock(return_value=return_value)

    return _create_async_mock


# =============================================================================
# CONFLICT RESOLUTION FIXTURES
# =============================================================================


@pytest.fixture
def conflict_resolution_config():
    """Create a test ConflictResolutionConfig."""
    from precept.conflict_resolution import ConflictResolutionConfig

    return ConflictResolutionConfig(
        similarity_threshold=0.7,
        contradiction_confidence_threshold=0.8,
        high_confidence_threshold=0.85,
        recency_weight=0.4,
        reliability_weight=0.3,
        specificity_weight=0.3,
    )


@pytest.fixture
def static_knowledge_item():
    """Create a sample static knowledge item."""
    from datetime import datetime

    from precept.conflict_resolution import KnowledgeItem, KnowledgeSource

    return KnowledgeItem(
        id="static-rotterdam-001",
        content="Rotterdam port operates 24/7 with capacity for mega-container vessels",
        source=KnowledgeSource.STATIC_KB,
        timestamp=datetime(2023, 6, 1),
        confidence=0.95,
        metadata={"port": "rotterdam", "type": "port_info"},
    )


@pytest.fixture
def dynamic_knowledge_item():
    """Create a sample dynamic knowledge item."""
    from datetime import datetime

    from precept.conflict_resolution import KnowledgeItem, KnowledgeSource

    return KnowledgeItem(
        id="dynamic-rotterdam-001",
        content="Rotterdam blocked due to R-482 error, use Hamburg or Antwerp",
        source=KnowledgeSource.DYNAMIC_EXPERIENCE,
        timestamp=datetime.now(),
        confidence=0.85,
        metadata={"error_code": "R-482"},
    )


@pytest.fixture
def conflicting_knowledge_pair():
    """Create a pair of conflicting knowledge items."""
    from datetime import datetime

    from precept.conflict_resolution import KnowledgeItem, KnowledgeSource

    static = KnowledgeItem(
        id="static-hamburg-001",
        content="Hamburg port labor situation is stable",
        source=KnowledgeSource.STATIC_KB,
        timestamp=datetime(2023, 1, 1),
        confidence=0.9,
    )
    dynamic = KnowledgeItem(
        id="dynamic-hamburg-001",
        content="Hamburg experiencing severe labor strikes, all bulk cargo delayed",
        source=KnowledgeSource.DYNAMIC_EXPERIENCE,
        timestamp=datetime.now(),
        confidence=0.88,
    )
    return static, dynamic


@pytest.fixture
def agreement_knowledge_pair():
    """Create a pair of agreeing knowledge items."""
    from datetime import datetime

    from precept.conflict_resolution import KnowledgeItem, KnowledgeSource

    static = KnowledgeItem(
        id="static-europe-ports-001",
        content="Rotterdam and Hamburg are alternative ports in Europe",
        source=KnowledgeSource.STATIC_KB,
        timestamp=datetime(2023, 1, 1),
        confidence=0.95,
    )
    dynamic = KnowledgeItem(
        id="dynamic-fallback-001",
        content="When Rotterdam blocked, Hamburg works as reliable alternative",
        source=KnowledgeSource.DYNAMIC_EXPERIENCE,
        timestamp=datetime.now(),
        confidence=0.9,
    )
    return static, dynamic


@pytest.fixture
def mock_conflict_manager():
    """Create a mock ConflictManager."""
    from unittest.mock import AsyncMock, MagicMock

    manager = MagicMock()
    manager.detector = MagicMock()
    manager.resolver = MagicMock()
    manager.conflict_history = []
    manager.resolution_history = []

    manager.get_stats = MagicMock(
        return_value={
            "detector": {"conflicts_detected": 0},
            "resolver": {"conflicts_resolved": 0},
            "conflict_history_size": 0,
            "resolution_history_size": 0,
        }
    )

    manager.check_and_resolve = AsyncMock(return_value=([], []))
    manager.detect_and_resolve = AsyncMock(return_value=None)
    manager.get_recent_conflicts = MagicMock(return_value=[])

    return manager


# =============================================================================
# DUAL RETRIEVAL FIXTURES
# =============================================================================


@pytest.fixture
def mock_mcp_client_with_dual_retrieval() -> MagicMock:
    """Create a mock MCP client with dual retrieval support."""
    client = MagicMock()

    # Standard methods
    client.retrieve_memories = AsyncMock(
        return_value="Previous experience: Rotterdam blocked, use Antwerp"
    )
    client.get_procedure = AsyncMock(return_value="No procedure found")
    client.get_learned_rules = AsyncMock(
        return_value="R-482: Rotterdam blocked → Use Antwerp"
    )
    client.record_error = AsyncMock(return_value="Error recorded")
    client.record_solution = AsyncMock(return_value="Solution recorded")
    client.store_experience = AsyncMock(return_value="Experience stored")
    client.trigger_consolidation = AsyncMock(return_value="Consolidation triggered")
    client.trigger_compass_evolution = AsyncMock(return_value="Evolution triggered")
    client.get_evolved_prompt = AsyncMock(return_value="Evolved system prompt...")

    # Dual retrieval method
    client.retrieve_with_dual_mode = AsyncMock(
        return_value="""
=== STATIC KNOWLEDGE BASE ===
Rotterdam port operates 24/7 with capacity for mega-container vessels.

=== DYNAMIC KNOWLEDGE (Patches) ===
R-482: Rotterdam blocked → Use Hamburg or Antwerp

=== EPISODIC MEMORY ===
Previous task: Rotterdam booking failed, pivoted to Antwerp successfully.

⚠️ No conflicts detected between static and dynamic knowledge.
"""
    )

    # Static knowledge methods
    client.ingest_static_knowledge = AsyncMock(
        return_value="✓ Ingested 16 chunks into static knowledge base"
    )
    client.get_static_knowledge_stats = AsyncMock(
        return_value="Static Knowledge Base: 16 documents"
    )

    # Conflict resolution stats
    client.call_tool = AsyncMock(
        return_value='{"status": "active", "summary": {"conflicts_detected": 0}}'
    )

    return client


@pytest.fixture
def sample_static_knowledge() -> List[Dict[str, Any]]:
    """Sample static knowledge for testing."""
    return [
        {
            "content": "Rotterdam port operates 24/7 with capacity for mega-container vessels",
            "metadata": {"port": "rotterdam", "type": "port_info"},
        },
        {
            "content": "Hamburg port has extensive rail connections",
            "metadata": {"port": "hamburg", "type": "port_info"},
        },
        {
            "content": "For customs clearance, attach certificate of origin",
            "metadata": {"type": "procedure", "category": "customs"},
        },
    ]


@pytest.fixture
def sample_dynamic_knowledge() -> List[Dict[str, Any]]:
    """Sample dynamic knowledge (learned rules) for testing."""
    return [
        {
            "rule_id": "R-482",
            "trigger": "rotterdam blocked",
            "solution": "use hamburg or antwerp",
            "confidence": 0.9,
        },
        {
            "rule_id": "H-903",
            "trigger": "hamburg to US blocked",
            "solution": "use antwerp or rotterdam",
            "confidence": 0.85,
        },
    ]


# =============================================================================
# SCENARIO GENERATOR FIXTURES
# =============================================================================


@pytest.fixture
def logistics_scenario_generator():
    """Create a LogisticsScenarioGenerator for testing."""
    from precept.scenario_generators.logistics import LogisticsScenarioGenerator

    return LogisticsScenarioGenerator(num_samples=10, train_ratio=0.6)


@pytest.fixture
def conflict_resolution_scenarios():
    """Sample conflict resolution scenarios for testing."""
    return [
        {
            "task": "Book shipment from Hamburg to New York for pharmaceutical cargo",
            "expected": "H-903 → labor strike, use Antwerp",
            "black_swan_type": "Logistics/Conflict_DynamicOverride",
            "precept_lesson": "Hamburg has labor strikes (overrides 'stable' static)",
            "phase": "training",
            "conflict_type": "dynamic_should_override",
        },
        {
            "task": "Ship pharmaceutical cargo to Boston without Certificate of Origin",
            "expected": "FAILURE → FDA requires Certificate of Origin",
            "black_swan_type": "Logistics/Conflict_StaticWins",
            "precept_lesson": "Boston pharma ALWAYS requires Certificate of Origin",
            "phase": "training",
            "conflict_type": "static_should_win",
        },
        {
            "task": "Verify best fallback port when Hamburg blocked",
            "expected": "HIGH CONFIDENCE: Rotterdam (static and dynamic agree)",
            "black_swan_type": "Logistics/Conflict_Agreement",
            "precept_lesson": "Confidence boosted when sources align",
            "phase": "test",
            "conflict_type": "agreement",
        },
    ]
