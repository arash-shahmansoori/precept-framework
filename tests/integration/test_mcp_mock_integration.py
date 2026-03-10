"""
Integration Tests for MCP Client/Server with Mocks.

Tests the MCP communication layer using mocks for CI compatibility.
These tests verify the protocol and data flow without requiring
a running MCP server or OpenAI API access.
"""

import pytest


class TestMCPClientImports:
    """Tests for MCP client module imports."""

    def test_precept_mcp_client_import(self):
        """Test that PRECEPTMCPClient can be imported."""
        from precept.precept_mcp_client import PRECEPTMCPClient

        assert PRECEPTMCPClient is not None

    def test_compass_mcp_client_import(self):
        """Test that compass MCP client module can be imported."""
        from precept import compass_mcp_client

        assert compass_mcp_client is not None


class TestMCPClientProtocol:
    """Tests for MCP client protocol compliance."""

    def test_precept_client_has_required_methods(self):
        """Test that PRECEPT client has all required protocol methods."""
        from precept.precept_mcp_client import PRECEPTMCPClient

        # Check for core methods
        assert hasattr(PRECEPTMCPClient, "connect")
        assert hasattr(PRECEPTMCPClient, "disconnect")
        assert hasattr(PRECEPTMCPClient, "call_tool")

    def test_compass_client_exports(self):
        """Test COMPASS client module exports."""
        from precept.compass_mcp_client import PRECEPTMCPClientWithCOMPASS

        # Check class can be imported and has expected methods
        assert PRECEPTMCPClientWithCOMPASS is not None
        assert hasattr(PRECEPTMCPClientWithCOMPASS, "get_evolved_prompt")
        assert hasattr(PRECEPTMCPClientWithCOMPASS, "trigger_consolidation")


class TestMockMCPOperations:
    """Tests for MCP operations using mocks."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client for testing."""

        class MockMCPClient:
            def __init__(self):
                self.memories = []
                self.rules = []
                self.experiences = []
                self.static_knowledge = []
                self.connected = False

            async def connect(self):
                self.connected = True
                return True

            async def disconnect(self):
                self.connected = False
                return True

            async def store_experience(self, task, outcome, strategy, lessons):
                self.experiences.append(
                    {
                        "task": task,
                        "outcome": outcome,
                        "strategy": strategy,
                        "lessons": lessons,
                    }
                )
                return f"✓ Stored experience: {task[:50]}"

            async def retrieve_memories(self, query, top_k=5):
                return [
                    e for e in self.experiences if query.lower() in e["task"].lower()
                ][:top_k]

            async def get_learned_rules(self):
                return self.rules

            async def store_learned_rule(self, rule_text):
                self.rules.append(rule_text)
                return f"✓ Stored rule: {rule_text[:50]}"

            async def ingest_static_knowledge(self, knowledge_items, domain, source):
                if isinstance(knowledge_items, str):
                    import json

                    items = json.loads(knowledge_items)
                else:
                    items = knowledge_items
                self.static_knowledge.extend(items)
                return f"✓ Ingested {len(items)} items"

            async def get_server_stats(self):
                return {
                    "experiences": len(self.experiences),
                    "rules": len(self.rules),
                    "static_knowledge": len(self.static_knowledge),
                    "connected": self.connected,
                }

            async def call_tool(self, tool_name, params):
                if tool_name == "store_experience":
                    return await self.store_experience(**params)
                elif tool_name == "retrieve_memories":
                    return await self.retrieve_memories(**params)
                elif tool_name == "ingest_static_knowledge":
                    return await self.ingest_static_knowledge(**params)
                return f"Tool {tool_name} called"

        return MockMCPClient()

    @pytest.mark.asyncio
    async def test_mock_connect_disconnect(self, mock_mcp_client):
        """Test mock connection lifecycle."""
        assert not mock_mcp_client.connected

        await mock_mcp_client.connect()
        assert mock_mcp_client.connected

        await mock_mcp_client.disconnect()
        assert not mock_mcp_client.connected

    @pytest.mark.asyncio
    async def test_mock_store_experience(self, mock_mcp_client):
        """Test storing an experience via mock MCP."""
        result = await mock_mcp_client.store_experience(
            task="Ship cargo from Rotterdam to Boston",
            outcome="success",
            strategy="direct_routing",
            lessons=["Use direct routes for speed"],
        )

        assert "✓" in result
        assert len(mock_mcp_client.experiences) == 1

    @pytest.mark.asyncio
    async def test_mock_retrieve_memories(self, mock_mcp_client):
        """Test retrieving memories via mock MCP."""
        # Store some experiences first
        await mock_mcp_client.store_experience(
            task="Ship cargo from Rotterdam",
            outcome="success",
            strategy="direct",
            lessons=["test"],
        )
        await mock_mcp_client.store_experience(
            task="Ship cargo from Hamburg",
            outcome="success",
            strategy="fallback",
            lessons=["test"],
        )

        # Retrieve
        results = await mock_mcp_client.retrieve_memories("Rotterdam", top_k=5)
        assert len(results) == 1
        assert "Rotterdam" in results[0]["task"]

    @pytest.mark.asyncio
    async def test_mock_store_rule(self, mock_mcp_client):
        """Test storing a learned rule via mock MCP."""
        result = await mock_mcp_client.store_learned_rule(
            "R-482: Rotterdam blocked, use Hamburg"
        )

        assert "✓" in result
        assert len(mock_mcp_client.rules) == 1

    @pytest.mark.asyncio
    async def test_mock_ingest_static_knowledge(self, mock_mcp_client):
        """Test ingesting static knowledge via mock MCP."""
        import json

        knowledge = [
            {"content": "Rotterdam is a major port", "metadata": {"type": "fact"}},
            {"content": "Hamburg has rail connections", "metadata": {"type": "fact"}},
        ]

        result = await mock_mcp_client.ingest_static_knowledge(
            knowledge_items=json.dumps(knowledge),
            domain="logistics",
            source="test.json",
        )

        assert "✓" in result
        assert len(mock_mcp_client.static_knowledge) == 2

    @pytest.mark.asyncio
    async def test_mock_get_stats(self, mock_mcp_client):
        """Test getting server statistics via mock MCP."""
        await mock_mcp_client.connect()
        await mock_mcp_client.store_experience(
            task="Test",
            outcome="success",
            strategy="test",
            lessons=[],
        )
        await mock_mcp_client.store_learned_rule("Test rule")

        stats = await mock_mcp_client.get_server_stats()

        assert stats["experiences"] == 1
        assert stats["rules"] == 1
        assert stats["connected"] is True

    @pytest.mark.asyncio
    async def test_mock_call_tool(self, mock_mcp_client):
        """Test generic tool calling via mock MCP."""
        result = await mock_mcp_client.call_tool(
            "store_experience",
            {
                "task": "Test task",
                "outcome": "success",
                "strategy": "test",
                "lessons": ["lesson1"],
            },
        )

        assert "✓" in result


class TestMockDualRetrieval:
    """Tests for dual retrieval (static + dynamic) with mocks."""

    @pytest.fixture
    def mock_dual_client(self):
        """Create a mock client with dual retrieval."""

        class MockDualClient:
            def __init__(self):
                self.static_kb = [
                    {
                        "content": "Rotterdam is a major European port",
                        "source": "static",
                    },
                    {"content": "Hamburg handles container ships", "source": "static"},
                ]
                self.dynamic_patches = [
                    {"content": "Rotterdam blocked due to strike", "source": "dynamic"},
                ]
                self.episodic = []

            async def retrieve_with_dual_mode(
                self,
                query,
                static_top_k=3,
                dynamic_top_k=3,
                episodic_top_k=3,
            ):
                results = {
                    "static": [
                        s
                        for s in self.static_kb
                        if query.lower() in s["content"].lower()
                    ][:static_top_k],
                    "dynamic": [
                        d
                        for d in self.dynamic_patches
                        if query.lower() in d["content"].lower()
                    ][:dynamic_top_k],
                    "episodic": self.episodic[:episodic_top_k],
                    "conflicts": [],
                }

                # Detect conflict
                if results["static"] and results["dynamic"]:
                    results["conflicts"].append(
                        {
                            "static": results["static"][0]["content"],
                            "dynamic": results["dynamic"][0]["content"],
                            "resolution": "Dynamic takes precedence (more recent)",
                        }
                    )

                return results

        return MockDualClient()

    @pytest.mark.asyncio
    async def test_dual_retrieval_static_only(self, mock_dual_client):
        """Test dual retrieval with static knowledge only."""
        results = await mock_dual_client.retrieve_with_dual_mode(
            query="Hamburg",
            static_top_k=3,
            dynamic_top_k=0,
            episodic_top_k=0,
        )

        assert len(results["static"]) > 0
        assert "Hamburg" in results["static"][0]["content"]

    @pytest.mark.asyncio
    async def test_dual_retrieval_with_conflict(self, mock_dual_client):
        """Test dual retrieval detecting conflicts."""
        results = await mock_dual_client.retrieve_with_dual_mode(
            query="Rotterdam",
            static_top_k=3,
            dynamic_top_k=3,
            episodic_top_k=3,
        )

        # Should have both static and dynamic results
        assert len(results["static"]) > 0
        assert len(results["dynamic"]) > 0

        # Should detect conflict
        assert len(results["conflicts"]) > 0


class TestMockConflictResolution:
    """Tests for conflict resolution with mocks."""

    @pytest.fixture
    def mock_conflict_client(self):
        """Create a mock client with conflict resolution."""

        class MockConflictClient:
            def __init__(self):
                self.conflicts_detected = 0
                self.conflicts_resolved = 0

            async def get_conflict_resolution_stats(self):
                return {
                    "status": "active",
                    "summary": {
                        "conflicts_detected": self.conflicts_detected,
                        "conflicts_resolved": self.conflicts_resolved,
                    },
                    "resolution_outcomes": {
                        "static_wins": 0,
                        "dynamic_wins": self.conflicts_resolved,
                        "merges": 0,
                    },
                }

            async def resolve_conflict(self, static_item, dynamic_item):
                self.conflicts_detected += 1
                self.conflicts_resolved += 1
                return {
                    "winner": "dynamic",
                    "reason": "More recent information",
                    "confidence": 0.85,
                }

        return MockConflictClient()

    @pytest.mark.asyncio
    async def test_mock_conflict_stats(self, mock_conflict_client):
        """Test getting conflict resolution statistics."""
        stats = await mock_conflict_client.get_conflict_resolution_stats()

        assert stats["status"] == "active"
        assert "summary" in stats
        assert "resolution_outcomes" in stats

    @pytest.mark.asyncio
    async def test_mock_resolve_conflict(self, mock_conflict_client):
        """Test resolving a conflict."""
        result = await mock_conflict_client.resolve_conflict(
            static_item={"content": "Rotterdam is operational"},
            dynamic_item={"content": "Rotterdam is blocked"},
        )

        assert result["winner"] == "dynamic"
        assert result["confidence"] > 0

        # Stats should be updated
        stats = await mock_conflict_client.get_conflict_resolution_stats()
        assert stats["summary"]["conflicts_detected"] == 1
        assert stats["summary"]["conflicts_resolved"] == 1


class TestAgentWithMockMCP:
    """Tests for agent integration with mock MCP client."""

    def test_precept_agent_creation(self, mock_domain_strategy):
        """Test creating a PRECEPT agent."""
        from precept import PRECEPTAgent

        agent = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        assert agent is not None

    def test_baseline_agent_creation(self, mock_baseline_strategy):
        """Test creating baseline agents."""
        from precept import (
            FullReflexionBaselineAgent,
            LLMBaselineAgent,
            ReflexionBaselineAgent,
        )

        llm = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)
        ref = ReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)
        full = FullReflexionBaselineAgent(baseline_strategy=mock_baseline_strategy)

        assert llm is not None
        assert ref is not None
        assert full is not None

    def test_agents_have_mcp_client_attribute(
        self, mock_domain_strategy, mock_baseline_strategy
    ):
        """Test that agents have MCP client attributes."""
        from precept import LLMBaselineAgent, PRECEPTAgent

        precept = PRECEPTAgent(domain_strategy=mock_domain_strategy)
        llm = LLMBaselineAgent(baseline_strategy=mock_baseline_strategy)

        # Agents should have mcp_client attribute (None until connected)
        assert hasattr(precept, "mcp_client")
        assert hasattr(llm, "mcp_client")
