#!/usr/bin/env python3
"""
PRECEPT MCP Client - Abstract Base Class for PRECEPT MCP Server Connections.

This module provides:
1. AbstractPRECEPTMCPClient - Base class with GENERAL PRECEPT methods only
2. Domain-specific tools (book_shipment, check_port) should be implemented
   in example-specific subclasses or called via call_tool() dynamically.

Design Philosophy (from black_swan_gen.py categories):
- PRECEPT core methods (memory, learning, COMPASS) are UNIVERSAL across all domains
- Domain-specific tools vary by black swan category:
  - Logistics: book_shipment, check_port, check_carrier
  - Coding: install_package, run_tests, check_imports
  - DevOps: create_stack, assume_role, deploy_pod
  - Finance: execute_order, check_market, validate_trade
  - Booking: reserve_seat, check_availability, process_payment
  - Integration: sync_data, refresh_token, call_api

Usage:
    # For logistics black swan scenarios:
    class LogisticsMCPClient(AbstractPRECEPTMCPClient):
        async def book_shipment(self, origin: str, destination: str) -> str:
            return await self.call_tool("book_shipment", {"origin": origin, "destination": destination})

    # For coding black swan scenarios:
    class CodingMCPClient(AbstractPRECEPTMCPClient):
        async def install_package(self, package: str) -> str:
            return await self.call_tool("install_package", {"package": package})
"""

import asyncio
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import mcp
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_precept_server_params(
    server_script: Optional[Path] = None,
    use_uv: bool = True,
) -> StdioServerParameters:
    """
    Get parameters to start the PRECEPT MCP server.

    Args:
        server_script: Path to MCP server script (defaults to precept_mcp_server.py)
        use_uv: Use 'uv run' instead of 'python3' (recommended)

    Returns:
        StdioServerParameters for MCP client connection
    """
    if server_script is None:
        server_script = PROJECT_ROOT / "src" / "precept" / "precept_mcp_server.py"

    # CRITICAL: Inherit parent environment to pass OPENAI_API_KEY and other env vars
    import os

    parent_env = dict(os.environ)
    parent_env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    if use_uv:
        return StdioServerParameters(
            command="uv",
            args=["run", str(server_script)],
            env=parent_env,  # Pass parent environment including API keys
        )
    else:
        return StdioServerParameters(
            command="python3",
            args=[str(server_script)],
            env=parent_env,  # Pass parent environment including API keys
        )


class AbstractPRECEPTMCPClient(ABC):
    """
    Abstract Base Class for PRECEPT MCP Clients.

    Provides GENERAL PRECEPT methods that work across ALL black swan categories:
    - Memory: retrieve_memories, store_experience
    - Learning: get_learned_rules, record_error
    - COMPASS: trigger_compass_evolution, trigger_consolidation
    - Procedures: get_procedure, store_procedure
    - Stats: get_server_stats

    Domain-specific tools should be implemented in subclasses.
    See black_swan_gen.py for category-specific tool requirements.
    """

    def __init__(self, server_params: Optional[StdioServerParameters] = None):
        """
        Initialize the PRECEPT MCP client.

        Args:
            server_params: Optional custom server parameters
        """
        self.server_params = server_params or get_precept_server_params()
        self._stdio_context = None
        self._session_context = None
        self._streams = None
        self._session = None
        self._tools: Dict[str, Any] = {}
        self._connected = False

    async def __aenter__(self):
        """Async context manager entry - connect to server."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect from server."""
        await self.disconnect()

    async def connect(self):
        """Connect to the PRECEPT MCP server."""
        if self._connected:
            return

        self._stdio_context = stdio_client(self.server_params)
        self._streams = await self._stdio_context.__aenter__()

        self._session_context = mcp.ClientSession(*self._streams)
        self._session = await self._session_context.__aenter__()

        await self._session.initialize()

        # Cache available tools
        tools_result = await self._session.list_tools()
        self._tools = {tool.name: tool for tool in tools_result.tools}
        self._connected = True

    async def disconnect(self):
        """Disconnect from the PRECEPT MCP server."""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if self._stdio_context:
                await self._stdio_context.__aexit__(None, None, None)
        except Exception:
            pass  # MCP cleanup can fail, which is fine
        finally:
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    async def list_tools(self) -> List[Dict]:
        """List all available tools from the server."""
        if not self._tools:
            tools_result = await self._session.list_tools()
            self._tools = {tool.name: tool for tool in tools_result.tools}

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in self._tools.values()
        ]

    async def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        Call any tool on the PRECEPT server.

        This is the universal method for calling tools. Subclasses can use this
        to implement domain-specific convenience methods.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string
        """
        if not self._connected:
            raise RuntimeError("Client not connected. Call connect() first.")

        result = await self._session.call_tool(tool_name, arguments)

        # Extract content from result
        if result.content:
            return result.content[0].text
        return str(result)

    # =========================================================================
    # UNIVERSAL PRECEPT METHODS (Work across ALL black swan categories)
    # =========================================================================

    # --- Memory System ---

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        force: bool = False,
    ) -> str:
        """
        Retrieve relevant memories using semantic search.

        Uses ChromaDB + OpenAI embeddings for vector similarity search.
        Implements Reactive Retrieval (Memory-as-a-Tool) pattern.

        Args:
            query: What you're looking for (describe the situation)
            top_k: Number of memories to retrieve (default 5)
            force: Force retrieval even if system thinks it's not needed

        Returns:
            Semantically similar past experiences
        """
        return await self.call_tool(
            "retrieve_memories",
            {
                "query": query,
                "top_k": top_k,
                "force": force,
            },
        )

    async def store_experience(
        self,
        task: str,
        outcome: str,
        strategy: str = "",
        lessons: str = "",
        domain: str = "general",
        error_code: str = "",
        solution: str = "",
        failed_options: str = "",
        task_type: str = "",
        condition_key: str = "",
    ) -> str:
        """
        Store a new experience to PRECEPT memory.

        Uses Background Memory Writer pattern for async non-blocking writes.

        Args:
            task: What task was attempted
            outcome: "success" or "failure"
            strategy: Strategy used (e.g., "Fallback to Antwerp")
            lessons: Lessons learned
            domain: Domain for memory scoping
            error_code: The error code encountered (e.g., BK-401)
            solution: The working solution found
            failed_options: Comma-separated list of options that failed
            task_type: Type of task for indexing (e.g., "booking:book_flight")
            condition_key: The composite condition key for multi-condition scenarios

        Returns:
            Confirmation of storage
        """
        return await self.call_tool(
            "store_experience",
            {
                "task": task,
                "outcome": outcome,
                "strategy": strategy,
                "lessons": lessons,
                "domain": domain,
                "error_code": error_code,
                "solution": solution,
                "failed_options": failed_options,
                "task_type": task_type,
                "condition_key": condition_key,
            },
        )

    # --- Learning System ---

    async def get_learned_rules(self) -> str:
        """
        Get all learned rules from PRECEPT.

        Rules are extracted from error patterns and memory consolidation.
        Call this BEFORE executing tasks to apply proactive learning.

        Returns:
            All currently learned rules
        """
        return await self.call_tool("get_learned_rules", {})

    async def get_rule_hybrid(
        self,
        condition_key: str,
        task_description: str = "",
        similarity_threshold: float = 0.5,
        top_k: int = 3,
    ) -> str:
        """
        HYBRID RULE RETRIEVAL: 3-Tier strategy combining best of PRECEPT and ExpeL.

        TIER 1: O(1) hash lookup (instant, exact match) - PRECEPT's unique advantage
        TIER 2: Vector similarity (semantic, like ExpeL) - for partial matching
        TIER 3: Jaccard similarity (structural) - fallback on condition codes

        This gives PRECEPT the BEST of ALL worlds:
        - O(1) deterministic lookup when exact match exists (matched mode)
        - Semantic matching like ExpeL for unseen combinations (random mode)
        - Structural matching as final fallback

        Args:
            condition_key: The multi-condition key (e.g., "C-COLD+C-HZMT+E-HEAT+...")
            task_description: Optional task text for semantic matching (Tier 2)
            similarity_threshold: Minimum similarity for partial matches (0.0-1.0)
            top_k: Number of top similar rules to return

        Returns:
            JSON with exact_match, vector_matches, jaccard_matches, and strategy_used
        """
        return await self.call_tool(
            "get_rule_hybrid",
            {
                "condition_key": condition_key,
                "task_description": task_description,
                "similarity_threshold": similarity_threshold,
                "top_k": top_k,
            },
        )

    async def clear_learned_data(
        self,
        clear_rules: bool = True,
        clear_experiences: bool = False,
        clear_domain_mappings: bool = True,
    ) -> str:
        """
        Clear learned data for fair experiment comparison.

        Call this at the start of an experiment to ensure a clean slate,
        matching Full Reflexion's behavior of starting fresh each run.

        Args:
            clear_rules: Clear learned rules (precept_learned_rules.json)
            clear_experiences: Clear episodic memory (precept_experiences.json)
            clear_domain_mappings: Clear domain mappings (precept_domain_mappings.json)

        Returns:
            Summary of what was cleared
        """
        return await self.call_tool(
            "clear_learned_data",
            {
                "clear_rules": clear_rules,
                "clear_experiences": clear_experiences,
                "clear_domain_mappings": clear_domain_mappings,
            },
        )

    async def reload_learned_rules(self) -> str:
        """
        Reload learned rules from disk to ensure in-memory state is fresh.

        CRITICAL: Call this AFTER training and BEFORE testing to ensure
        the hybrid lookup uses the LATEST rules from training.

        Returns:
            Status message with number of rules loaded
        """
        return await self.call_tool("reload_learned_rules", {})

    async def retrieve_by_error_code(self, error_code: str, top_k: int = 5) -> str:
        """
        Retrieve experiences by error code (exact match in metadata).

        This enables error-code-based retrieval instead of just semantic search.

        Args:
            error_code: The error code to search for (e.g., BK-401, ROUTE_BLOCKED)
            top_k: Maximum number of results to return

        Returns:
            Relevant experiences for this error code
        """
        return await self.call_tool(
            "retrieve_by_error_code",
            {"error_code": error_code, "top_k": top_k},
        )

    async def record_error(
        self, error_code: str, context: str, solution: str = ""
    ) -> str:
        """
        Record an error pattern for learning.

        If solution is provided, a rule is learned IMMEDIATELY.
        Otherwise, after 2+ occurrences PRECEPT looks for a solution in patterns.

        Args:
            error_code: The error code (e.g., "R-482", "PIP_INSTALL_FAIL")
            context: Description of when error occurred
            solution: (Optional) The working solution for this error

        Returns:
            Status and any new rules learned
        """
        return await self.call_tool(
            "record_error",
            {
                "error_code": error_code,
                "context": context,
                "solution": solution,
            },
        )

    async def record_solution(
        self, error_code: str, solution: str, context: str = "", task_succeeded: bool = False
    ) -> str:
        """
        Record a successful solution for an error code.

        Call this when you discover what works to resolve an error.
        This creates/updates a learned rule with the specific solution.

        CRITICAL: Rules are ONLY persisted when task_succeeded=True!
        This prevents learning incorrect rules from failed tasks.

        Args:
            error_code: The error code that was resolved
            solution: The solution that worked (e.g., "use antwerp")
            context: Optional context for when this applies
            task_succeeded: Whether the overall task completed successfully.
                           Must be True for the rule to be persisted.

        Returns:
            Confirmation of the learned rule
        """
        return await self.call_tool(
            "record_solution",
            {
                "error_code": error_code,
                "solution": solution,
                "context": context,
                "task_succeeded": task_succeeded,
            },
        )

    # --- COMPASS Evolution ---

    async def trigger_compass_evolution(
        self,
        failure_context: str = "",
        trajectory: str = "",
    ) -> str:
        """
        Trigger COMPASS prompt evolution.

        This invokes the COMPASS evolution engine with LLM-based:
        - Reflective analysis of failures
        - Prompt mutation with lessons learned
        - Pareto-based candidate selection
        - ML-based complexity detection
        - Smart rollout allocation
        - Multi-strategy coordination

        Args:
            failure_context: Description of recent failures
            trajectory: Recent execution trajectory

        Returns:
            Evolution result with prompt mutations
        """
        return await self.call_tool(
            "trigger_compass_evolution",
            {
                "failure_context": failure_context,
                "trajectory": trajectory,
            },
        )

    async def trigger_consolidation(self) -> str:
        """
        Trigger memory consolidation.

        This invokes the REAL MemoryConsolidator with LLM-based:
        - Pattern frequency analysis
        - Duplicate memory merging
        - Rule extraction from patterns
        - Irrelevance-based pruning

        Returns:
            Consolidation result with new rules
        """
        return await self.call_tool("trigger_consolidation", {})

    async def get_evolved_prompt(self, include_rules: bool = True) -> str:
        """
        Get the BEST EVOLVED PROMPT from COMPASS optimization.

        THIS IS THE KEY PRECEPT ADVANTAGE:
        - Returns the prompt evolved through COMPASS
        - Includes consolidated learned rules and domain mappings
        - Should be used to update the agent's system prompt

        COMPASS advantages over basic approaches:
        - ML-based complexity analysis
        - Smart rollout allocation
        - Dynamic prompt evolution with learned rules

        Args:
            include_rules: Whether to append learned rules to the prompt

        Returns:
            The best evolved prompt with learned knowledge
        """
        return await self.call_tool(
            "get_evolved_prompt",
            {
                "include_rules": include_rules,
            },
        )

    async def get_prompt_evolution_status(self) -> str:
        """
        Get detailed status of COMPASS prompt evolution.

        Returns:
            Detailed evolution status including Pareto front info
        """
        return await self.call_tool("get_prompt_evolution_status", {})

    # --- Procedural Memory ---

    async def get_procedure(self, task_type: str) -> str:
        """
        Retrieve a stored procedure for a task type.

        Procedural memory stores "how-to" strategies that worked.

        Args:
            task_type: Type of task (e.g., "shipping_booking", "package_install")

        Returns:
            Matching procedures with steps
        """
        return await self.call_tool(
            "get_procedure",
            {
                "task_type": task_type,
            },
        )

    async def store_procedure(
        self,
        name: str,
        task_type: str,
        steps: str,
    ) -> str:
        """
        Store a procedure in procedural memory.

        Args:
            name: Procedure name
            task_type: Type of task this procedure handles
            steps: Step-by-step instructions

        Returns:
            Confirmation of storage
        """
        return await self.call_tool(
            "store_procedure",
            {
                "name": name,
                "task_type": task_type,
                "steps": steps,
            },
        )

    # --- Statistics ---

    async def get_server_stats(self) -> str:
        """
        Get comprehensive server statistics.

        Includes:
        - Memory stats
        - Learning stats
        - COMPASS stats
        - Context Engineering stats

        Returns:
            Formatted statistics
        """
        return await self.call_tool("get_server_stats", {})

    # --- Memory Feedback Loop ---

    async def update_memory_usefulness(
        self,
        feedback: float,
        task_succeeded: bool = False,
        memory_ids: str = "",
    ) -> str:
        """
        Update usefulness of retrieved memories based on task outcome.

        FEEDBACK LOOP: Call this after task completion to improve future retrievals.
        Memories that help solve tasks get higher usefulness scores.

        Args:
            feedback: -1.0 to 1.0 (negative = hindered, positive = helped)
            task_succeeded: Whether the overall task completed successfully
            memory_ids: Comma-separated memory IDs (optional, uses last retrieved if empty)

        Returns:
            Confirmation of updated memories
        """
        return await self.call_tool(
            "update_memory_usefulness",
            {
                "feedback": feedback,
                "task_succeeded": task_succeeded,
                "memory_ids": memory_ids,
            },
        )

    async def get_last_retrieved_ids(self) -> str:
        """
        Get the IDs of memories from the last retrieval.

        Use this to track which memories were retrieved for feedback purposes.

        Returns:
            Comma-separated list of memory IDs from last retrieval
        """
        return await self.call_tool("get_last_retrieved_ids", {})

    # =========================================================================
    # AutoGen Integration Helpers
    # =========================================================================

    def get_openai_tools(self) -> List[Dict]:
        """
        Get tools in OpenAI function calling format.

        This can be used to provide tools to AutoGen agents.
        """
        openai_tools = []

        for tool in self._tools.values():
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        **tool.inputSchema,
                        "additionalProperties": False,
                    },
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools


# =============================================================================
# CONCRETE IMPLEMENTATIONS FOR EACH BLACK SWAN CATEGORY
# =============================================================================
# These can be used directly or as templates for custom implementations.


class LogisticsMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for Logistics/Shipping black swan scenarios.

    Domain tools:
    - book_shipment: Book cargo shipment
    - check_port: Check port availability
    - check_carrier: Check carrier status
    - calculate_route: Calculate shipping route
    """

    async def book_shipment(self, origin: str, destination: str) -> str:
        """Book a shipment from origin to destination."""
        return await self.call_tool(
            "book_shipment",
            {
                "origin": origin,
                "destination": destination,
            },
        )

    async def check_port(self, port: str) -> str:
        """Check if a port is available."""
        return await self.call_tool("check_port", {"port": port})

    async def check_carrier(self, carrier: str) -> str:
        """Check carrier availability."""
        return await self.call_tool("check_carrier", {"carrier": carrier})

    async def clear_customs(
        self, destination: str, documentation: str = "standard"
    ) -> str:
        """Clear customs for a shipment to a destination with specified documentation."""
        return await self.call_tool(
            "clear_customs",
            {
                "destination": destination,
                "documentation": documentation,
            },
        )


class CodingMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for Coding/Development black swan scenarios.

    Domain tools:
    - install_package: Install Python package
    - run_tests: Execute test suite
    - check_imports: Verify import dependencies
    - execute_code: Run code snippet
    """

    async def install_package(self, package: str) -> str:
        """Install a Python package."""
        return await self.call_tool("install_package", {"package": package})

    async def run_tests(self, test_path: str) -> str:
        """Run tests at specified path."""
        return await self.call_tool("run_tests", {"test_path": test_path})

    async def check_imports(self, module: str) -> str:
        """Check if module can be imported."""
        return await self.call_tool("check_imports", {"module": module})


class DevOpsMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for DevOps/Infrastructure black swan scenarios.

    Domain tools:
    - create_stack: Create CloudFormation stack
    - assume_role: Assume IAM role
    - deploy_pod: Deploy Kubernetes pod
    - check_stack_status: Check stack status
    """

    async def create_stack(self, stack_name: str, template: str) -> str:
        """Create a CloudFormation stack."""
        return await self.call_tool(
            "create_stack",
            {
                "stack_name": stack_name,
                "template": template,
            },
        )

    async def assume_role(self, role_name: str) -> str:
        """Assume an IAM role."""
        return await self.call_tool("assume_role", {"role_name": role_name})

    async def deploy_pod(self, pod_name: str, image: str) -> str:
        """Deploy a Kubernetes pod."""
        return await self.call_tool(
            "deploy_pod",
            {
                "pod_name": pod_name,
                "image": image,
            },
        )


class FinanceMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for Finance/Trading black swan scenarios.

    Domain tools:
    - execute_order: Execute a trade order
    - check_market: Check market data
    - validate_trade: Validate trade parameters
    """

    async def execute_order(
        self,
        symbol: str,
        quantity: int,
        order_type: str,
    ) -> str:
        """Execute a trade order."""
        return await self.call_tool(
            "execute_order",
            {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
            },
        )

    async def check_market(self, symbol: str) -> str:
        """Check market data for symbol."""
        return await self.call_tool("check_market", {"symbol": symbol})


class BookingMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for Booking/Travel black swan scenarios.

    Domain tools:
    - reserve_seat: Reserve flight/train seat
    - check_availability: Check inventory
    - process_payment: Process payment
    """

    async def reserve_seat(
        self,
        flight: str,
        seat: str,
    ) -> str:
        """Reserve a seat on a flight."""
        return await self.call_tool(
            "reserve_seat",
            {
                "flight": flight,
                "seat": seat,
            },
        )

    async def check_availability(self, resource_id: str) -> str:
        """Check availability of a resource."""
        return await self.call_tool("check_availability", {"resource_id": resource_id})

    async def process_payment(
        self,
        amount: float,
        currency: str,
    ) -> str:
        """Process a payment."""
        return await self.call_tool(
            "process_payment",
            {
                "amount": amount,
                "currency": currency,
            },
        )


class IntegrationMCPClient(AbstractPRECEPTMCPClient):
    """
    MCP Client for Integration/API black swan scenarios.

    Domain tools:
    - sync_data: Sync data with external system
    - refresh_token: Refresh OAuth token
    - call_api: Call external API
    """

    async def sync_data(self, source: str, target: str) -> str:
        """Sync data between systems."""
        return await self.call_tool(
            "sync_data",
            {
                "source": source,
                "target": target,
            },
        )

    async def refresh_token(self, service: str) -> str:
        """Refresh OAuth token for a service."""
        return await self.call_tool("refresh_token", {"service": service})

    async def call_api(self, endpoint: str, method: str = "GET") -> str:
        """Call an external API."""
        return await self.call_tool(
            "call_api",
            {
                "endpoint": endpoint,
                "method": method,
            },
        )


# =============================================================================
# STANDALONE FUNCTIONS (for simple usage)
# =============================================================================


async def call_precept_tool(tool_name: str, arguments: Dict) -> str:
    """
    Call a PRECEPT tool (one-shot usage).

    This creates a new connection for each call.
    For multiple calls, use a PRECEPTMCPClient subclass.
    """
    params = get_precept_server_params()

    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            if result.content:
                return result.content[0].text
            return str(result)


async def list_precept_tools() -> List[Dict]:
    """List all tools available from the PRECEPT server."""
    params = get_precept_server_params()

    async with stdio_client(params) as streams:
        async with mcp.ClientSession(*streams) as session:
            await session.initialize()
            tools_result = await session.list_tools()

            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                }
                for tool in tools_result.tools
            ]


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# Alias for backward compatibility - LogisticsMCPClient is the default
PRECEPTMCPClient = LogisticsMCPClient


# =============================================================================
# TEST
# =============================================================================


async def test_abstract_client():
    """Test the abstract PRECEPT MCP client."""
    print("🧪 Testing Abstract PRECEPT MCP Client...")
    print()

    # Use the logistics client for testing
    async with LogisticsMCPClient() as client:
        # List tools
        print("📋 Available tools:")
        tools = await client.list_tools()
        for tool in tools:
            print(f"   • {tool['name']}: {tool['description'][:60]}...")
        print()

        # Test universal PRECEPT methods
        print("📚 Getting learned rules (universal method)...")
        rules = await client.get_learned_rules()
        print(rules[:200] if len(rules) > 200 else rules)
        print()

        # Test domain-specific method
        print("📦 Testing logistics-specific method: book_shipment...")
        result = await client.book_shipment("rotterdam", "boston")
        print(result)
        print()

        # Test stats
        print("📊 Server stats:")
        stats = await client.get_server_stats()
        print(stats[:500] if len(stats) > 500 else stats)


if __name__ == "__main__":
    print("=" * 60)
    print("PRECEPT MCP Client Test - Abstract Base Class")
    print("=" * 60)
    asyncio.run(test_abstract_client())
