"""
PRECEPT AutoGen Integration: Scalable Agentic Framework.

This module integrates AutoGen 0.7.5 with PRECEPT to provide:
- Multi-agent architectures with PRECEPT's learning capabilities
- Scalable tool execution through MCP protocol
- Preservation of all existing PRECEPT features (ReMem, GEPA, Context Engineering)

AutoGen 0.7.5 Features Used:
- AssistantAgent: For reasoning and planning
- UserProxyAgent: For tool execution
- GroupChat: For multi-agent coordination
- ConversableAgent: Base class for custom agents

Architecture:
┌──────────────────────────────────────────────────────────────────────────────┐
│                         PRECEPT + AutoGen + MCP Architecture                    │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐     ┌─────────────────────┐                        │
│  │   AutoGen Agents    │────▶│   PRECEPT Core        │                        │
│  │                     │     │                     │                        │
│  │  - PRECEPTAssistant   │     │  - ReMem Pipeline   │                        │
│  │  - ToolExecutor     │     │  - Memory Store     │                        │
│  │  - Researcher       │     │  - GEPA Evolution   │                        │
│  │  - Coordinator      │     │  - Context Eng.     │                        │
│  └─────────┬───────────┘     └─────────────────────┘                        │
│            │                                                                 │
│            ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MCP Tool Layer                                  │    │
│  │                                                                      │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│  │  │ Memory MCP    │  │ Retrieval MCP │  │ External MCP  │            │    │
│  │  │ Server        │  │ Server        │  │ Servers       │            │    │
│  │  │               │  │               │  │               │            │    │
│  │  │ - store_exp   │  │ - semantic    │  │ - fetch       │            │    │
│  │  │ - retrieve    │  │ - episodic    │  │ - browse      │            │    │
│  │  │ - consolidate │  │ - procedural  │  │ - file_system │            │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Dependencies:
- autogen-agentchat>=0.7.5
- mcp (Model Context Protocol)
- All existing PRECEPT dependencies
"""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field

# PRECEPT Core Imports
from .memory_store import Experience, ExperienceType, MemoryPriority, MemoryStore
from .remem_pipeline import ReMem, ReMemState, ReMemPhase, ThinkActRefineLoop
from .context_engineering import (
    ContextEngineeringManager,
    MemoryScope,
    ProceduralMemoryStore,
    ReactiveRetriever,
    SessionCompactor,
    BackgroundMemoryWriter,
)
from .llm_clients import precept_llm_client, precept_embedding_fn

# Try to import AutoGen
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    from autogen_agentchat.messages import TextMessage, ToolCallMessage, ToolCallResultMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError as e:
    AUTOGEN_AVAILABLE = False
    AUTOGEN_IMPORT_ERROR = str(e)

# Try to import MCP
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import Tool as MCPTool
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    MCP_IMPORT_ERROR = str(e)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AutoGenPRECEPTConfig:
    """Configuration for AutoGen-PRECEPT integration."""

    # LLM Configuration
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # Agent Configuration
    max_consecutive_auto_reply: int = 10
    human_input_mode: str = "NEVER"  # NEVER, ALWAYS, TERMINATE

    # PRECEPT Configuration
    enable_memory: bool = True
    enable_learning: bool = True
    enable_context_engineering: bool = True
    max_memory_retrievals: int = 5

    # MCP Configuration
    enable_mcp: bool = True
    mcp_server_timeout: int = 120

    # Multi-Agent Configuration
    enable_multi_agent: bool = False
    team_size: int = 3

    def get_llm_config(self) -> Dict:
        """Get AutoGen-compatible LLM configuration."""
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        return {
            "model": self.model,
            "api_key": api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# =============================================================================
# PRECEPT TOOL DEFINITIONS (For AutoGen Function Calling)
# =============================================================================

class PRECEPTToolDefinitions:
    """
    Tool definitions for PRECEPT capabilities exposed to AutoGen agents.

    These tools wrap PRECEPT's core functionality for use by AutoGen agents.
    """

    @staticmethod
    def get_memory_tools() -> List[Dict]:
        """Get tool definitions for memory operations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "store_experience",
                    "description": "Store a new experience/lesson learned to PRECEPT's episodic memory. Use when you learn something new that should be remembered.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The experience or lesson to store"
                            },
                            "outcome": {
                                "type": "string",
                                "enum": ["success", "failure", "partial"],
                                "description": "The outcome of the experience"
                            },
                            "strategy": {
                                "type": "string",
                                "description": "The strategy used (optional)"
                            },
                            "domain": {
                                "type": "string",
                                "description": "Domain/category of this experience"
                            }
                        },
                        "required": ["content", "outcome"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memories",
                    "description": "Retrieve relevant memories for a given query. Use before attempting a task to check if you have relevant past experience.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to search memories for"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of memories to retrieve (default: 5)",
                                "default": 5
                            },
                            "filter_success": {
                                "type": "boolean",
                                "description": "Only retrieve successful experiences",
                                "default": False
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "load_procedural_memory",
                    "description": "Load a procedural memory (how-to strategy/playbook) for a specific task type.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "description": "Type of task to get procedure for"
                            }
                        },
                        "required": ["task_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_learned_rules",
                    "description": "Get all rules learned from past experiences. Use to check what has been learned before starting a task.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]

    @staticmethod
    def get_reasoning_tools() -> List[Dict]:
        """Get tool definitions for reasoning operations."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_task",
                    "description": "Analyze task complexity and determine best approach using PRECEPT's complexity analyzer.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Task to analyze"
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "reflect_on_outcome",
                    "description": "Reflect on a task outcome and extract lessons learned. Triggers GEPA evolution if patterns detected.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The task that was attempted"
                            },
                            "outcome": {
                                "type": "string",
                                "description": "What happened"
                            },
                            "success": {
                                "type": "boolean",
                                "description": "Whether the task succeeded"
                            },
                            "errors_encountered": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Any error codes or messages encountered"
                            }
                        },
                        "required": ["task", "outcome", "success"]
                    }
                }
            }
        ]


# =============================================================================
# PRECEPT-ENHANCED AUTOGEN AGENTS
# =============================================================================

class PRECEPTAgentMixin:
    """
    Mixin class providing PRECEPT capabilities to AutoGen agents.

    This mixin adds:
    - Episodic memory integration
    - Experience-based learning
    - Context engineering patterns
    - GEPA evolution triggers
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        context_engineering: Optional[ContextEngineeringManager] = None,
        procedural_store: Optional[ProceduralMemoryStore] = None,
        enable_learning: bool = True,
    ):
        self.memory_store = memory_store or MemoryStore()
        self.context_engineering = context_engineering
        self.procedural_store = procedural_store or ProceduralMemoryStore()
        self.enable_learning = enable_learning

        # Learning statistics
        self.experiences_stored = 0
        self.memories_retrieved = 0
        self.rules_applied = 0
        self.tasks_completed = 0

        # Learned rules (persistent across tasks)
        self.learned_rules: Dict[str, str] = {}
        self.error_patterns: Dict[str, List[str]] = {}

    async def store_experience(
        self,
        content: str,
        outcome: str,
        strategy: str = "",
        domain: str = "general",
    ) -> str:
        """Store an experience to episodic memory."""
        if not self.enable_learning:
            return "Learning disabled"

        exp_type = ExperienceType.SUCCESS if outcome == "success" else ExperienceType.FAILURE

        self.memory_store.store_experience(
            task_description=content,
            goal="Complete task successfully",
            trajectory=[{"action": "task", "observation": content, "thought": strategy}],
            outcome=outcome,
            correctness=1.0 if outcome == "success" else 0.0,
            strategy_used=strategy,
            lessons_learned=[content] if outcome != "success" else [],
            skills_demonstrated=[],
            experience_type=exp_type,
            priority=MemoryPriority.MEDIUM,
            domain=domain,
        )
        self.experiences_stored += 1

        # Check for pattern-based rule extraction
        await self._check_for_rule_extraction(content, outcome)

        return f"Experience stored (Total: {self.experiences_stored})"

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        filter_success: bool = False,
    ) -> List[Dict]:
        """Retrieve relevant memories."""
        memories = self.memory_store.retrieve_experiences(query, k=top_k)

        if filter_success:
            memories = [m for m in memories if m.correctness > 0.5]

        self.memories_retrieved += len(memories)

        return [
            {
                "content": m.content,
                "outcome": m.outcome,
                "strategy": m.strategy,
                "domain": m.domain,
                "relevance": getattr(m, "similarity", 0.0),
            }
            for m in memories
        ]

    async def load_procedural_memory(self, task_type: str) -> Optional[str]:
        """Load procedural memory for a task type."""
        procedures = self.procedural_store.get_procedures_for_task(task_type)
        if procedures:
            return procedures[0].steps_text
        return None

    def get_learned_rules(self) -> Dict[str, str]:
        """Get all learned rules."""
        return self.learned_rules.copy()

    async def _check_for_rule_extraction(self, content: str, outcome: str):
        """Check if we can extract a rule from patterns."""
        # Extract error codes from content
        import re
        error_codes = re.findall(r'[A-Z]{1,2}-\d{3}', content)

        for code in error_codes:
            if code not in self.error_patterns:
                self.error_patterns[code] = []
            self.error_patterns[code].append(content)

            # After 3 occurrences, try to extract a rule
            if len(self.error_patterns[code]) >= 3 and code not in self.learned_rules:
                rule = await self._extract_rule_from_patterns(code, self.error_patterns[code])
                if rule:
                    self.learned_rules[code] = rule

    async def _extract_rule_from_patterns(
        self,
        error_code: str,
        patterns: List[str],
    ) -> Optional[str]:
        """Use LLM to extract a rule from observed patterns."""
        prompt = f"""Analyze these error patterns and extract a rule:

Error Code: {error_code}
Occurrences:
{chr(10).join(f'- {p}' for p in patterns[-5:])}

Extract a concise rule for avoiding this error. Format: "When X, do Y instead of Z"
"""
        try:
            response = await precept_llm_client(
                system_prompt="You extract rules from error patterns.",
                user_prompt=prompt,
            )
            return response.strip()
        except Exception:
            return None

    def get_context_for_task(self, task: str) -> str:
        """Get relevant context for a task (memories + rules)."""
        context_parts = []

        # Add learned rules
        if self.learned_rules:
            context_parts.append("LEARNED RULES (MUST FOLLOW):")
            for code, rule in self.learned_rules.items():
                context_parts.append(f"- [{code}] {rule}")

        # Add recent memories (sync call for simplicity)
        try:
            memories = self.memory_store.retrieve_experiences(task, k=3)
            if memories:
                context_parts.append("\nRELEVANT PAST EXPERIENCES:")
                for m in memories:
                    context_parts.append(f"- {m.content} (Outcome: {m.outcome})")
        except Exception:
            pass

        return "\n".join(context_parts) if context_parts else ""


# =============================================================================
# AUTOGEN AGENT IMPLEMENTATIONS
# =============================================================================

if AUTOGEN_AVAILABLE:

    class PRECEPTAssistantAgent(AssistantAgent, PRECEPTAgentMixin):
        """
        AutoGen AssistantAgent enhanced with PRECEPT capabilities.

        This agent combines AutoGen's planning and reasoning with PRECEPT's:
        - Episodic memory for experience-based learning
        - Context engineering for efficient memory usage
        - GEPA evolution for prompt improvement
        """

        def __init__(
            self,
            name: str,
            model_client: Any,
            memory_store: Optional[MemoryStore] = None,
            context_engineering: Optional[ContextEngineeringManager] = None,
            enable_learning: bool = True,
            system_message: Optional[str] = None,
            tools: Optional[List] = None,
            **kwargs,
        ):
            # Initialize PRECEPT mixin
            PRECEPTAgentMixin.__init__(
                self,
                memory_store=memory_store,
                context_engineering=context_engineering,
                enable_learning=enable_learning,
            )

            # Build system message with PRECEPT context
            base_message = system_message or self._get_default_system_message()
            enhanced_message = self._enhance_system_message(base_message)

            # Add PRECEPT tools to agent
            all_tools = list(tools or [])
            all_tools.extend(self._get_precept_tool_functions())

            # Initialize AutoGen agent
            AssistantAgent.__init__(
                self,
                name=name,
                model_client=model_client,
                system_message=enhanced_message,
                tools=all_tools if all_tools else None,
                **kwargs,
            )

        def _get_default_system_message(self) -> str:
            return """You are a PRECEPT-enhanced AI assistant with learning capabilities.

You have access to:
1. EPISODIC MEMORY: Store and retrieve past experiences
2. LEARNED RULES: Apply rules discovered from patterns
3. PROCEDURAL MEMORY: Use proven strategies for task types

WORKFLOW:
1. RETRIEVE: Check memories before acting
2. APPLY: Use learned rules to avoid past mistakes
3. EXECUTE: Take action based on context
4. LEARN: Store new experiences after completion

Always check your memory first and apply learned rules."""

        def _enhance_system_message(self, base_message: str) -> str:
            """Enhance system message with learned rules."""
            if self.learned_rules:
                rules_text = "\n".join(
                    f"- [{code}] {rule}"
                    for code, rule in self.learned_rules.items()
                )
                return f"""{base_message}

CRITICAL LEARNED RULES (ALWAYS APPLY):
{rules_text}"""
            return base_message

        def _get_precept_tool_functions(self) -> List[Callable]:
            """Get PRECEPT tool functions for the agent."""
            async def store_experience_tool(
                content: str,
                outcome: str,
                strategy: str = "",
                domain: str = "general"
            ) -> str:
                """Store a new experience to memory."""
                return await self.store_experience(content, outcome, strategy, domain)

            async def retrieve_memories_tool(
                query: str,
                top_k: int = 5,
                filter_success: bool = False
            ) -> str:
                """Retrieve relevant memories."""
                memories = await self.retrieve_memories(query, top_k, filter_success)
                if not memories:
                    return "No relevant memories found."
                return "\n".join(
                    f"- {m['content']} (Outcome: {m['outcome']})"
                    for m in memories
                )

            async def get_learned_rules_tool() -> str:
                """Get all learned rules."""
                rules = self.get_learned_rules()
                if not rules:
                    return "No rules learned yet."
                return "\n".join(f"[{code}] {rule}" for code, rule in rules.items())

            return [
                store_experience_tool,
                retrieve_memories_tool,
                get_learned_rules_tool,
            ]


    class PRECEPTToolExecutorAgent(UserProxyAgent, PRECEPTAgentMixin):
        """
        AutoGen UserProxyAgent enhanced for PRECEPT tool execution.

        This agent:
        - Executes tools on behalf of the assistant
        - Tracks outcomes for learning
        - Integrates with MCP servers for external tools
        """

        def __init__(
            self,
            name: str,
            tool_executor: Optional[Callable] = None,
            memory_store: Optional[MemoryStore] = None,
            enable_learning: bool = True,
            mcp_servers: Optional[List[Dict]] = None,
            **kwargs,
        ):
            # Initialize PRECEPT mixin
            PRECEPTAgentMixin.__init__(
                self,
                memory_store=memory_store,
                enable_learning=enable_learning,
            )

            self.tool_executor = tool_executor
            self.mcp_servers = mcp_servers or []
            self.execution_history: List[Dict] = []

            # Initialize AutoGen agent
            UserProxyAgent.__init__(
                self,
                name=name,
                **kwargs,
            )

        async def execute_tool(
            self,
            tool_name: str,
            tool_args: Dict,
        ) -> Dict:
            """Execute a tool and track the outcome."""
            start_time = time.time()

            try:
                # Try PRECEPT's tool executor first
                if self.tool_executor:
                    result = await self.tool_executor(tool_name, json.dumps(tool_args))
                    success = "error" not in result.lower() and "failed" not in result.lower()
                else:
                    result = f"No tool executor configured for {tool_name}"
                    success = False

                execution_record = {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                    "success": success,
                    "duration": time.time() - start_time,
                    "timestamp": time.time(),
                }
                self.execution_history.append(execution_record)

                # Learn from execution if enabled
                if self.enable_learning:
                    await self._learn_from_execution(execution_record)

                return {"success": success, "result": result}

            except Exception as e:
                return {"success": False, "result": str(e)}

        async def _learn_from_execution(self, record: Dict):
            """Learn from tool execution outcomes."""
            if not record["success"]:
                # Store failure experience
                await self.store_experience(
                    content=f"Tool {record['tool']} with args {record['args']} failed: {record['result']}",
                    outcome="failure",
                    strategy=f"Avoid: {record['args']}",
                    domain="tool_execution",
                )


# =============================================================================
# MCP SERVER IMPLEMENTATIONS
# =============================================================================

if MCP_AVAILABLE:

    def create_precept_memory_mcp_server(
        memory_store: MemoryStore,
        enable_learning: bool = True,
    ) -> FastMCP:
        """
        Create an MCP server for PRECEPT memory operations.

        This server exposes PRECEPT's memory capabilities via MCP protocol,
        allowing any MCP-compatible agent to use PRECEPT's learning system.
        """
        mcp = FastMCP("precept_memory_server")

        @mcp.tool()
        async def store_experience(
            content: str,
            outcome: str,
            strategy: str = "",
            domain: str = "general",
        ) -> str:
            """Store a new experience to PRECEPT's episodic memory.

            Args:
                content: The experience content to store
                outcome: Outcome (success, failure, partial)
                strategy: Strategy used (optional)
                domain: Domain category
            """
            if not enable_learning:
                return "Learning disabled"

            experience = Experience(
                content=content,
                experience_type=ExperienceType.LESSON if outcome == "success" else ExperienceType.FAILURE,
                outcome=outcome,
                strategy=strategy,
                domain=domain,
                correctness=1.0 if outcome == "success" else 0.0,
            )
            memory_store.store_experience(experience)
            return f"Experience stored successfully"

        @mcp.tool()
        async def retrieve_memories(query: str, top_k: int = 5) -> str:
            """Retrieve relevant memories from PRECEPT's episodic store.

            Args:
                query: Search query
                top_k: Number of results to return
            """
            memories = memory_store.retrieve_experiences(query, k=top_k)
            if not memories:
                return "No relevant memories found"

            results = []
            for m in memories:
                results.append(f"- {m.content} (Outcome: {m.outcome})")
            return "\n".join(results)

        @mcp.tool()
        async def get_memory_stats() -> str:
            """Get statistics about PRECEPT's memory store."""
            stats = memory_store.get_stats()
            return json.dumps(stats, indent=2)

        @mcp.resource("precept://memories/all")
        async def get_all_memories() -> str:
            """Get all stored memories as a resource."""
            memories = memory_store.get_all_experiences()
            return json.dumps([m.__dict__ for m in memories], indent=2, default=str)

        return mcp


    def create_precept_retrieval_mcp_server(
        memory_store: MemoryStore,
        procedural_store: Optional[ProceduralMemoryStore] = None,
    ) -> FastMCP:
        """
        Create an MCP server for PRECEPT retrieval operations.

        Provides dual retrieval (semantic + episodic) and procedural memory access.
        """
        mcp = FastMCP("precept_retrieval_server")
        procedural = procedural_store or ProceduralMemoryStore()

        @mcp.tool()
        async def semantic_search(query: str, top_k: int = 5) -> str:
            """Perform semantic search across memories.

            Args:
                query: Search query
                top_k: Number of results
            """
            memories = memory_store.retrieve_experiences(query, k=top_k)
            if not memories:
                return "No results found"
            return "\n".join(f"- {m.content}" for m in memories)

        @mcp.tool()
        async def get_procedure(task_type: str) -> str:
            """Get procedural memory (how-to) for a task type.

            Args:
                task_type: Type of task
            """
            procedures = procedural.get_procedures_for_task(task_type)
            if not procedures:
                return f"No procedure found for {task_type}"
            return procedures[0].steps_text

        @mcp.tool()
        async def store_procedure(
            name: str,
            task_type: str,
            steps: str,
            success_rate: float = 1.0,
        ) -> str:
            """Store a new procedure (strategy/playbook).

            Args:
                name: Procedure name
                task_type: Type of task this applies to
                steps: Step-by-step instructions
                success_rate: Historical success rate
            """
            from .context_engineering import Procedure
            proc = Procedure(
                name=name,
                task_type=task_type,
                steps_text=steps,
                success_rate=success_rate,
            )
            procedural.store_procedure(proc)
            return f"Procedure '{name}' stored successfully"

        return mcp


    def create_precept_learning_mcp_server(
        memory_store: MemoryStore,
        context_engineering: Optional[ContextEngineeringManager] = None,
    ) -> FastMCP:
        """
        Create an MCP server for PRECEPT learning operations.

        Provides access to GEPA evolution and consolidation triggers.
        """
        mcp = FastMCP("precept_learning_server")
        ce_manager = context_engineering

        # Track learned rules
        learned_rules: Dict[str, str] = {}
        error_patterns: Dict[str, List[str]] = {}

        @mcp.tool()
        async def record_error_pattern(error_code: str, context: str) -> str:
            """Record an error pattern for rule learning.

            Args:
                error_code: Error code encountered
                context: Context in which error occurred
            """
            if error_code not in error_patterns:
                error_patterns[error_code] = []
            error_patterns[error_code].append(context)

            count = len(error_patterns[error_code])

            # Check if we can extract a rule
            if count >= 3 and error_code not in learned_rules:
                # Use LLM to extract rule
                try:
                    response = await precept_llm_client(
                        system_prompt="Extract a rule from error patterns.",
                        user_prompt=f"Error {error_code} occurred in: {', '.join(error_patterns[error_code][-3:])}. What rule should be followed?",
                    )
                    learned_rules[error_code] = response.strip()
                    return f"Rule learned for {error_code}: {learned_rules[error_code]}"
                except Exception:
                    pass

            return f"Pattern recorded ({count} occurrences)"

        @mcp.tool()
        async def get_learned_rules() -> str:
            """Get all learned rules."""
            if not learned_rules:
                return "No rules learned yet"
            return "\n".join(f"[{code}] {rule}" for code, rule in learned_rules.items())

        @mcp.tool()
        async def trigger_consolidation() -> str:
            """Manually trigger memory consolidation."""
            if ce_manager:
                ce_manager.smart_trigger.check_consolidation_needed()
                return "Consolidation check triggered"
            return "Context engineering not configured"

        @mcp.resource("precept://rules/all")
        async def get_rules_resource() -> str:
            """Get all learned rules as a resource."""
            return json.dumps(learned_rules, indent=2)

        return mcp


# =============================================================================
# MCP CLIENT ADAPTER
# =============================================================================

class PRECEPTMCPAdapter:
    """
    Adapter for connecting PRECEPT agents to external MCP servers.

    This allows PRECEPT agents to use any MCP-compatible tool server,
    enabling scalable tool integration.
    """

    def __init__(self):
        self.connected_servers: Dict[str, Any] = {}
        self.available_tools: Dict[str, Dict] = {}
        self._exit_stack: Optional[AsyncExitStack] = None

    async def connect_server(
        self,
        name: str,
        params: Dict,
        timeout: int = 120,
    ) -> List[Dict]:
        """
        Connect to an MCP server and get available tools.

        Args:
            name: Server name for reference
            params: MCP server parameters (command, args, env)
            timeout: Connection timeout

        Returns:
            List of available tools from the server
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP not available. Install with: pip install mcp")

        # Import here to avoid issues if MCP not installed
        from agents.mcp import MCPServerStdio

        if self._exit_stack is None:
            self._exit_stack = AsyncExitStack()

        server = await self._exit_stack.enter_async_context(
            MCPServerStdio(params=params, client_session_timeout_seconds=timeout)
        )

        self.connected_servers[name] = server

        # Get available tools
        tools = await server.list_tools()
        self.available_tools[name] = tools

        return tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict,
    ) -> Any:
        """
        Call a tool on a connected MCP server.

        Args:
            server_name: Name of the connected server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if server_name not in self.connected_servers:
            raise ValueError(f"Server '{server_name}' not connected")

        server = self.connected_servers[server_name]
        result = await server.call_tool(tool_name, arguments)
        return result

    def get_tools_as_functions(self, server_name: str) -> List[Callable]:
        """
        Convert MCP tools to callable functions for AutoGen.

        Args:
            server_name: Name of the connected server

        Returns:
            List of async callable functions
        """
        if server_name not in self.available_tools:
            return []

        tools = self.available_tools[server_name]
        functions = []

        for tool in tools:
            async def tool_fn(
                _server=server_name,
                _tool=tool.name,
                **kwargs
            ):
                return await self.call_tool(_server, _tool, kwargs)

            tool_fn.__name__ = tool.name
            tool_fn.__doc__ = tool.description
            functions.append(tool_fn)

        return functions

    async def disconnect_all(self):
        """Disconnect all connected servers."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self.connected_servers.clear()
        self.available_tools.clear()


# =============================================================================
# PRECEPT TEAM (Multi-Agent Architecture)
# =============================================================================

if AUTOGEN_AVAILABLE:

    class PRECEPTTeam:
        """
        Multi-agent team with PRECEPT learning capabilities.

        Creates a team of specialized agents:
        - Planner: Strategic planning with memory retrieval
        - Executor: Tool execution with error tracking
        - Learner: Pattern analysis and rule extraction
        """

        def __init__(
            self,
            config: AutoGenPRECEPTConfig,
            memory_store: Optional[MemoryStore] = None,
            tool_executor: Optional[Callable] = None,
        ):
            self.config = config
            self.memory_store = memory_store or MemoryStore()
            self.tool_executor = tool_executor

            # Create shared context engineering manager
            self.context_engineering = ContextEngineeringManager(
                memory_store=self.memory_store,
                llm_client=precept_llm_client,
            )

            # Create model client
            self.model_client = OpenAIChatCompletionClient(
                model=config.model,
                api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            )

            # Create agents
            self.planner = self._create_planner()
            self.executor = self._create_executor()
            self.learner = self._create_learner()

            # Create team
            self.team = self._create_team()

        def _create_planner(self) -> PRECEPTAssistantAgent:
            """Create the planning agent."""
            return PRECEPTAssistantAgent(
                name="Planner",
                model_client=self.model_client,
                memory_store=self.memory_store,
                context_engineering=self.context_engineering,
                system_message="""You are the strategic planner.

Your role:
1. Analyze tasks and break them into steps
2. Retrieve relevant memories before planning
3. Apply learned rules to avoid past mistakes
4. Coordinate with Executor and Learner

Always start by checking memories and learned rules.""",
            )

        def _create_executor(self) -> PRECEPTToolExecutorAgent:
            """Create the execution agent."""
            return PRECEPTToolExecutorAgent(
                name="Executor",
                tool_executor=self.tool_executor,
                memory_store=self.memory_store,
            )

        def _create_learner(self) -> PRECEPTAssistantAgent:
            """Create the learning agent."""
            return PRECEPTAssistantAgent(
                name="Learner",
                model_client=self.model_client,
                memory_store=self.memory_store,
                context_engineering=self.context_engineering,
                system_message="""You are the learning agent.

Your role:
1. Analyze execution outcomes
2. Extract patterns from errors
3. Formulate rules from repeated patterns
4. Store successful strategies

After each task, reflect and store lessons learned.""",
            )

        def _create_team(self) -> RoundRobinGroupChat:
            """Create the team with termination conditions."""
            termination = MaxMessageTermination(max_messages=20)

            return RoundRobinGroupChat(
                participants=[self.planner, self.executor, self.learner],
                termination_condition=termination,
            )

        async def run_task(self, task: str) -> Dict:
            """
            Run a task with the PRECEPT team.

            Args:
                task: Task description

            Returns:
                Task result with learning statistics
            """
            start_time = time.time()

            # Get context for task
            context = self.planner.get_context_for_task(task)

            # Enhance task with context
            enhanced_task = f"""TASK: {task}

{context if context else 'No prior experience with similar tasks.'}

Planner: Start by retrieving memories and planning the approach.
Executor: Execute the plan steps.
Learner: After execution, extract and store any lessons learned."""

            # Run team
            result = await self.team.run(task=enhanced_task)

            return {
                "success": True,  # Determined by actual outcome
                "duration": time.time() - start_time,
                "messages": len(result.messages),
                "experiences_stored": self.planner.experiences_stored,
                "rules_learned": len(self.planner.learned_rules),
                "final_output": result.messages[-1].content if result.messages else "",
            }

        def get_team_stats(self) -> Dict:
            """Get statistics for the team."""
            return {
                "planner": {
                    "experiences_stored": self.planner.experiences_stored,
                    "memories_retrieved": self.planner.memories_retrieved,
                    "rules_applied": self.planner.rules_applied,
                },
                "executor": {
                    "executions": len(self.executor.execution_history),
                    "successes": sum(1 for e in self.executor.execution_history if e["success"]),
                },
                "learner": {
                    "rules_learned": len(self.learner.learned_rules),
                },
                "shared": {
                    "total_rules": len(self.planner.learned_rules),
                    "total_memories": len(self.memory_store.get_all_experiences()),
                },
            }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_autogen_availability() -> Dict:
    """Check if AutoGen is available and properly configured."""
    return {
        "autogen_available": AUTOGEN_AVAILABLE,
        "mcp_available": MCP_AVAILABLE,
        "error": AUTOGEN_IMPORT_ERROR if not AUTOGEN_AVAILABLE else None,
        "mcp_error": MCP_IMPORT_ERROR if not MCP_AVAILABLE else None,
    }


async def create_precept_autogen_agent(
    name: str = "PRECEPTAgent",
    config: Optional[AutoGenPRECEPTConfig] = None,
    memory_store: Optional[MemoryStore] = None,
    tool_executor: Optional[Callable] = None,
) -> "PRECEPTAssistantAgent":
    """
    Factory function to create a PRECEPT-enhanced AutoGen agent.

    Args:
        name: Agent name
        config: Configuration (uses defaults if not provided)
        memory_store: Memory store (creates new if not provided)
        tool_executor: Tool executor function

    Returns:
        Configured PRECEPTAssistantAgent
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError(
            f"AutoGen not available. Install with: pip install autogen-agentchat>=0.7.5\n"
            f"Error: {AUTOGEN_IMPORT_ERROR}"
        )

    config = config or AutoGenPRECEPTConfig()
    memory_store = memory_store or MemoryStore()

    model_client = OpenAIChatCompletionClient(
        model=config.model,
        api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
    )

    return PRECEPTAssistantAgent(
        name=name,
        model_client=model_client,
        memory_store=memory_store,
        enable_learning=config.enable_learning,
    )


async def create_precept_team(
    config: Optional[AutoGenPRECEPTConfig] = None,
    memory_store: Optional[MemoryStore] = None,
    tool_executor: Optional[Callable] = None,
) -> "PRECEPTTeam":
    """
    Factory function to create a PRECEPT multi-agent team.

    Args:
        config: Configuration
        memory_store: Shared memory store
        tool_executor: Tool executor function

    Returns:
        Configured PRECEPTTeam
    """
    if not AUTOGEN_AVAILABLE:
        raise ImportError(
            f"AutoGen not available. Install with: pip install autogen-agentchat>=0.7.5\n"
            f"Error: {AUTOGEN_IMPORT_ERROR}"
        )

    config = config or AutoGenPRECEPTConfig()
    return PRECEPTTeam(
        config=config,
        memory_store=memory_store,
        tool_executor=tool_executor,
    )


def get_mcp_server_params(server_type: str) -> Dict:
    """
    Get MCP server parameters for common server types.

    Args:
        server_type: Type of server (fetch, filesystem, memory, brave_search)

    Returns:
        MCP server parameters dict
    """
    params = {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "filesystem": lambda path: {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", path]
        },
        "brave_search": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-brave-search"],
            "env": {"BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")}
        },
        "memory": lambda name: {
            "command": "npx",
            "args": ["-y", "mcp-memory-libsql"],
            "env": {"LIBSQL_URL": f"file:./memory/{name}.db"}
        },
        "playwright": {
            "command": "npx",
            "args": ["@playwright/mcp@latest"]
        },
    }

    if server_type not in params:
        raise ValueError(f"Unknown server type: {server_type}")

    return params[server_type]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "AutoGenPRECEPTConfig",
    # Tool Definitions
    "PRECEPTToolDefinitions",
    # Mixins
    "PRECEPTAgentMixin",
    # Agents (if AutoGen available)
    "PRECEPTAssistantAgent",
    "PRECEPTToolExecutorAgent",
    # MCP Servers (if MCP available)
    "create_precept_memory_mcp_server",
    "create_precept_retrieval_mcp_server",
    "create_precept_learning_mcp_server",
    # MCP Client
    "PRECEPTMCPAdapter",
    # Team
    "PRECEPTTeam",
    # Factory Functions
    "check_autogen_availability",
    "create_precept_autogen_agent",
    "create_precept_team",
    "get_mcp_server_params",
]
