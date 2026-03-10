"""
DevOps Domain Strategy for PRECEPT.

Handles infrastructure black swan scenarios.

Black Swan Types (from black_swan_gen.py):
- Zombie_Stack: Agent tries to update a stuck CloudFormation stack
- Consistency_Race: Agent ignores IAM propagation delay
- Hidden_Policy_Block: SCP/WAF block masked as generic 403
- Pod_Eviction: Pod dies due to ephemeral storage/OOM

🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

What this strategy KNOWS (configuration, not learning):
- What stacks/roles/pods exist (from DevOpsConfig)
- What cloud providers and regions exist (vocabulary)
- How to parse tasks into structured format
- How to call MCP tools

What this strategy does NOT KNOW (must be learned):
- Which stacks are stuck
- Which recovery actions work for which errors
- What error codes mean (vague: CFN-ERR-001 doesn't reveal solution)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..config import DevOpsConfig
from ..rule_parser import DynamicRuleParser
from .base import (
    ActionResult,
    BaselineDomainStrategy,
    BlackSwanCategory,
    DomainStrategy,
    ParsedTask,
)

# AutoGen imports (optional)
try:
    from autogen_core.tools import FunctionTool

    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    FunctionTool = None


class DevOpsDomainStrategy(DomainStrategy):
    """
    DevOps domain strategy for infrastructure black swan scenarios.

    Black Swan Types (from black_swan_gen.py):
    - Zombie_Stack: Agent tries to update a stuck CloudFormation stack
    - Consistency_Race: Agent ignores IAM propagation delay
    - Hidden_Policy_Block: SCP/WAF block masked as generic 403
    - Pod_Eviction: Pod dies due to ephemeral storage/OOM

    🚨 CRITICAL: This strategy does NOT have hardcoded error->solution mappings.

    What this strategy KNOWS (configuration, not learning):
    - What stacks/roles/pods exist (from DevOpsConfig)
    - How to parse tasks into structured format
    - How to call MCP tools

    What this strategy does NOT KNOW (must be learned):
    - Which stacks are stuck or have issues
    - Which recovery actions work for which errors
    - What error codes mean

    FAIR COMPARISON: Options are presented as OPAQUE identifiers (option_a, option_b)
    so the LLM cannot infer solutions from the option names. This matches how
    logistics uses neutral port names (hamburg, antwerp) that don't reveal which works.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth
    # ═══════════════════════════════════════════════════════════════════════════
    PROVIDERS = DevOpsConfig.PROVIDERS
    REGIONS = DevOpsConfig.REGIONS
    KNOWN_STACKS = DevOpsConfig.KNOWN_STACKS
    KNOWN_ROLES = DevOpsConfig.KNOWN_ROLES
    KNOWN_PODS = DevOpsConfig.KNOWN_PODS
    STACK_STATES = DevOpsConfig.STACK_STATES

    # NO ERROR_PATTERNS here - that would be cheating!
    # The agent must LEARN which stacks are stuck.

    # ═══════════════════════════════════════════════════════════════════════════
    # OPAQUE OPTIONS - Map neutral identifiers to real actions
    # This prevents the LLM from inferring solutions from option names
    # ═══════════════════════════════════════════════════════════════════════════
    STACK_OPTIONS_MAP = {
        "strategy_a": "update",
        "strategy_b": "create",
        "strategy_c": "continue_update_rollback",
        "strategy_d": "delete_and_recreate",
        "strategy_e": "wait_for_completion",
        "strategy_f": "remove_dependencies",
    }
    IAM_OPTIONS_MAP = {
        "config_1": "immediate",
        "config_2": "delayed",
    }
    POD_OPTIONS_MAP = {
        "action_1": "describe",
        "action_2": "logs",
        "action_3": "restart",
        "action_4": "increase_resources",
    }

    # Reverse maps for internal use
    STACK_REVERSE_MAP = {v: k for k, v in STACK_OPTIONS_MAP.items()}
    IAM_REVERSE_MAP = {v: k for k, v in IAM_OPTIONS_MAP.items()}
    POD_REVERSE_MAP = {v: k for k, v in POD_OPTIONS_MAP.items()}

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the DevOps domain strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
                        - 1 = near first-try only (1 initial + 1 retry = 2 attempts)
                        - 2 = balanced (1 initial + 2 retries = 3 attempts) [default]
                        - 4 = lenient (1 initial + 4 retries = 5 attempts)
        """
        super().__init__(max_retries=max_retries)

        # Dynamic rule parser - only knows the vocabulary (stack/role/pod names)
        # Does NOT know which are stuck or which recovery actions work
        self.rule_parser = DynamicRuleParser(
            known_entities=self.PROVIDERS + self.REGIONS + self.KNOWN_STACKS
        )

        # Runtime learned knowledge (empty at start!)
        self._learned_alternatives: Dict[str, str] = {}

        # KEY LEARNING: Resource → Recovery action mapping
        # Real-world: Stacks get stuck, IAM has propagation delays, pods get OOMKilled
        self._learned_resource_recovery: Dict[str, str] = {}

    @property
    def category(self) -> BlackSwanCategory:
        return BlackSwanCategory.DEVOPS

    @property
    def domain_name(self) -> str:
        return "devops"

    def get_system_prompt(self, learned_rules: List[str] = None) -> str:
        base = """You are a DevOps engineer with PRECEPT learning capabilities.

AVAILABLE ACTIONS:
- update_stack(name): Update a CloudFormation stack
- create_role(name): Create an IAM role
- deploy_pod(name): Deploy a Kubernetes pod
- check_region(region): Check region availability

PRECEPT ADVANTAGES:
- Learns from stack failures
- Remembers IAM propagation delays
- Detects hidden policy blocks"""

        if learned_rules:
            rules_section = "\n\n═══ LEARNED RULES ═══\n"
            for i, rule in enumerate(learned_rules, 1):
                rules_section += f"{i}. {rule}\n"
            base = rules_section + base

        return base

    def get_available_actions(self) -> List[str]:
        return ["update_stack", "create_role", "deploy_pod", "check_region"]

    def get_available_entities(self) -> List[str]:
        return self.PROVIDERS + self.REGIONS

    def get_available_options(self) -> List[str]:
        """Return all opaque stack options (for general queries)."""
        return list(self.STACK_OPTIONS_MAP.keys())

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return OPAQUE options SHUFFLED for fair exploration.

        FAIR COMPARISON: Returns neutral identifiers (strategy_a, config_1, action_1)
        that don't reveal what the options do. Shuffled so both PRECEPT and
        baselines have the same random chance.
        """
        import random

        action = parsed_task.action
        if action == "create_iam_role":
            options = list(self.IAM_OPTIONS_MAP.keys())
        elif action == "debug_pod":
            options = list(self.POD_OPTIONS_MAP.keys())
        else:
            options = list(self.STACK_OPTIONS_MAP.keys())

        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str, action: str) -> str:
        """Resolve opaque option to real action name.

        Maps user-facing opaque options (strategy_a, config_1, action_1)
        back to real action names for MCP execution.
        """
        if action == "create_iam_role":
            return self.IAM_OPTIONS_MAP.get(opaque_option, opaque_option)
        elif action == "debug_pod":
            return self.POD_OPTIONS_MAP.get(opaque_option, opaque_option)
        else:
            return self.STACK_OPTIONS_MAP.get(opaque_option, opaque_option)

    def parse_task(self, task: str) -> ParsedTask:
        task_lower = task.lower()

        # ═══════════════════════════════════════════════════════════════════════════
        # ENTITY-FIRST PARSING: Check for explicit entity names BEFORE keywords
        # This prevents "IAM policy modification" from being mistaken for IAM role task
        # ═══════════════════════════════════════════════════════════════════════════
        action = "deploy_stack"
        entity = "stack"
        found_entity_type = None

        # 1. First, check for explicit stack names (highest priority)
        for stack in self.KNOWN_STACKS:
            if stack.lower() in task_lower or stack in task:
                action = "deploy_stack"
                entity = stack
                found_entity_type = "stack"
                break

        # 2. Check for explicit role names
        if not found_entity_type:
            for role in self.KNOWN_ROLES:
                if role.lower() in task_lower or role in task:
                    action = "create_iam_role"
                    entity = role
                    found_entity_type = "role"
                    break

        # 3. Check for explicit pod names
        if not found_entity_type:
            for pod in self.KNOWN_PODS:
                if pod.lower() in task_lower or pod in task:
                    action = "debug_pod"
                    entity = pod
                    found_entity_type = "pod"
                    break

        # 4. Fall back to keyword detection ONLY if no explicit entity found
        if not found_entity_type:
            if "pod" in task_lower or "kubernetes" in task_lower or "k8s" in task_lower:
                action = "debug_pod"
            elif "role" in task_lower and "policy" not in task_lower:
                # Only treat as IAM role if it mentions "role" without "policy"
                # "IAM policy modification" is about stacks, not roles!
                action = "create_iam_role"
            # Default is deploy_stack

        # Detect action modifier
        action_modifier = "update"  # Default for stacks
        if "continue" in task_lower or "rollback" in task_lower:
            action_modifier = "continue_update_rollback"
        elif "create" in task_lower and action == "create_iam_role":
            action_modifier = "create"

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z]
        # ═══════════════════════════════════════════════════════════════════
        import re

        condition_key = None
        conditions = []
        condition_match = re.search(r"\[Conditions:\s*([^\]]+)\]", task, re.IGNORECASE)
        if condition_match:
            cond_str = condition_match.group(1)
            conditions = [c.strip() for c in cond_str.split("+")]
            # Generate deterministic key (sorted, joined with +)
            condition_key = "+".join(sorted(conditions))

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source=action_modifier,  # For stacks: update/continue_update_rollback
            parameters={
                "action": action_modifier,
                "entity": entity,
                "condition_key": condition_key,  # Multi-condition key for rule storage
                "conditions": conditions,  # Individual conditions
            },
        )

    def apply_learned_rules(
        self,
        parsed_task: ParsedTask,
        rules: List[str],
    ) -> Tuple[ParsedTask, bool, str]:
        """
        Apply learned rules - THE KEY PRECEPT ADVANTAGE!

        Real-world: After learning that prod-api-stack needs continue_update_rollback,
        PRECEPT applies this recovery action FIRST on subsequent deployments.
        """
        entity = parsed_task.entity

        # Check local learned cache (resource → recovery_action)
        if entity in self._learned_resource_recovery:
            recovery = self._learned_resource_recovery[entity]
            parsed_task.parameters["action"] = recovery
            parsed_task.source = recovery
            return parsed_task, True, f"Learned:{entity}→{recovery}"

        return parsed_task, False, "Exploration"

    async def execute_action(
        self,
        mcp_client: Any,
        parsed_task: ParsedTask,
    ) -> ActionResult:
        """Execute DevOps action via MCP server tools with REAL blocking logic.

        Uses preferred_solution from LLM reasoning (Tier 2) when available.
        This aligns with the logistics domain benchmark pattern.

        OPAQUE OPTIONS: The LLM suggests opaque options (strategy_a, config_1, action_1)
        which we resolve to real actions before MCP execution.
        """
        action = parsed_task.action
        # ═══════════════════════════════════════════════════════════════
        # BUGFIX: Guard against None entity which causes Pydantic
        # validation error at the MCP tool level. Test scenarios may use
        # entity names not in the known vocabulary, causing parse_task
        # to set parsed_task.entity=None.
        # ═══════════════════════════════════════════════════════════════
        entity = parsed_task.entity or "stack"
        used_strategy = None

        # ═══════════════════════════════════════════════════════════════════
        # CRITICAL: Read preferred_solution from LLM reasoning (Tier 2)
        # The LLM suggests opaque options (strategy_a, config_1) which we
        # resolve to real actions here.
        # ═══════════════════════════════════════════════════════════════════
        preferred = parsed_task.parameters.get("preferred_solution")
        action_modifier = parsed_task.parameters.get("action", "update")

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use hash-based enforcement where
        # Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # ═══════════════════════════════════════════════════════════════════
        condition_key = parsed_task.parameters.get("condition_key")

        try:
            if action == "deploy_stack":
                # Resolve opaque option to real action
                if preferred:
                    resolved = self._resolve_option(preferred, action)
                    if resolved in DevOpsConfig.STACK_ACTIONS:
                        action_modifier = resolved
                used_strategy = action_modifier
                if condition_key:
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_devops_multi_condition",
                        {
                            "condition_key": condition_key,
                            "strategy": action_modifier,
                            "resource_name": entity,
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "deploy_stack",
                        {"stack_name": entity, "action": action_modifier},
                    )
            elif action == "create_iam_role":
                # Resolve opaque option to real action
                use_immediately = True  # default (strategy_a equivalent)
                if preferred:
                    resolved = self._resolve_option(preferred, action)
                    if resolved == "delayed":
                        use_immediately = False
                    elif resolved == "immediate":
                        use_immediately = True
                elif parsed_task.parameters.get("action") == "delayed":
                    use_immediately = False
                used_strategy = "immediate" if use_immediately else "delayed"
                response = await mcp_client.call_tool(
                    "create_iam_role",
                    {"role_name": entity, "use_immediately": use_immediately},
                )
            elif action == "debug_pod":
                # Resolve opaque option to real action
                pod_action = "describe"  # default (action_1)
                if preferred:
                    resolved = self._resolve_option(preferred, action)
                    if resolved in [
                        "describe",
                        "logs",
                        "restart",
                        "increase_resources",
                    ]:
                        pod_action = resolved
                elif parsed_task.parameters.get("action"):
                    pod_action = parsed_task.parameters.get("action")
                used_strategy = pod_action
                response = await mcp_client.call_tool(
                    "debug_pod", {"pod_name": entity, "action": pod_action}
                )
            else:
                return ActionResult(
                    success=False,
                    response=f"Unknown action: {action}",
                    error_code="UNKNOWN-ACTION",
                )

            response_str = str(response) if response else ""

            if "SUCCESS" in response_str:
                return ActionResult(
                    success=True,
                    response=response_str,
                    strategy_used=used_strategy or f"{action}:{entity}",
                )

            # Extract error code - use VAGUE codes only (CFN-XXX, IAM-XXX, K8S-XXX)
            # NO hardcoded detection of revealing codes!
            import re

            error_code = "DEVOPS-ERROR"
            match = re.search(r"Error code: (\S+)", response_str)
            if match:
                error_code = match.group(1)

            return ActionResult(
                success=False,
                response=response_str,
                error_code=error_code,
                strategy_used=f"{action}:{entity}",
            )

        except Exception as e:
            return ActionResult(
                success=False,
                response=f"MCP call failed: {str(e)}",
                error_code="MCP-ERROR",
            )

    async def handle_error(
        self,
        mcp_client: Any,
        error_code: str,
        parsed_task: ParsedTask,
        context: Dict[str, Any],
    ) -> ActionResult:
        """
        Handle DevOps errors - TRUE LEARNING without hardcoded mappings.

        🚨 CRITICAL: This method does NOT know which recovery action works!
        It tries all available options and LEARNS which one succeeds.

        FAIR COMPARISON: Limited to MAX_RETRIES (same budget as baseline)
        The difference: After learning, PRECEPT knows which action works!

        Available recovery actions per resource type:
        - Stacks: update, create, continue_update_rollback, delete_and_recreate
        - IAM: use_immediately=True, use_immediately=False
        - Pods: describe, logs, restart, increase_resources
        """
        import random

        await mcp_client.record_error(
            error_code, f"{parsed_task.action} {parsed_task.entity}"
        )

        action = parsed_task.action
        entity = parsed_task.entity

        # Track retries for fair comparison
        retries_made = context.get("retries_made", 0)

        # Get condition_key for multi-condition enforcement
        condition_key = (parsed_task.parameters or {}).get("condition_key")

        if action == "deploy_stack":
            # ═══════════════════════════════════════════════════════════════════
            # Use OPAQUE options for fair comparison (same as finance fix)
            # ═══════════════════════════════════════════════════════════════════
            all_opaque_options = list(self.STACK_OPTIONS_MAP.keys())
            tried_options = context.get("tried_options", set())

            # Add initially tried option
            initial_option = parsed_task.parameters.get("preferred_solution")
            if initial_option:
                tried_options.add(initial_option)

            remaining = [o for o in all_opaque_options if o not in tried_options]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            # Known working options FIRST, then random (fair comparison)
            known_working = [
                o
                for o in remaining
                if entity in self._learned_resource_recovery
                and self._learned_resource_recovery[entity] == o
            ]
            unknown = [o for o in remaining if o not in known_working]
            random.shuffle(unknown)  # Random order like baseline!
            remaining = known_working + unknown

            for opaque_option in remaining:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_options.add(opaque_option)
                retries_made += 1

                # Resolve opaque option to real strategy for MCP call
                resolved_strategy = self._resolve_option(opaque_option, action)

                try:
                    # ═══════════════════════════════════════════════════════
                    # CONSISTENT WITH LOGISTICS: Use BASE tool directly
                    # ═══════════════════════════════════════════════════════
                    response = await mcp_client.call_tool(
                        "deploy_stack",
                        {"stack_name": entity, "action": resolved_strategy},
                    )

                    if "SUCCESS" in str(response):
                        # 🎓 LEARN: This opaque option works!
                        self._learned_resource_recovery[entity] = opaque_option

                        # Store the OPAQUE option name for rule learning
                        rule_key = condition_key if condition_key else error_code

                        await mcp_client.record_solution(
                            error_code=rule_key,
                            solution=opaque_option,  # Store OPAQUE option, not resolved
                            context=f"Stack {entity} (conditions: {condition_key or error_code})",
                        )
                        return ActionResult(
                            success=True,
                            response=str(response),
                            strategy_used=f"stack:{opaque_option} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                except Exception:
                    continue

        elif action == "create_iam_role":
            # ═══════════════════════════════════════════════════════════════════
            # Use OPAQUE options for fair comparison
            # ═══════════════════════════════════════════════════════════════════
            all_opaque_options = list(self.IAM_OPTIONS_MAP.keys())
            tried_options = context.get("tried_options", set())

            initial_option = parsed_task.parameters.get("preferred_solution")
            if initial_option:
                tried_options.add(initial_option)

            remaining = [o for o in all_opaque_options if o not in tried_options]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            known_working = [
                o
                for o in remaining
                if entity in self._learned_resource_recovery
                and self._learned_resource_recovery[entity] == o
            ]
            unknown = [o for o in remaining if o not in known_working]
            random.shuffle(unknown)
            remaining = known_working + unknown

            for opaque_option in remaining:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_options.add(opaque_option)
                retries_made += 1

                # Resolve opaque option to real strategy
                resolved_strategy = self._resolve_option(opaque_option, action)
                use_immediately = resolved_strategy == "immediate"

                try:
                    # CONSISTENT WITH LOGISTICS: Use BASE tool directly
                    response = await mcp_client.call_tool(
                        "create_iam_role",
                        {"role_name": entity, "use_immediately": use_immediately},
                    )

                    if "SUCCESS" in str(response):
                        # 🎓 LEARN: This opaque option works!
                        self._learned_resource_recovery[entity] = opaque_option

                        rule_key = condition_key if condition_key else error_code
                        await mcp_client.record_solution(
                            error_code=rule_key,
                            solution=opaque_option,
                            context=f"Role {entity} (conditions: {condition_key or error_code})",
                        )
                        return ActionResult(
                            success=True,
                            response=str(response),
                            strategy_used=f"iam:{opaque_option} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                except Exception:
                    continue

        elif action == "debug_pod":
            # ═══════════════════════════════════════════════════════════════════
            # Use OPAQUE options for fair comparison
            # ═══════════════════════════════════════════════════════════════════
            all_opaque_options = list(self.POD_OPTIONS_MAP.keys())
            tried_options = context.get("tried_options", set())

            initial_option = parsed_task.parameters.get("preferred_solution")
            if initial_option:
                tried_options.add(initial_option)

            remaining = [o for o in all_opaque_options if o not in tried_options]

            # PRECEPT's TRUE ADVANTAGE: Use learned knowledge!
            known_working = [
                o
                for o in remaining
                if entity in self._learned_resource_recovery
                and self._learned_resource_recovery[entity] == o
            ]
            unknown = [o for o in remaining if o not in known_working]
            random.shuffle(unknown)  # Random order like baseline!
            remaining = known_working + unknown

            for opaque_option in remaining:
                if retries_made >= self.MAX_RETRIES:
                    break

                tried_options.add(opaque_option)
                retries_made += 1

                # Resolve opaque option to real action
                resolved_action = self._resolve_option(opaque_option, action)

                try:
                    # CONSISTENT WITH LOGISTICS: Use BASE tool directly
                    response = await mcp_client.call_tool(
                        "debug_pod",
                        {"pod_name": entity, "action": resolved_action},
                    )

                    if "SUCCESS" in str(response):
                        # 🎓 LEARN: This opaque option works!
                        self._learned_resource_recovery[entity] = opaque_option

                        rule_key = condition_key if condition_key else error_code
                        await mcp_client.record_solution(
                            error_code=rule_key,
                            solution=opaque_option,  # Store OPAQUE option
                            context=f"Pod {entity} (conditions: {condition_key or error_code})",
                        )
                        return ActionResult(
                            success=True,
                            response=str(response),
                            strategy_used=f"pod:{opaque_option} (retry {retries_made}/{self.MAX_RETRIES})",
                        )
                except Exception:
                    continue

        return ActionResult(
            success=False,
            response=f"All retries exhausted ({retries_made}/{self.MAX_RETRIES})",
            strategy_used="Failed",
        )

    def _get_domain_tools(self, mcp_client: Any) -> List[Any]:
        return []


class DevOpsBaselineStrategy(BaselineDomainStrategy):
    """
    DevOps baseline strategy - NO LEARNING.

    Behavior:
    - Always tries default action first (update for stacks)
    - On failure, tries RANDOM alternative actions
    - Does NOT know which stacks are stuck
    - Does NOT learn from failures

    FAIR COMPARISON: Uses same OPAQUE options as PRECEPT so the LLM cannot
    infer solutions from option names.
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIGURATION - Import from single source of truth (same as PRECEPT)
    # ═══════════════════════════════════════════════════════════════════════════
    STACK_ACTIONS = DevOpsConfig.STACK_ACTIONS
    KNOWN_STACKS = DevOpsConfig.KNOWN_STACKS
    KNOWN_ROLES = DevOpsConfig.KNOWN_ROLES
    KNOWN_PODS = DevOpsConfig.KNOWN_PODS

    # ═══════════════════════════════════════════════════════════════════════════
    # OPAQUE OPTIONS - Same mapping as PRECEPT for fair comparison
    # ═══════════════════════════════════════════════════════════════════════════
    STACK_OPTIONS_MAP = DevOpsDomainStrategy.STACK_OPTIONS_MAP
    IAM_OPTIONS_MAP = DevOpsDomainStrategy.IAM_OPTIONS_MAP
    POD_OPTIONS_MAP = DevOpsDomainStrategy.POD_OPTIONS_MAP

    def __init__(self, max_retries: Optional[int] = None):
        """
        Initialize the DevOps baseline strategy.

        Args:
            max_retries: Maximum number of retries allowed. If None, uses DEFAULT_MAX_RETRIES.
        """
        super().__init__(max_retries=max_retries)

    @property
    def domain_name(self) -> str:
        return "devops"

    def get_available_options(self) -> List[str]:
        """Return all opaque stack options (for general queries)."""
        return list(self.STACK_OPTIONS_MAP.keys())

    def get_options_for_task(self, parsed_task: ParsedTask) -> List[str]:
        """Return OPAQUE options SHUFFLED for fair exploration. Same as PRECEPT."""
        import random

        action = parsed_task.action
        if action == "create_iam_role":
            options = list(self.IAM_OPTIONS_MAP.keys())
        elif action == "debug_pod":
            options = list(self.POD_OPTIONS_MAP.keys())
        else:
            options = list(self.STACK_OPTIONS_MAP.keys())

        random.shuffle(options)
        return options

    def _resolve_option(self, opaque_option: str, action: str) -> str:
        """Resolve opaque option to real action name."""
        if action == "create_iam_role":
            return self.IAM_OPTIONS_MAP.get(opaque_option, opaque_option)
        elif action == "debug_pod":
            return self.POD_OPTIONS_MAP.get(opaque_option, opaque_option)
        else:
            return self.STACK_OPTIONS_MAP.get(opaque_option, opaque_option)

    def parse_task(self, task: str) -> ParsedTask:
        """Parse task and extract entity (entity-first detection)."""
        task_lower = task.lower()

        action = "deploy_stack"
        entity = "stack"
        found_entity_type = None

        # 1. First, check for explicit stack names (highest priority)
        for stack in self.KNOWN_STACKS:
            if stack.lower() in task_lower or stack in task:
                action = "deploy_stack"
                entity = stack
                found_entity_type = "stack"
                break

        # 2. Check for explicit role names
        if not found_entity_type:
            for role in self.KNOWN_ROLES:
                if role.lower() in task_lower or role in task:
                    action = "create_iam_role"
                    entity = role
                    found_entity_type = "role"
                    break

        # 3. Check for explicit pod names
        if not found_entity_type:
            for pod in self.KNOWN_PODS:
                if pod.lower() in task_lower or pod in task:
                    action = "debug_pod"
                    entity = pod
                    found_entity_type = "pod"
                    break

        # 4. Fall back to keyword detection only if no explicit entity
        if not found_entity_type:
            if "pod" in task_lower or "kubernetes" in task_lower or "k8s" in task_lower:
                action = "debug_pod"
            elif "role" in task_lower and "policy" not in task_lower:
                action = "create_iam_role"

        # Build parameters
        parameters = {"entity": entity}

        # ═══════════════════════════════════════════════════════════════════
        # MULTI-CONDITION EXTRACTION: Parse [Conditions: X + Y + Z] pattern
        # CRITICAL: Must match PRECEPT strategy to ensure fair comparison
        # ═══════════════════════════════════════════════════════════════════
        import re

        condition_key_match = re.search(r"\[Conditions:\s*(.+?)\]", task)
        if condition_key_match:
            cond_str = condition_key_match.group(1).strip()
            conditions = [c.strip() for c in cond_str.split("+")]
            condition_key = "+".join(sorted(conditions))  # SORTED like PRECEPT!
            parameters["condition_key"] = condition_key
            parameters["conditions"] = conditions

        return ParsedTask(
            raw_task=task,
            action=action,
            entity=entity,
            source="update",  # Default action
            parameters=parameters,
        )

    def get_default_option(self, parsed_task: ParsedTask) -> str:
        """Return default opaque option for the task type."""
        action = parsed_task.action
        if action == "create_iam_role":
            return "config_1"  # Default: immediate
        elif action == "debug_pod":
            return "action_1"  # Default: describe
        else:
            return "strategy_a"  # Default: update for stacks

    async def execute_action(
        self,
        mcp_client: Any,
        option: str,
        parsed_task: ParsedTask,
    ) -> Tuple[bool, str]:
        """Execute via MCP server with REAL blocking logic.

        OPAQUE OPTIONS: Resolves opaque option names (strategy_a, config_1, action_1)
        to real action names before MCP execution.

        BLACK SWAN CSP: For multi-condition scenarios, uses hash-based enforcement
        where Solution = f(hash(composite_key)). This is FAIR - baselines face the
        same strict enforcement as PRECEPT.
        """
        action = parsed_task.action
        entity = parsed_task.entity
        condition_key = parsed_task.parameters.get("condition_key")

        # Resolve opaque option to real action
        resolved_option = self._resolve_option(option, action)

        # ═══════════════════════════════════════════════════════════════════
        # BLACK SWAN CSP ENFORCEMENT:
        # For multi-condition scenarios, use the multi-condition tool which
        # enforces Solution = f(hash(composite_key)) - ONLY ONE solution works!
        # This is FAIR - baselines face the same strict enforcement as PRECEPT.
        # ═══════════════════════════════════════════════════════════════════
        try:
            if action == "deploy_stack":
                if condition_key:
                    # BUGFIX: Guard against None entity which causes Pydantic
                    # validation error at the MCP tool level (same pattern as
                    # the logistics destination fix).
                    resource_name = entity or "stack"
                    # Multi-condition: use hash-based enforcement
                    response = await mcp_client.call_tool(
                        "execute_devops_multi_condition",
                        {
                            "condition_key": condition_key,
                            "strategy": resolved_option,
                            "resource_name": resource_name,
                        },
                    )
                else:
                    # Single-condition: use base tool
                    response = await mcp_client.call_tool(
                        "deploy_stack",
                        {"stack_name": entity or "stack", "action": resolved_option},
                    )
            elif action == "create_iam_role":
                # Resolved option is "immediate" or "delayed"
                use_immediately = resolved_option != "delayed"
                response = await mcp_client.call_tool(
                    "create_iam_role",
                    {"role_name": entity, "use_immediately": use_immediately},
                )
            elif action == "debug_pod":
                # Resolved option is "describe", "logs", "restart", or "increase_resources"
                pod_action = (
                    resolved_option
                    if resolved_option
                    in ["describe", "logs", "restart", "increase_resources"]
                    else "describe"
                )
                response = await mcp_client.call_tool(
                    "debug_pod", {"pod_name": entity, "action": pod_action}
                )
            else:
                return False, f"Unknown action: {action}"

            if isinstance(response, str) and "SUCCESS" in response:
                return True, response
            else:
                return False, str(response)

        except Exception as e:
            return False, f"MCP call failed: {str(e)}"
