"""
CSP Constraint Manager for PRECEPT Framework.

This module provides specialized capabilities for handling Constraint Satisfaction
Problems (CSPs), particularly Black Swan events with interdependent and contradicting
constraints.

Key Components:
1. CSPConstraintManager - Dependency graph mapping (Evo-Memory)
2. RefineInterceptor - Real-time constraint discovery from execution feedback
3. ConstraintHierarchy - Physics > Policy > Instruction resolution (COMPASS)
4. CausalChainTracker - Forward checking for interdependencies
5. ConflictResolver - Detects and resolves contradicting constraints

Scientific Basis:
- PRECEPT solves problems TOPOLOGICALLY, not linearly
- Builds a map of "Minefields" (Interdependencies)
- Uses hierarchy of laws to resolve "Stalemates" (Contradictions)

Reference: PRECEPT CSP Solving Capabilities
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONSTRAINT TIERS (Hierarchy of Law)
# =============================================================================

class ConstraintTier(Enum):
    """
    Constraint tiers from lowest to highest priority.

    COMPASS uses this hierarchy to resolve conflicts:
    - Physics ALWAYS overrides Policy and Instruction
    - Policy overrides Instruction
    - Instruction is the lowest priority (can be negotiated)
    """
    INSTRUCTION = 1  # User requests, preferences (can be negotiated)
    POLICY = 2       # Security rules, budget, organizational rules
    PHYSICS = 3      # Network status, file permissions, OS architecture (IMMUTABLE)


class ConstraintType(Enum):
    """Types of constraints for CSP classification."""
    HARD = auto()        # Must be satisfied, no exceptions
    SOFT = auto()        # Should be satisfied, can be relaxed
    INTERDEPENDENT = auto()  # Coupled with other constraints
    CONTRADICTING = auto()   # Conflicts with another constraint


@dataclass
class Constraint:
    """
    A constraint in the CSP domain.

    Attributes:
        id: Unique identifier (e.g., "C01", "NETWORK_DOWN")
        name: Human-readable name
        tier: Hierarchy level (Physics > Policy > Instruction)
        type: Hard, Soft, Interdependent, or Contradicting
        description: Detailed description
        dependencies: List of constraint IDs this depends on
        conflicts_with: List of constraint IDs this conflicts with
        solution_patterns: Patterns that satisfy this constraint
        discovered: Whether this constraint has been discovered
        satisfied: Whether this constraint has been satisfied
    """
    id: str
    name: str
    tier: ConstraintTier
    type: ConstraintType
    description: str
    dependencies: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)
    solution_patterns: List[str] = field(default_factory=list)
    discovered: bool = False
    satisfied: bool = False
    discovered_at: Optional[float] = None
    satisfied_at: Optional[float] = None


# =============================================================================
# SECTION 2: CAUSAL CHAIN TRACKER (Evo-Memory for Dependencies)
# =============================================================================

@dataclass
class CausalChain:
    """
    Represents a causal chain of constraint failures.

    Example: Network Down → Pip Fails → Git Fails → Apt Fails
    Stored in Evo-Memory for forward checking.
    """
    root_constraint: str
    triggered_constraints: List[str]
    discovered_at: float = field(default_factory=time.time)
    frequency: int = 1  # How often this chain has been observed

    def to_dict(self) -> Dict:
        return {
            "root": self.root_constraint,
            "triggers": self.triggered_constraints,
            "discovered_at": self.discovered_at,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CausalChain':
        return cls(
            root_constraint=data["root"],
            triggered_constraints=data["triggers"],
            discovered_at=data.get("discovered_at", time.time()),
            frequency=data.get("frequency", 1),
        )


class CausalChainTracker:
    """
    Tracks causal chains of constraint failures for forward checking.

    Evo-Memory capability: Stores causal chains (trajectories), not just facts.
    Enables PRECEPT to perform forward checking and prune impossible branches.

    Example:
        If NETWORK_DOWN is detected, immediately pre-activate:
        - PIP_BLOCKED
        - GIT_BLOCKED
        - APT_BLOCKED

    This provides O(1) dependency resolution instead of O(n) sequential testing.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.chains: Dict[str, CausalChain] = {}
        self.persistence_path = persistence_path or Path("data/causal_chains.json")
        self._load()

    def _load(self) -> None:
        """Load causal chains from disk."""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    data = json.load(f)
                    for chain_data in data.get("chains", []):
                        chain = CausalChain.from_dict(chain_data)
                        self.chains[chain.root_constraint] = chain
                logger.info(f"Loaded {len(self.chains)} causal chains")
            except Exception as e:
                logger.warning(f"Failed to load causal chains: {e}")

    def save(self) -> None:
        """Persist causal chains to disk."""
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persistence_path, 'w') as f:
            json.dump({
                "chains": [c.to_dict() for c in self.chains.values()],
                "last_updated": time.time(),
            }, f, indent=2)

    def record_chain(self, root: str, triggered: List[str]) -> None:
        """
        Record a causal chain observation.

        Called when we discover that constraint X triggers constraints Y, Z, etc.
        """
        if root in self.chains:
            # Update existing chain
            chain = self.chains[root]
            chain.frequency += 1
            # Merge triggered constraints
            for t in triggered:
                if t not in chain.triggered_constraints:
                    chain.triggered_constraints.append(t)
        else:
            # Create new chain
            self.chains[root] = CausalChain(
                root_constraint=root,
                triggered_constraints=triggered,
            )
        self.save()

    def get_triggered_constraints(self, root: str) -> List[str]:
        """
        Get all constraints triggered by a root constraint.

        This is the FORWARD CHECKING capability:
        "If A is broken, I know B and C are also broken. Skip them."
        """
        if root in self.chains:
            return self.chains[root].triggered_constraints
        return []

    def get_all_chains(self) -> List[CausalChain]:
        """Get all recorded causal chains."""
        return list(self.chains.values())


# =============================================================================
# SECTION 3: REFINE INTERCEPTOR (Real-time Constraint Discovery)
# =============================================================================

@dataclass
class ExecutionFeedback:
    """
    Feedback from a tool execution for constraint discovery.

    The Refine Interceptor monitors these signals to discover hidden constraints.
    """
    return_code: int
    stdout: str
    stderr: str
    duration: float
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class RefineInterceptor:
    """
    Real-time feedback loop for constraint discovery.

    Monitors stderr and return codes to discover hidden constraints dynamically.
    Updates the constraint graph as new constraints are discovered.

    From the description:
    "The Refine Interceptor acts as a Real-Time Feedback Loop.
     It monitors stderr and return codes of every tool execution."
    """

    # Pattern mapping: stderr patterns → constraint IDs
    CONSTRAINT_PATTERNS = {
        # Network-related
        "network is unreachable": ("NETWORK_DOWN", ConstraintTier.PHYSICS, ["PIP_BLOCKED", "GIT_BLOCKED", "APT_BLOCKED"]),
        "connection refused": ("NETWORK_DOWN", ConstraintTier.PHYSICS, ["PIP_BLOCKED", "GIT_BLOCKED"]),
        "no route to host": ("NETWORK_DOWN", ConstraintTier.PHYSICS, ["PIP_BLOCKED", "GIT_BLOCKED"]),
        "name resolution failed": ("DNS_FAILED", ConstraintTier.PHYSICS, ["NETWORK_DOWN"]),

        # File system
        "read-only file system": ("READONLY_FS", ConstraintTier.PHYSICS, ["WRITE_BLOCKED"]),
        "permission denied": ("PERMISSION_DENIED", ConstraintTier.POLICY, []),
        "no space left on device": ("DISK_FULL", ConstraintTier.PHYSICS, ["WRITE_BLOCKED"]),

        # Process/resource
        "address already in use": ("PORT_BLOCKED", ConstraintTier.PHYSICS, []),
        "resource temporarily unavailable": ("RESOURCE_LOCKED", ConstraintTier.PHYSICS, []),
        "lock file exists": ("RESOURCE_LOCKED", ConstraintTier.PHYSICS, []),

        # Package management
        "could not find a version": ("PACKAGE_NOT_FOUND", ConstraintTier.PHYSICS, []),
        "dependency conflict": ("DEPENDENCY_CONFLICT", ConstraintTier.PHYSICS, []),

        # Authentication/Authorization
        "authentication required": ("AUTH_REQUIRED", ConstraintTier.POLICY, []),
        "access denied": ("ACCESS_DENIED", ConstraintTier.POLICY, []),

        # Environment
        "command not found": ("TOOL_MISSING", ConstraintTier.PHYSICS, []),
        "module not found": ("IMPORT_FAILED", ConstraintTier.PHYSICS, []),
    }

    def __init__(self, causal_tracker: Optional[CausalChainTracker] = None):
        self.causal_tracker = causal_tracker or CausalChainTracker()
        self.discovered_constraints: Dict[str, Constraint] = {}
        self.active_constraints: Set[str] = set()

    def intercept(self, feedback: ExecutionFeedback) -> List[Constraint]:
        """
        Intercept execution feedback and discover constraints.

        Returns list of newly discovered constraints.
        """
        discovered = []
        combined_output = f"{feedback.stdout}\n{feedback.stderr}".lower()

        for pattern, (constraint_id, tier, triggers) in self.CONSTRAINT_PATTERNS.items():
            if pattern in combined_output:
                if constraint_id not in self.discovered_constraints:
                    constraint = Constraint(
                        id=constraint_id,
                        name=constraint_id.replace("_", " ").title(),
                        tier=tier,
                        type=ConstraintType.HARD if tier == ConstraintTier.PHYSICS else ConstraintType.SOFT,
                        description=f"Discovered from execution feedback: {pattern}",
                        dependencies=triggers,
                        discovered=True,
                        discovered_at=time.time(),
                    )
                    self.discovered_constraints[constraint_id] = constraint
                    self.active_constraints.add(constraint_id)
                    discovered.append(constraint)

                    # Record causal chain if there are triggers
                    if triggers:
                        self.causal_tracker.record_chain(constraint_id, triggers)
                        # Pre-activate triggered constraints (forward checking)
                        for triggered_id in triggers:
                            self.active_constraints.add(triggered_id)

        return discovered

    def get_active_constraints(self) -> Set[str]:
        """Get all currently active constraints."""
        return self.active_constraints

    def get_constraint_context(self) -> str:
        """
        Get constraint context for LLM injection.

        Context Engineering: Transform latent state into active tokens.
        """
        if not self.active_constraints:
            return ""

        lines = ["SYSTEM CONSTRAINTS (DISCOVERED - DO NOT VIOLATE):"]
        for c_id in sorted(self.active_constraints):
            if c_id in self.discovered_constraints:
                c = self.discovered_constraints[c_id]
                tier_label = c.tier.name
                lines.append(f"  [{tier_label}] {c_id}: {c.description}")

                # Add triggered constraints (forward checking hint)
                triggered = self.causal_tracker.get_triggered_constraints(c_id)
                if triggered:
                    lines.append(f"       → Also blocked: {', '.join(triggered)}")

        return "\n".join(lines)

    def clear_constraint(self, constraint_id: str) -> None:
        """Mark a constraint as no longer active (resolved)."""
        self.active_constraints.discard(constraint_id)


# =============================================================================
# SECTION 4: CONFLICT RESOLVER (COMPASS Hierarchical Judge)
# =============================================================================

@dataclass
class ConflictResolution:
    """Result of conflict resolution."""
    conflicting_constraints: List[str]
    winner: str
    loser: str
    resolution_strategy: str
    negotiated_action: Optional[str] = None


class ConflictResolver:
    """
    COMPASS-style hierarchical conflict resolution.

    Resolves contradicting constraints using the Hierarchy of Law:
    1. Physics (Immutable) - Always wins
    2. Policy (Hard) - Overrides Instruction
    3. Instruction (Soft) - Can be negotiated

    Example:
        User: "Download the dataset"  (Instruction, Tier 1)
        Physics: "No Internet Access"  (Physics, Tier 3)
        Resolution: Physics wins → Negotiate: "Generate synthetic dataset instead"
    """

    # Negotiation strategies for when Instruction loses
    NEGOTIATION_STRATEGIES = {
        "NETWORK_DOWN": {
            "download": "Use cached version or generate synthetic data",
            "install": "Use offline packages or conda",
            "fetch": "Use local copy if available",
        },
        "READONLY_FS": {
            "write": "Write to /tmp instead",
            "save": "Save to user's home directory",
            "create": "Create in temporary directory",
        },
        "PORT_BLOCKED": {
            "listen": "Use alternative port (50000+)",
            "serve": "Use different port or socket file",
        },
        "PERMISSION_DENIED": {
            "execute": "Request elevated permissions or use alternative",
            "access": "Use accessible path or request access",
        },
    }

    def __init__(self, constraints: Dict[str, Constraint]):
        self.constraints = constraints

    def detect_conflicts(self) -> List[Tuple[str, str]]:
        """
        Detect all conflicting constraint pairs.

        Returns list of (constraint_a, constraint_b) tuples.
        """
        conflicts = []
        constraint_list = list(self.constraints.values())

        for i, c1 in enumerate(constraint_list):
            for c2 in constraint_list[i+1:]:
                if c1.id in c2.conflicts_with or c2.id in c1.conflicts_with:
                    conflicts.append((c1.id, c2.id))

        return conflicts

    def resolve_conflict(self, constraint_a: str, constraint_b: str,
                        user_goal: Optional[str] = None) -> ConflictResolution:
        """
        Resolve conflict between two constraints using hierarchy.

        Physics > Policy > Instruction
        """
        c_a = self.constraints.get(constraint_a)
        c_b = self.constraints.get(constraint_b)

        if not c_a or not c_b:
            raise ValueError(f"Constraint not found: {constraint_a} or {constraint_b}")

        # Determine winner based on tier
        if c_a.tier.value > c_b.tier.value:
            winner, loser = c_a, c_b
        elif c_b.tier.value > c_a.tier.value:
            winner, loser = c_b, c_a
        else:
            # Same tier - prefer HARD over SOFT
            if c_a.type == ConstraintType.HARD:
                winner, loser = c_a, c_b
            else:
                winner, loser = c_b, c_a

        # Determine resolution strategy
        if winner.tier == ConstraintTier.PHYSICS:
            strategy = f"Physics constraint '{winner.id}' overrides '{loser.id}'"
        elif winner.tier == ConstraintTier.POLICY:
            strategy = f"Policy constraint '{winner.id}' overrides '{loser.id}'"
        else:
            strategy = f"Constraint '{winner.id}' takes precedence over '{loser.id}'"

        # Try to negotiate if loser is an Instruction
        negotiated_action = None
        if loser.tier == ConstraintTier.INSTRUCTION and user_goal:
            negotiated_action = self._negotiate_alternative(winner.id, user_goal)

        return ConflictResolution(
            conflicting_constraints=[constraint_a, constraint_b],
            winner=winner.id,
            loser=loser.id,
            resolution_strategy=strategy,
            negotiated_action=negotiated_action,
        )

    def _negotiate_alternative(self, blocker: str, goal: str) -> Optional[str]:
        """
        Negotiate an alternative action when a goal is blocked.

        "I cannot satisfy 'Download' because 'Physics' forbids it.
         I will generate a synthetic dataset instead."
        """
        strategies = self.NEGOTIATION_STRATEGIES.get(blocker, {})

        goal_lower = goal.lower()
        for keyword, alternative in strategies.items():
            if keyword in goal_lower:
                return alternative

        return f"Cannot perform '{goal}' due to {blocker}. Consider alternative approaches."


# =============================================================================
# SECTION 5: CSP CONSTRAINT MANAGER (Main Interface)
# =============================================================================

class CSPConstraintManager:
    """
    Main interface for CSP constraint handling in PRECEPT.

    Combines all capabilities:
    - Evo-Memory (CausalChainTracker) for dependency mapping
    - Context Engineering (RefineInterceptor) for constraint discovery
    - COMPASS (ConflictResolver) for hierarchy-based resolution

    Usage:
        manager = CSPConstraintManager()

        # Discover constraints from execution feedback
        feedback = ExecutionFeedback(return_code=1, stderr="Network is unreachable", ...)
        new_constraints = manager.intercept_feedback(feedback)

        # Get context for LLM (inject discovered constraints)
        context = manager.get_llm_context()

        # Resolve conflicts
        if manager.has_conflicts():
            resolutions = manager.resolve_all_conflicts(user_goal="download dataset")
    """

    def __init__(self, persistence_dir: Optional[Path] = None):
        self.persistence_dir = persistence_dir or Path("data/csp")
        # NOTE: Directory is created lazily when data is actually persisted
        # (in CausalChainTracker.save() and other persistence methods)
        # This avoids creating empty directories when CSP is not actively used.

        self.causal_tracker = CausalChainTracker(
            persistence_path=self.persistence_dir / "causal_chains.json"
        )
        self.refine_interceptor = RefineInterceptor(self.causal_tracker)
        self.constraints: Dict[str, Constraint] = {}
        self.conflict_resolver: Optional[ConflictResolver] = None

        # Statistics
        self.stats = {
            "constraints_discovered": 0,
            "conflicts_resolved": 0,
            "forward_checks_performed": 0,
            "chains_learned": 0,
        }

    def intercept_feedback(self, feedback: ExecutionFeedback) -> List[Constraint]:
        """
        Intercept execution feedback to discover constraints.

        This is the Refine Layer: "I found a new wall. Updating the map immediately."
        """
        discovered = self.refine_interceptor.intercept(feedback)

        for constraint in discovered:
            self.constraints[constraint.id] = constraint
            self.stats["constraints_discovered"] += 1

        # Perform forward checking
        for constraint in discovered:
            triggered = self.causal_tracker.get_triggered_constraints(constraint.id)
            if triggered:
                self.stats["forward_checks_performed"] += 1

        # Update conflict resolver with new constraints
        self.conflict_resolver = ConflictResolver(self.constraints)

        return discovered

    def get_llm_context(self) -> str:
        """
        Get constraint context for LLM injection.

        Context Engineering: Make invisible walls visible in the prompt.
        """
        context = self.refine_interceptor.get_constraint_context()

        if not context:
            return ""

        return f"""
═══════════════════════════════════════════════════════════════════════════════
⚠️ ACTIVE CONSTRAINTS (Discovered from execution - MUST RESPECT)
═══════════════════════════════════════════════════════════════════════════════
{context}

NOTE: These constraints are PHYSICS-level (immutable). Do NOT attempt actions
that violate them. Instead, use alternative approaches.
═══════════════════════════════════════════════════════════════════════════════
"""

    def has_conflicts(self) -> bool:
        """Check if there are any conflicting constraints."""
        if not self.conflict_resolver:
            return False
        return len(self.conflict_resolver.detect_conflicts()) > 0

    def resolve_all_conflicts(self, user_goal: Optional[str] = None) -> List[ConflictResolution]:
        """
        Resolve all detected conflicts using COMPASS hierarchy.

        COMPASS capability: "Physics overrides User Instructions. I will pivot the goal."
        """
        if not self.conflict_resolver:
            return []

        resolutions = []
        conflicts = self.conflict_resolver.detect_conflicts()

        for c_a, c_b in conflicts:
            resolution = self.conflict_resolver.resolve_conflict(c_a, c_b, user_goal)
            resolutions.append(resolution)
            self.stats["conflicts_resolved"] += 1

        return resolutions

    def get_forward_check_advice(self, constraint_id: str) -> str:
        """
        Get forward checking advice for a constraint.

        Evo-Memory capability: "If A is broken, I know B and C are also broken. Skip them."
        """
        triggered = self.causal_tracker.get_triggered_constraints(constraint_id)

        if not triggered:
            return ""

        return f"""
FORWARD CHECK: Because '{constraint_id}' is active, these are also blocked:
{chr(10).join(f'  • {t}' for t in triggered)}
Do NOT waste steps testing these - they will also fail.
"""

    def add_manual_constraint(
        self,
        id: str,
        name: str,
        tier: ConstraintTier,
        type: ConstraintType,
        description: str,
        dependencies: List[str] = None,
        conflicts_with: List[str] = None,
        solution_patterns: List[str] = None,
    ) -> Constraint:
        """Manually add a constraint (e.g., from scenario configuration)."""
        constraint = Constraint(
            id=id,
            name=name,
            tier=tier,
            type=type,
            description=description,
            dependencies=dependencies or [],
            conflicts_with=conflicts_with or [],
            solution_patterns=solution_patterns or [],
            discovered=True,
            discovered_at=time.time(),
        )
        self.constraints[id] = constraint
        self.refine_interceptor.active_constraints.add(id)
        self.conflict_resolver = ConflictResolver(self.constraints)
        return constraint

    def mark_satisfied(self, constraint_id: str) -> None:
        """Mark a constraint as satisfied."""
        if constraint_id in self.constraints:
            self.constraints[constraint_id].satisfied = True
            self.constraints[constraint_id].satisfied_at = time.time()
            self.refine_interceptor.clear_constraint(constraint_id)

    def get_unsatisfied_constraints(self) -> List[Constraint]:
        """Get all discovered but unsatisfied constraints."""
        return [c for c in self.constraints.values() if c.discovered and not c.satisfied]

    def get_stats(self) -> Dict[str, Any]:
        """Get CSP handling statistics."""
        return {
            **self.stats,
            "active_constraints": len(self.refine_interceptor.active_constraints),
            "total_constraints": len(self.constraints),
            "causal_chains": len(self.causal_tracker.chains),
        }

    def to_dict(self) -> Dict:
        """Serialize manager state for persistence."""
        return {
            "constraints": {
                c_id: {
                    "id": c.id,
                    "name": c.name,
                    "tier": c.tier.name,
                    "type": c.type.name,
                    "description": c.description,
                    "dependencies": c.dependencies,
                    "conflicts_with": c.conflicts_with,
                    "solution_patterns": c.solution_patterns,
                    "discovered": c.discovered,
                    "satisfied": c.satisfied,
                }
                for c_id, c in self.constraints.items()
            },
            "active": list(self.refine_interceptor.active_constraints),
            "stats": self.stats,
        }

    def save(self) -> None:
        """Persist state to disk."""
        self.causal_tracker.save()
        state_path = self.persistence_dir / "csp_state.json"
        with open(state_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# SECTION 6: CSP SCENARIO TEMPLATES (For Black Swan Generation)
# =============================================================================

# Pre-defined constraint clusters for common Black Swan scenarios
CONSTRAINT_CLUSTERS = {
    "network_failure": {
        "root": "NETWORK_DOWN",
        "tier": ConstraintTier.PHYSICS,
        "triggers": ["PIP_BLOCKED", "GIT_BLOCKED", "APT_BLOCKED", "CURL_BLOCKED"],
        "description": "Network is completely unavailable",
    },
    "filesystem_readonly": {
        "root": "READONLY_FS",
        "tier": ConstraintTier.PHYSICS,
        "triggers": ["WRITE_BLOCKED", "CREATE_BLOCKED", "MODIFY_BLOCKED"],
        "description": "Filesystem is mounted read-only",
    },
    "port_exhaustion": {
        "root": "PORT_BLOCKED",
        "tier": ConstraintTier.PHYSICS,
        "triggers": ["SERVER_BLOCKED", "LISTEN_BLOCKED"],
        "description": "All common ports are in use or blocked",
    },
    "permission_lockdown": {
        "root": "PERMISSION_DENIED",
        "tier": ConstraintTier.POLICY,
        "triggers": ["SUDO_BLOCKED", "ROOT_BLOCKED"],
        "description": "User lacks required permissions",
    },
    "dependency_hell": {
        "root": "DEPENDENCY_CONFLICT",
        "tier": ConstraintTier.PHYSICS,
        "triggers": ["VERSION_MISMATCH", "PACKAGE_INCOMPATIBLE"],
        "description": "Unresolvable dependency conflicts",
    },
}


def create_csp_scenario(
    cluster_name: str,
    additional_constraints: List[Constraint] = None,
    conflicts: List[Tuple[str, str]] = None,
) -> CSPConstraintManager:
    """
    Create a CSP scenario from a pre-defined cluster.

    Args:
        cluster_name: Name of the constraint cluster
        additional_constraints: Extra constraints to add
        conflicts: Pairs of conflicting constraint IDs

    Returns:
        Configured CSPConstraintManager
    """
    manager = CSPConstraintManager()

    if cluster_name in CONSTRAINT_CLUSTERS:
        cluster = CONSTRAINT_CLUSTERS[cluster_name]

        # Add root constraint
        root = manager.add_manual_constraint(
            id=cluster["root"],
            name=cluster["root"].replace("_", " ").title(),
            tier=cluster["tier"],
            type=ConstraintType.HARD,
            description=cluster["description"],
            dependencies=cluster["triggers"],
        )

        # Add triggered constraints
        for triggered_id in cluster["triggers"]:
            manager.add_manual_constraint(
                id=triggered_id,
                name=triggered_id.replace("_", " ").title(),
                tier=cluster["tier"],
                type=ConstraintType.INTERDEPENDENT,
                description=f"Triggered by {cluster['root']}",
            )

    # Add additional constraints
    if additional_constraints:
        for c in additional_constraints:
            manager.constraints[c.id] = c
            manager.refine_interceptor.active_constraints.add(c.id)

    # Add conflicts
    if conflicts:
        for c_a, c_b in conflicts:
            if c_a in manager.constraints:
                manager.constraints[c_a].conflicts_with.append(c_b)
            if c_b in manager.constraints:
                manager.constraints[c_b].conflicts_with.append(c_a)

    manager.conflict_resolver = ConflictResolver(manager.constraints)

    return manager
