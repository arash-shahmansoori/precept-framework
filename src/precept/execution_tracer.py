"""
Execution Tracer for PRECEPT.

Captures detailed step-by-step execution logs for training and testing phases.
This enables deep analysis of agent behavior, metrics, and learning dynamics.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """A single event in the execution trace."""

    timestamp: float
    event_type: str
    details: Dict[str, Any]
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


@dataclass
class TaskTrace:
    """Complete trace for a single task execution."""

    task_id: int
    task: str
    phase: str  # "training" or "testing"
    agent_type: str  # "precept", "full_reflexion", "llm_baseline", etc.
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None

    def add_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        duration_ms: Optional[float] = None,
    ):
        """Add an event to the trace."""
        self.events.append(
            TraceEvent(
                timestamp=time.time(),
                event_type=event_type,
                details=details,
                duration_ms=duration_ms,
            )
        )

    def complete(self, result: Dict[str, Any]):
        """Mark the task as complete with result."""
        self.end_time = time.time()
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task": self.task,
            "phase": self.phase,
            "agent_type": self.agent_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_sec": (
                self.end_time - self.start_time if self.end_time else None
            ),
            "events": [e.to_dict() for e in self.events],
            "result": self.result,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of the trace."""
        event_counts = {}
        for e in self.events:
            event_counts[e.event_type] = event_counts.get(e.event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_counts": event_counts,
            "success": self.result.get("success", False) if self.result else None,
            "total_steps": self.result.get("task_steps", 0) if self.result else 0,
            "strategy_used": self.result.get("strategy", "") if self.result else "",
        }


@dataclass
class ExecutionLog:
    """Complete execution log for an experiment."""

    experiment_id: str
    domain: str
    started_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    training_traces: List[TaskTrace] = field(default_factory=list)
    testing_traces: Dict[str, List[TaskTrace]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_training_trace(self, trace: TaskTrace):
        """Add a training phase trace."""
        self.training_traces.append(trace)

    def add_testing_trace(self, agent_type: str, trace: TaskTrace):
        """Add a testing phase trace for a specific agent."""
        if agent_type not in self.testing_traces:
            self.testing_traces[agent_type] = []
        self.testing_traces[agent_type].append(trace)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "started_at": self.started_at,
            "metadata": self.metadata,
            "training": {
                "total_tasks": len(self.training_traces),
                "traces": [t.to_dict() for t in self.training_traces],
                "summary": self._training_summary(),
            },
            "testing": {
                agent: {
                    "total_tasks": len(traces),
                    "traces": [t.to_dict() for t in traces],
                    "summary": self._testing_summary(traces),
                }
                for agent, traces in self.testing_traces.items()
            },
        }

    def _training_summary(self) -> Dict[str, Any]:
        """Generate training phase summary."""
        if not self.training_traces:
            return {}

        # Group by agent
        agent_results = {}
        for trace in self.training_traces:
            agent = trace.agent_type
            if agent not in agent_results:
                agent_results[agent] = {"successes": 0, "total": 0, "steps": []}
            agent_results[agent]["total"] += 1
            if trace.result and trace.result.get("success"):
                agent_results[agent]["successes"] += 1
            if trace.result:
                agent_results[agent]["steps"].append(
                    trace.result.get("task_steps", 0)
                )

        return {
            agent: {
                "success_rate": (
                    data["successes"] / data["total"] if data["total"] > 0 else 0
                ),
                "avg_steps": (
                    sum(data["steps"]) / len(data["steps"])
                    if data["steps"]
                    else 0
                ),
            }
            for agent, data in agent_results.items()
        }

    def _testing_summary(self, traces: List[TaskTrace]) -> Dict[str, Any]:
        """Generate testing phase summary for a set of traces."""
        if not traces:
            return {}

        successes = sum(
            1 for t in traces if t.result and t.result.get("success")
        )
        steps = [t.result.get("task_steps", 0) for t in traces if t.result]

        return {
            "success_rate": successes / len(traces) if traces else 0,
            "avg_steps": sum(steps) / len(steps) if steps else 0,
            "first_try_success": sum(
                1
                for t in traces
                if t.result
                and t.result.get("success")
                and t.result.get("task_steps", 0) <= 2
            ),
        }


class ExecutionTracer:
    """
    Manages execution tracing for experiments.

    Usage:
        tracer = ExecutionTracer(domain="booking", experiment_id="exp_001")

        # During training
        trace = tracer.start_task(1, task, "training", "precept")
        trace.add_event("parse_task", {"action": "book_flight", "entity": "AA-999"})
        trace.add_event("fetch_context", {"rules_found": 2})
        trace.add_event("llm_reasoning", {"suggested": "UA-200"})
        trace.add_event("execute_action", {"flight": "UA-200", "success": True})
        trace.complete(result)
        tracer.end_task(trace)

        # Save logs
        tracer.save_to_file("logs/experiment.json")
    """

    def __init__(
        self,
        domain: str,
        experiment_id: Optional[str] = None,
        enabled: bool = True,
    ):
        """
        Initialize the execution tracer.

        Args:
            domain: The domain being tested
            experiment_id: Unique ID for this experiment (auto-generated if None)
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled
        self.domain = domain
        self.experiment_id = experiment_id or f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log = ExecutionLog(
            experiment_id=self.experiment_id,
            domain=domain,
        )
        self._current_traces: Dict[str, TaskTrace] = {}

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set experiment metadata (e.g., config, seeds)."""
        self.log.metadata = metadata

    def start_task(
        self,
        task_id: int,
        task: str,
        phase: str,
        agent_type: str,
    ) -> TaskTrace:
        """
        Start tracing a new task.

        Args:
            task_id: Sequential task number
            task: The task description
            phase: "training" or "testing"
            agent_type: "precept", "full_reflexion", etc.

        Returns:
            TaskTrace object to add events to
        """
        if not self.enabled:
            return TaskTrace(task_id, task, phase, agent_type)

        trace = TaskTrace(
            task_id=task_id,
            task=task,
            phase=phase,
            agent_type=agent_type,
        )
        key = f"{phase}_{agent_type}_{task_id}"
        self._current_traces[key] = trace
        return trace

    def end_task(self, trace: TaskTrace, result: Optional[Dict[str, Any]] = None):
        """
        End tracing for a task.

        Args:
            trace: The TaskTrace to complete
            result: Optional result dict (can also be set via trace.complete())
        """
        if not self.enabled:
            return

        if result:
            trace.complete(result)

        if trace.phase == "training":
            self.log.add_training_trace(trace)
        else:
            self.log.add_testing_trace(trace.agent_type, trace)

    def save_to_file(self, filepath: str):
        """
        Save the execution log to a JSON file.

        Args:
            filepath: Path to save the log
        """
        import json
        import os

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.log.to_dict(), f, indent=2, default=str)

    def get_log(self) -> ExecutionLog:
        """Get the execution log."""
        return self.log

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the execution."""
        return {
            "experiment_id": self.experiment_id,
            "domain": self.domain,
            "training_tasks": len(self.log.training_traces),
            "testing_agents": list(self.log.testing_traces.keys()),
            "testing_tasks_per_agent": {
                agent: len(traces)
                for agent, traces in self.log.testing_traces.items()
            },
        }


# Helper functions for common trace events
def trace_parse_task(
    trace: TaskTrace,
    action: str,
    entity: str,
    parameters: Dict[str, Any],
):
    """Add a parse_task event to the trace."""
    trace.add_event(
        "parse_task",
        {
            "action": action,
            "entity": entity,
            "parameters": parameters,
        },
    )


def trace_fetch_context(
    trace: TaskTrace,
    rules_found: int,
    memories_found: int,
    procedure_hint: Optional[str] = None,
):
    """Add a fetch_context event to the trace."""
    trace.add_event(
        "fetch_context",
        {
            "rules_found": rules_found,
            "memories_found": memories_found,
            "procedure_hint": procedure_hint,
        },
    )


def trace_compass_decision(
    trace: TaskTrace,
    action: str,
    reason: str,
    blocking_constraint: Optional[str] = None,
    negotiated_alternative: Optional[str] = None,
):
    """Add a COMPASS decision event to the trace."""
    trace.add_event(
        "compass_decision",
        {
            "action": action,
            "reason": reason,
            "blocking_constraint": blocking_constraint,
            "negotiated_alternative": negotiated_alternative,
        },
    )


def trace_llm_reasoning(
    trace: TaskTrace,
    suggested_solution: Optional[str],
    strategy_used: str,
    was_rule_applied: bool,
):
    """Add an LLM reasoning event to the trace."""
    trace.add_event(
        "llm_reasoning",
        {
            "suggested_solution": suggested_solution,
            "strategy_used": strategy_used,
            "was_rule_applied": was_rule_applied,
        },
    )


def trace_execute_action(
    trace: TaskTrace,
    action: str,
    entity: str,
    success: bool,
    error_code: Optional[str] = None,
    response: Optional[str] = None,
):
    """Add an execute_action event to the trace."""
    trace.add_event(
        "execute_action",
        {
            "action": action,
            "entity": entity,
            "success": success,
            "error_code": error_code,
            "response": response[:200] if response else None,
        },
    )


def trace_error_recovery(
    trace: TaskTrace,
    error_code: str,
    pivot_number: int,
    tried_solution: str,
    success: bool,
):
    """Add an error recovery/pivot event to the trace."""
    trace.add_event(
        "error_recovery",
        {
            "error_code": error_code,
            "pivot_number": pivot_number,
            "tried_solution": tried_solution,
            "success": success,
        },
    )


def trace_probe_execution(
    trace: TaskTrace,
    probe_id: str,
    constraint_discovered: Optional[str] = None,
    alternative_suggested: Optional[str] = None,
):
    """Add a probe execution event to the trace."""
    trace.add_event(
        "probe_execution",
        {
            "probe_id": probe_id,
            "constraint_discovered": constraint_discovered,
            "alternative_suggested": alternative_suggested,
        },
    )


def trace_rule_applied(
    trace: TaskTrace,
    rule: str,
    original_solution: str,
    new_solution: str,
):
    """Add a rule application event to the trace."""
    trace.add_event(
        "rule_applied",
        {
            "rule": rule,
            "original_solution": original_solution,
            "new_solution": new_solution,
        },
    )


def trace_learning(
    trace: TaskTrace,
    learning_type: str,
    details: Dict[str, Any],
):
    """Add a learning event to the trace."""
    trace.add_event(
        "learning",
        {
            "learning_type": learning_type,
            **details,
        },
    )

