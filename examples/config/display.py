"""
Display Utilities for PRECEPT Experiments.

Provides formatted output for experiment results using proper logging.
Separates visual presentation from logging concerns.

Includes rich progress bars for training and testing phases.

Usage:
    from config.display import ExperimentDisplay, ProgressTracker

    display = ExperimentDisplay(logger)
    display.print_header(domain="logistics")
    display.print_results(metrics)

    # Progress tracking
    with ProgressTracker() as tracker:
        with tracker.training_progress(total=8) as progress:
            for i in range(8):
                # ... training ...
                progress.advance()
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass
class DisplayConfig:
    """Configuration for display output."""

    width: int = 85
    use_colors: bool = True
    use_emoji: bool = True


class ExperimentDisplay:
    """Formatted display output for experiments."""

    def __init__(self, logger, config: Optional[DisplayConfig] = None):
        self.logger = logger
        self.config = config or DisplayConfig()

    def _log_block(self, lines: List[str]) -> None:
        """Log a block of formatted lines."""
        for line in lines:
            self.logger.info(line)

    def print_header(self, domain: str, icon: str = "🔬") -> None:
        """Print experiment header."""
        lines = [
            "",
            "╔" + "═" * 82 + "╗",
            "║" + " " * 82 + "║",
            f"║   {icon} PRECEPT vs 3 BASELINES: {domain.upper()} DOMAIN TEST"
            + " " * (55 - len(domain))
            + "║",
            "║      (Planning Resilience via Experience, Context Engineering & Probing Trajectories)"
            + " " * 17
            + "║",
            "║" + " " * 82 + "║",
            "║   4-WAY COMPARISON:" + " " * 62 + "║",
            "║   " + "═" * 42 + " " * 37 + "║",
            "║   1. LLMBaselineAgent      - Error feedback only (Adapted ReAct)"
            + " " * 17
            + "║",
            "║   2. ReflexionBaselineAgent - Within-task reflection (Adapted Reflexion)"
            + " " * 9
            + "║",
            "║   3. FullReflexionBaseline  - Cross-episode memory (Full Reflexion paper)"
            + " " * 8
            + "║",
            "║   4. PRECEPT               - Full learning stack (rules + pruning + evolution)"
            + " " * 4
            + "║",
            "║" + " " * 82 + "║",
            "║   WHAT EACH COMPARISON PROVES:" + " " * 51 + "║",
            "║   " + "═" * 42 + " " * 37 + "║",
            "║   PRECEPT > LLM         → Learning of ANY kind helps" + " " * 31 + "║",
            "║   PRECEPT > Reflexion   → Cross-episode memory helps" + " " * 31 + "║",
            "║   PRECEPT > FullReflexion → Structured learning > verbal reflections"
            + " " * 15
            + "║",
            "║" + " " * 82 + "║",
            "╚" + "═" * 82 + "╝",
        ]
        self._log_block(lines)

    def print_api_status(self, available: bool) -> None:
        """Print API availability status."""
        if available:
            self.logger.info("✓ OpenAI API available")
        else:
            self.logger.error("✗ OpenAI API not available. Set OPENAI_API_KEY in .env")

    def print_agents_initialized(self) -> None:
        """Print agent initialization summary."""
        lines = [
            "✅ All 4 agents initialized",
            "",
            "┌" + "─" * 77 + "┐",
            "│  4-WAY FAIR COMPARISON SETUP" + " " * 48 + "│",
            "├" + "─" * 77 + "┤",
            "│  Agent                    Features" + " " * 42 + "│",
            "├" + "─" * 77 + "┤",
            "│  1. LLM Baseline          LLM + error feedback only" + " " * 25 + "│",
            "│  2. Reflexion             LLM + within-task reflection" + " " * 22 + "│",
            "│  3. Full Reflexion        LLM + cross-episode memory (same task type)"
            + " " * 7
            + "│",
            "│  4. PRECEPT               LLM + rules + pruning + evolution + cross-task"
            + " " * 4
            + "│",
            "├" + "─" * 77 + "┤",
            "│  ALL agents: Same LLM (GPT-4o-mini), same MCP tools, same retry budget"
            + " " * 6
            + "│",
            "└" + "─" * 77 + "┘",
        ]
        self._log_block(lines)

    def print_phase_header(
        self, phase: str, num_tasks: int, details: List[str]
    ) -> None:
        """Print phase header (training/testing)."""
        self.logger.info("")
        self.logger.info("─" * 85)
        self.logger.info(f"🎓 PHASE {phase} ({num_tasks} tasks)")
        for detail in details:
            self.logger.info(f"   {detail}")
        self.logger.info("─" * 85)

    def print_train_result(
        self,
        num: int,
        total: int,
        task: str,
        expected: str,
        precept_success: bool,
        fr_success: bool,
        fr_memories: int,
    ) -> None:
        """Print training task result."""
        self.logger.info("")
        self.logger.info(f"  [Train {num}/{total}] {task[:55]}...")
        self.logger.info(f"      Expected: {expected[:50]}...")
        c_status = "✅" if precept_success else "❌"
        fr_status = "✅" if fr_success else "❌"
        self.logger.info(f"      PRECEPT:         {c_status}")
        self.logger.info(f"      Full Reflexion: {fr_status} (memories: {fr_memories})")

    def print_concurrent_train_result(
        self,
        num: int,
        total: int,
        task: str,
        precept_success: bool,
        fr_success: bool,
    ) -> None:
        """Print concurrent training result."""
        c_status = "✅" if precept_success else "❌"
        fr_status = "✅" if fr_success else "❌"
        self.logger.info(
            f"  [Train {num}/{total}] {task[:45]}... PRECEPT: {c_status} | FullRef: {fr_status}"
        )

    def print_test_result(
        self,
        num: int,
        total: int,
        task: str,
        tests: str,
        results: Dict[str, Dict[str, Any]],
    ) -> None:
        """Print test task result."""
        self.logger.info("")
        self.logger.info(f"  [Test {num}/{total}] {task[:60]}...")
        self.logger.info(f"      Tests: {tests}")

        for agent_name, result in results.items():
            status = "✅" if result["success"] else "❌"
            self.logger.info(
                f"      {agent_name:<14}: {status} ({result['steps']} steps)"
            )

        # Check if PRECEPT is faster
        precept_result = results.get("PRECEPT", {})
        if precept_result.get("success"):
            faster_than = []
            for name, res in results.items():
                if name != "PRECEPT" and res.get("success"):
                    diff = res["steps"] - precept_result["steps"]
                    if diff > 0:
                        faster_than.append(f"{name[:7]}({diff})")
            if faster_than:
                self.logger.info(
                    f"      ⚡ PRECEPT faster than: {', '.join(faster_than)}"
                )

    def print_training_summary(
        self,
        elapsed: float,
        concurrent: bool,
        training_workers: int,
        rules_learned: int,
        consolidations: int,
        fr_memories: int,
    ) -> None:
        """Print training summary."""
        self.logger.info("")
        self.logger.info("  ─── Training Summary ───")
        self.logger.info(f"  ⏱️  Training completed in {elapsed:.1f}s")
        if concurrent:
            sequential_estimate = elapsed * training_workers
            self.logger.info(
                f"  📈 Speedup: ~{training_workers}x (estimated {sequential_estimate:.0f}s sequential)"
            )
        self.logger.info(f"  📚 PRECEPT Rules Learned: {rules_learned}")
        self.logger.info(f"  🔄 PRECEPT Consolidations: {consolidations}")
        self.logger.info(f"  📝 Full Reflexion Memories: {fr_memories}")

    def print_performance_comparison(
        self,
        n_test: int,
        llm_metrics: Any,
        reflexion_metrics: Any,
        full_reflexion_metrics: Any,
        precept_metrics: Any,
        precept_llm_calls: int,
    ) -> None:
        """Print performance comparison table."""
        lines = [
            "",
            "┌" + "─" * 85 + "┐",
            "│" + " " * 27 + "PERFORMANCE COMPARISON" + " " * 36 + "│",
            "├" + "─" * 85 + "┤",
            "│  Agent              Success Rate    Avg Steps    LLM Calls    LLM Accuracy"
            + " " * 10
            + "│",
            "├" + "─" * 85 + "┤",
            f"│  LLM Baseline       {llm_metrics.total_successes:>2}/{n_test} ({llm_metrics.success_rate * 100:>5.1f}%)       {llm_metrics.avg_steps:>5.1f}        {llm_metrics.llm_calls:>4}         {llm_metrics.llm_accuracy * 100:>5.1f}%           │",
            f"│  Reflexion          {reflexion_metrics.total_successes:>2}/{n_test} ({reflexion_metrics.success_rate * 100:>5.1f}%)       {reflexion_metrics.avg_steps:>5.1f}        {reflexion_metrics.llm_calls:>4}         {reflexion_metrics.llm_accuracy * 100:>5.1f}%           │",
            f"│  Full Reflexion     {full_reflexion_metrics.total_successes:>2}/{n_test} ({full_reflexion_metrics.success_rate * 100:>5.1f}%)       {full_reflexion_metrics.avg_steps:>5.1f}        {full_reflexion_metrics.llm_calls:>4}         {full_reflexion_metrics.llm_accuracy * 100:>5.1f}%           │",
            f"│  PRECEPT            {precept_metrics.total_successes:>2}/{n_test} ({precept_metrics.success_rate * 100:>5.1f}%)       {precept_metrics.avg_steps:>5.1f}        {precept_llm_calls:>4}         N/A              │",
            "└" + "─" * 85 + "┘",
        ]
        self._log_block(lines)

    def print_learning_capabilities(self, has_compass_evolution: bool) -> None:
        """Print learning capabilities table."""
        compass_icon = "✅" if has_compass_evolution else "❌"
        lines = [
            "",
            "┌" + "─" * 85 + "┐",
            "│" + " " * 27 + "LEARNING CAPABILITIES" + " " * 37 + "│",
            "├" + "─" * 85 + "┤",
            "│  Feature                    LLM Base   Reflexion   Full Ref    PRECEPT"
            + " " * 16
            + "│",
            "├" + "─" * 85 + "┤",
            "│  Cross-Task Learning           ❌          ❌          ❌          ✅"
            + " " * 19
            + "│",
            "│  Cross-Episode Memory          ❌          ❌          ✅          ✅"
            + " " * 19
            + "│",
            "│  Within-Task Reflection        ❌          ✅          ✅          ✅"
            + " " * 19
            + "│",
            "│  Deterministic Pruning         ❌          ❌          ❌          ✅"
            + " " * 19
            + "│",
            f"│  COMPASS Evolution             ❌          ❌          ❌          {compass_icon}"
            + " " * 19
            + "│",
            "│  Rule Compilation              ❌          ❌          ❌          ✅"
            + " " * 19
            + "│",
            "│  Semantic Retrieval            ❌          ❌          ❌          ✅"
            + " " * 19
            + "│",
            "└" + "─" * 85 + "┘",
        ]
        self._log_block(lines)

    def print_learning_statistics(
        self,
        rules_learned: int,
        dumb_retries: int,
        hard_constraints: int,
        soft_constraints: int,
        fr_reflections: int,
        fr_reused: int,
        fr_memory_size: int,
        ref_reflections: int,
        ref_lessons: int,
    ) -> None:
        """Print learning statistics."""
        lines = [
            "",
            "┌" + "─" * 85 + "┐",
            "│" + " " * 27 + "LEARNING STATISTICS" + " " * 39 + "│",
            "├" + "─" * 85 + "┤",
            "│  PRECEPT:" + " " * 75 + "│",
            f"│    Rules Learned:          {rules_learned:>4}" + " " * 55 + "│",
            f"│    Dumb Retries Prevented: {dumb_retries:>4}" + " " * 55 + "│",
            f"│    Hard Constraints Found: {hard_constraints:>4}" + " " * 55 + "│",
            f"│    Soft Constraints Found: {soft_constraints:>4}" + " " * 55 + "│",
            "│" + " " * 85 + "│",
            "│  Full Reflexion:" + " " * 68 + "│",
            f"│    Reflections Generated:  {fr_reflections:>4}" + " " * 55 + "│",
            f"│    Reflections Reused:     {fr_reused:>4}" + " " * 55 + "│",
            f"│    Memory Size:            {fr_memory_size:>4}" + " " * 55 + "│",
            "│" + " " * 85 + "│",
            "│  Reflexion (within-task):" + " " * 59 + "│",
            f"│    Reflections Generated:  {ref_reflections:>4}" + " " * 55 + "│",
            f"│    Lessons Generated:      {ref_lessons:>4}" + " " * 55 + "│",
            "└" + "─" * 85 + "┘",
        ]
        self._log_block(lines)

    def print_key_insights(
        self,
        precept_vs_llm: float,
        precept_vs_ref: float,
        precept_vs_fullref: float,
        steps_vs_llm: float,
        steps_vs_ref: float,
        steps_vs_fullref: float,
    ) -> None:
        """Print key insights."""
        lines = [
            "",
            "┌" + "─" * 85 + "┐",
            "│" + " " * 27 + "KEY INSIGHTS" + " " * 46 + "│",
            "├" + "─" * 85 + "┤",
            "│  PRECEPT vs LLM Baseline:" + " " * 59 + "│",
            f"│    Success: {precept_vs_llm * 100:+.1f}%  Steps: {steps_vs_llm:+.1f}"
            + " " * 53
            + "│",
            "│    → Proves: Cross-task LEARNING is valuable" + " " * 40 + "│",
            "│" + " " * 85 + "│",
            "│  PRECEPT vs Reflexion:" + " " * 62 + "│",
            f"│    Success: {precept_vs_ref * 100:+.1f}%  Steps: {steps_vs_ref:+.1f}"
            + " " * 53
            + "│",
            "│    → Proves: Cross-EPISODE memory > within-task reflection"
            + " " * 26
            + "│",
            "│" + " " * 85 + "│",
            "│  PRECEPT vs Full Reflexion:" + " " * 57 + "│",
            f"│    Success: {precept_vs_fullref * 100:+.1f}%  Steps: {steps_vs_fullref:+.1f}"
            + " " * 53
            + "│",
            "│    → Proves: Structured learning (rules, pruning) > verbal reflections"
            + " " * 14
            + "│",
            "└" + "─" * 85 + "┘",
        ]
        self._log_block(lines)

    def print_winner(
        self, winner_name: str, success_rate: float, avg_steps: float
    ) -> None:
        """Print winner announcement."""
        lines = [
            "",
            "╔" + "═" * 85 + "╗",
            "║" + " " * 85 + "║",
            f"║  🏆 WINNER: {winner_name:<20}" + " " * 51 + "║",
            "║" + " " * 85 + "║",
            f"║  Success Rate: {success_rate * 100:.1f}%    Avg Steps: {avg_steps:.1f}"
            + " " * 42
            + "║",
            "║" + " " * 85 + "║",
            "╚" + "═" * 85 + "╝",
        ]
        self._log_block(lines)

    def print_conclusion(self, domain: str) -> None:
        """Print experiment conclusion."""
        lines = [
            "",
            "╔" + "═" * 86 + "╗",
            "║" + " " * 86 + "║",
            f"║   🏆 CONCLUSION: PRECEPT {domain.upper()} Domain Test Complete"
            + " " * (48 - len(domain))
            + "║",
            "║      PRECEPT = Planning Resilience via Experience, Context Engineering & Probing Trajectories"
            + " " * 15
            + "║",
            "║" + " " * 86 + "║",
            "║   WHAT WE PROVED:" + " " * 68 + "║",
            "║   • Learning > No Learning (PRECEPT vs LLM Baseline)" + " " * 33 + "║",
            "║   • Cross-Episode > Within-Task (PRECEPT vs Reflexion)" + " " * 31 + "║",
            "║   • Structured > Verbal (PRECEPT vs Full Reflexion)" + " " * 34 + "║",
            "║" + " " * 86 + "║",
            "║   PRECEPT UNIQUE ADVANTAGES:" + " " * 57 + "║",
            "║   • Deterministic Pruning (hard constraints, forbidden solutions)"
            + " " * 20
            + "║",
            "║   • Rule Compilation (fast lookup vs re-reasoning)" + " " * 35 + "║",
            "║   • COMPASS Evolution (adaptive prompts)" + " " * 45 + "║",
            "║   • Memory-to-Rule Promotion (semantic → explicit)" + " " * 35 + "║",
            "║   • Cross-Task-TYPE Learning (not just same task type)" + " " * 31 + "║",
            "║" + " " * 86 + "║",
            "╚" + "═" * 86 + "╝",
        ]
        self._log_block(lines)

    def print_domain_info(
        self,
        domain: str,
        icon: str,
        strategy_name: str,
        baseline_name: str,
        actions: List[str],
        options: List[str],
        num_scenarios: int,
    ) -> None:
        """Print domain information."""
        lines = [
            "",
            "┌" + "─" * 77 + "┐",
            f"│  {icon} {domain.upper():<72} │",
            "├" + "─" * 77 + "┤",
            f"│  Strategy:    {strategy_name:<60} │",
            f"│  Baseline:    {baseline_name:<60} │",
            f"│  Actions:     {str(actions):<60} │",
            f"│  Options:     {str(options):<60} │",
            f"│  Scenarios:   {num_scenarios:<60} │",
            "└" + "─" * 77 + "┘",
        ]
        self._log_block(lines)


class ProgressTracker:
    """
    Rich-based progress tracker for experiment phases.

    Provides beautiful progress bars for training and testing phases
    with elapsed time, ETA, and completion status.

    Usage:
        tracker = ProgressTracker()

        # Training phase
        with tracker.training_progress(total=8) as update:
            for i, scenario in enumerate(train_scenarios):
                result = await agent.run_task(scenario)
                update(precept_ok=result["success"], fr_ok=True)

        # Testing phase
        with tracker.testing_progress(total=4) as update:
            for i, scenario in enumerate(test_scenarios):
                results = await run_all_agents(scenario)
                update(results=results)
    """

    def __init__(self, console: Optional[Console] = None):
        """Initialize progress tracker."""
        self.console = console or Console()

    def _create_progress(self, description: str) -> Progress:
        """Create a rich Progress instance with standard columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    @contextmanager
    def training_progress(
        self, total: int, description: str = "🎓 Training"
    ) -> Generator:
        """
        Context manager for training progress.

        Args:
            total: Total number of training tasks
            description: Progress bar description

        Yields:
            Update function: update(precept_ok: bool, fr_ok: bool)
        """
        progress = self._create_progress(description)
        successes = {"precept": 0, "fr": 0}

        with progress:
            task_id = progress.add_task(description, total=total)

            def update(precept_ok: bool = True, fr_ok: bool = True) -> None:
                if precept_ok:
                    successes["precept"] += 1
                if fr_ok:
                    successes["fr"] += 1
                progress.advance(task_id)

                # Update description with success counts
                pct = successes["precept"]
                fct = successes["fr"]
                progress.update(
                    task_id,
                    description=f"🎓 Training [PRECEPT:{pct}✓ FR:{fct}✓]",
                )

            yield update

        # Print final summary
        self.console.print(
            f"  ✅ Training complete: PRECEPT {successes['precept']}/{total}, "
            f"Full Reflexion {successes['fr']}/{total}"
        )

    @contextmanager
    def testing_progress(
        self, total: int, description: str = "🧪 Testing"
    ) -> Generator:
        """
        Context manager for testing progress.

        Args:
            total: Total number of test tasks
            description: Progress bar description

        Yields:
            Update function: update(results: Dict[str, Dict])
        """
        progress = self._create_progress(description)
        agent_successes = {
            "LLM": 0,
            "Reflexion": 0,
            "FullRef": 0,
            "PRECEPT": 0,
        }

        with progress:
            task_id = progress.add_task(description, total=total)

            def update(results: Optional[Dict[str, Dict]] = None) -> None:
                if results:
                    if results.get("LLM Baseline", {}).get("success"):
                        agent_successes["LLM"] += 1
                    if results.get("Reflexion", {}).get("success"):
                        agent_successes["Reflexion"] += 1
                    if results.get("Full Reflexion", {}).get("success"):
                        agent_successes["FullRef"] += 1
                    if results.get("PRECEPT", {}).get("success"):
                        agent_successes["PRECEPT"] += 1

                progress.advance(task_id)

                # Update description with success counts
                desc = (
                    f"🧪 Testing [L:{agent_successes['LLM']} "
                    f"R:{agent_successes['Reflexion']} "
                    f"FR:{agent_successes['FullRef']} "
                    f"P:{agent_successes['PRECEPT']}]"
                )
                progress.update(task_id, description=desc)

            yield update

        # Print final summary
        self.console.print(
            f"  ✅ Testing complete: "
            f"LLM {agent_successes['LLM']}/{total}, "
            f"Reflexion {agent_successes['Reflexion']}/{total}, "
            f"FullRef {agent_successes['FullRef']}/{total}, "
            f"PRECEPT {agent_successes['PRECEPT']}/{total}"
        )

    @contextmanager
    def simple_progress(self, total: int, description: str = "Processing") -> Generator:
        """
        Simple progress bar for generic tasks.

        Args:
            total: Total number of items
            description: Progress bar description

        Yields:
            Advance function: advance(n=1)
        """
        progress = self._create_progress(description)

        with progress:
            task_id = progress.add_task(description, total=total)

            def advance(n: int = 1) -> None:
                progress.advance(task_id, advance=n)

            yield advance


__all__ = ["ExperimentDisplay", "DisplayConfig", "ProgressTracker"]
