"""
Single-Turn Docker Agent for PRECEPT.

A single-turn agent that uses real Docker execution and learns from execution errors.
This is the single-turn equivalent of MultiTurnDockerAgent.

This agent supports:
1. Real Docker-based code execution (via CodeExecutionManager)
2. Dynamic learning from execution errors (DynamicCodingConfig)
3. Error feedback processing and categorization
4. Recovery suggestion learning
5. Pattern persistence and loading
6. Experience storage with optional vector store

Data Directory Structure:
    data/
    └── precept_singleturn/
        ├── coding_config.json      # Learned patterns & recoveries
        ├── execution_history.json  # Execution records
        ├── experiences.json        # Episodic memory (JSON)
        └── chroma_db/              # Vector store (if enabled)

Usage:
    from precept import SingleTurnDockerAgent
    
    # Create agent with default data directory
    agent = SingleTurnDockerAgent(enable_docker=True)
    
    # Execute code and learn from result
    result, feedback = await agent.execute_and_learn(
        code="import missing_package",
        scenario_name="test_import",
        category="IMPORT-ERROR"
    )
    
    # Save learned patterns
    agent.save_learning()
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .code_executor import CodeExecutionManager, ExecutionResult
from .execution_feedback_processor import ExecutionFeedbackProcessor, ProcessedFeedback
from .dynamic_coding_config import DynamicCodingConfig

# ChromaDB for vector persistence (optional)
try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Chroma = None
    OpenAIEmbeddings = None


# Default data directory (relative to workspace)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "precept_singleturn"


class SingleTurnDockerAgent:
    """
    Single-turn agent with real Docker execution and dynamic learning.
    
    Combines:
    1. Docker code execution (CodeExecutionManager)
    2. Error feedback processing (ExecutionFeedbackProcessor)
    3. Dynamic learning (DynamicCodingConfig)
    4. Experience storage (JSON + optional ChromaDB)
    
    Features:
    - Execute Python code in Docker containers
    - Learn error patterns dynamically
    - Learn recovery suggestions
    - Persist and load learned patterns
    - Track execution history
    - Export learned rules for COMPASS prompt evolution
    
    Usage:
        agent = SingleTurnDockerAgent(enable_docker=True)
        
        result, feedback = await agent.execute_and_learn(
            code="print('hello')",
            scenario_name="test",
            category="SUCCESS"
        )
        
        agent.save_learning()
    """
    
    def __init__(
        self,
        enable_docker: bool = True,
        data_dir: Optional[Path] = None,
        timeout: int = 60,
        auto_save: bool = True,
        enable_vector_store: bool = True,
        enable_llm_categorization: bool = False,
    ):
        """
        Initialize the single-turn Docker agent.
        
        Args:
            enable_docker: Whether to enable Docker execution (fallback to subprocess if False)
            data_dir: Directory for persisting data (defaults to data/precept_singleturn/)
            timeout: Default timeout for code execution in seconds
            auto_save: Whether to auto-save after each execution
            enable_vector_store: Whether to enable ChromaDB vector store for semantic search
            enable_llm_categorization: Whether to use LLM for error categorization
        """
        self.executor = CodeExecutionManager(enable_docker=enable_docker)
        self.processor = ExecutionFeedbackProcessor(enable_llm_categorization=enable_llm_categorization)
        self.timeout = timeout
        self.auto_save = auto_save
        
        # Setup data directory structure
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._setup_data_directory()
        
        # Initialize DynamicCodingConfig with proper path
        self.config = DynamicCodingConfig()
        self.config._config_path = self.data_dir / "coding_config.json"
        
        # Load existing config if available
        if self.config._config_path.exists():
            self.config.load_from_json()
        
        # Initialize ChromaDB vector store for semantic experience retrieval
        self.vector_store = None
        self.embeddings = None
        self._vector_store_enabled = enable_vector_store
        if enable_vector_store:
            self._init_vector_store()
        
        # Load episodic experiences (JSON fallback)
        self.experiences: List[Dict] = []
        self._load_experiences()
        
        # Learning state
        self.learned_patterns: Dict[str, str] = {}  # pattern -> category
        self.learned_recoveries: Dict[str, str] = {}  # category -> recovery
        self.execution_history: List[Dict] = []
        
        # Load execution stats
        self._load_execution_stats()
    
    def _setup_data_directory(self) -> None:
        """Setup the data directory structure."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chroma_db").mkdir(exist_ok=True)
    
    def _init_vector_store(self) -> bool:
        """Initialize ChromaDB vector store with OpenAI embeddings."""
        import os
        
        if not CHROMA_AVAILABLE:
            return False
        
        # Check if OpenAI API key is available (don't block if not)
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                timeout=10,
            )
            self.vector_store = Chroma(
                collection_name="precept_singleturn_experiences",
                embedding_function=self.embeddings,
                persist_directory=str(self.data_dir / "chroma_db"),
            )
            return True
        except Exception:
            self.vector_store = None
            self.embeddings = None
            return False
    
    def _load_experiences(self) -> None:
        """Load experiences from JSON."""
        experiences_path = self.data_dir / "experiences.json"
        if experiences_path.exists():
            try:
                with open(experiences_path, 'r') as f:
                    self.experiences = json.load(f)
            except Exception:
                self.experiences = []
        else:
            self.experiences = []
    
    def _save_experiences(self) -> None:
        """Save experiences to JSON."""
        experiences_path = self.data_dir / "experiences.json"
        try:
            with open(experiences_path, 'w') as f:
                json.dump(self.experiences, f, indent=2, default=str)
        except Exception:
            pass
    
    def _load_execution_stats(self) -> None:
        """Load execution stats from disk."""
        stats_path = self.data_dir / "execution_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    self._persisted_stats = json.load(f)
            except Exception:
                self._persisted_stats = {"total_executions": 0, "total_errors": 0}
        else:
            self._persisted_stats = {"total_executions": 0, "total_errors": 0}
    
    def _save_execution_stats(self) -> None:
        """Save execution stats to disk."""
        stats_path = self.data_dir / "execution_stats.json"
        try:
            with open(stats_path, 'w') as f:
                json.dump(self._persisted_stats, f, indent=2)
        except Exception:
            pass
    
    @property
    def is_docker_available(self) -> bool:
        """Check if Docker is available for code execution."""
        return self.executor.is_docker_available()
    
    async def execute_and_learn(
        self,
        code: str,
        scenario_name: str,
        category: str,
        timeout: Optional[int] = None,
    ) -> Tuple[ExecutionResult, ProcessedFeedback]:
        """
        Execute code and learn from the result.
        
        This method:
        1. Executes code in Docker
        2. Processes feedback for learning
        3. Learns new error patterns dynamically
        4. Learns recovery suggestions
        5. Records execution in config and history
        6. Stores experience for future retrieval
        
        Args:
            code: Python code to execute
            scenario_name: Name of the scenario for tracking
            category: Error category for classification
            timeout: Optional timeout override
            
        Returns:
            Tuple of (ExecutionResult, ProcessedFeedback)
        """
        start_time = time.time()
        
        # Execute in Docker
        result = await self.executor.execute_python(code, timeout=timeout or self.timeout)
        
        # Process feedback
        feedback = await self.processor.process_result(result)
        
        # Update persisted stats
        self._persisted_stats["total_executions"] = self._persisted_stats.get("total_executions", 0) + 1
        if not result.success:
            self._persisted_stats["total_errors"] = self._persisted_stats.get("total_errors", 0) + 1
        
        # Record in execution history
        self.execution_history.append({
            "scenario": scenario_name,
            "category": category,
            "success": result.success,
            "error_category": feedback.error_category.value if feedback.error_category else None,
            "execution_time": result.execution_time,
            "timestamp": time.time(),
        })
        
        # ═══════════════════════════════════════════════════════════════════════
        # DYNAMIC LEARNING
        # ═══════════════════════════════════════════════════════════════════════
        
        if not result.success:
            error_key = category
            
            # Learn the pattern from this error
            if result.stderr:
                lines = [l for l in result.stderr.split('\n') if l.strip()]
                if lines:
                    pattern = lines[-1][:80]  # Last line often has the error
                    was_new = self.config.add_error_pattern(pattern, error_key)
                    if was_new:
                        self.learned_patterns[pattern] = error_key
            
            # Learn recovery if suggested
            if feedback.suggested_recovery:
                was_new = self.config.add_recovery_solution(error_key, feedback.suggested_recovery)
                if was_new:
                    self.learned_recoveries[error_key] = feedback.suggested_recovery
            
            # Record in config
            self.config.record_execution(
                code=code[:100],
                success=False,
                error_type=error_key,
                error_message=result.stderr[:200] if result.stderr else None,
                execution_time=result.execution_time,
            )
        else:
            self.config.record_execution(
                code=code[:100],
                success=True,
                execution_time=result.execution_time,
            )
        
        # Store experience
        outcome = "success" if result.success else "failure"
        lessons = ""
        if result.success:
            lessons = f"Code executed successfully. Output: {result.stdout[:100] if result.stdout else 'None'}"
        else:
            lessons = f"Error occurred: {category}. "
            if feedback.suggested_recovery:
                lessons += f"Recovery: {feedback.suggested_recovery}"
        
        self.store_experience(
            task=scenario_name,
            outcome=outcome,
            strategy="docker_execution",
            lessons=lessons,
            error_type=category if not result.success else None,
            code_snippet=code[:200],
        )
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_learning()
            self._save_execution_stats()
        
        return result, feedback
    
    def store_experience(
        self,
        task: str,
        outcome: str,
        strategy: str,
        lessons: str,
        error_type: Optional[str] = None,
        code_snippet: Optional[str] = None,
    ) -> bool:
        """
        Store an experience for future retrieval.
        
        Args:
            task: What the task was
            outcome: success/failure/partial
            strategy: What approach was used
            lessons: Key lessons learned
            error_type: Optional error category
            code_snippet: Optional code that was executed
            
        Returns:
            True if stored successfully
        """
        experience = {
            "task": task,
            "outcome": outcome,
            "strategy": strategy,
            "lessons": lessons,
            "error_type": error_type,
            "code_snippet": code_snippet[:200] if code_snippet else None,
            "timestamp": time.time(),
        }
        
        # Store in JSON
        self.experiences.append(experience)
        self._save_experiences()
        
        # Store in vector store if available
        if self.vector_store:
            try:
                doc_content = f"Task: {task}\nOutcome: {outcome}\nStrategy: {strategy}\nLessons: {lessons}"
                if error_type:
                    doc_content += f"\nError Type: {error_type}"
                
                self.vector_store.add_texts(
                    texts=[doc_content],
                    metadatas=[{
                        "outcome": outcome,
                        "error_type": error_type or "none",
                        "timestamp": time.time(),
                    }],
                )
            except Exception:
                pass
        
        return True
    
    def retrieve_similar_experiences(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve similar past experiences using semantic search.
        
        Args:
            query: What to search for
            top_k: Number of results to return
            
        Returns:
            List of similar experiences
        """
        results = []
        
        # Try vector store first
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=top_k)
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": "vector_store",
                    })
                if results:
                    return results
            except Exception:
                pass
        
        # Fallback: keyword search in JSON
        query_lower = query.lower()
        for exp in self.experiences:
            score = 0
            if query_lower in exp.get("task", "").lower():
                score += 2
            if query_lower in exp.get("lessons", "").lower():
                score += 1
            
            if score > 0:
                results.append({
                    "content": f"Task: {exp['task']}\nLessons: {exp['lessons']}",
                    "metadata": exp,
                    "source": "json",
                    "score": score,
                })
        
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]
    
    def save_learning(self) -> None:
        """Save learned config to disk."""
        self.config.save_to_json()
    
    def load_learning(self) -> None:
        """Load learned config from disk."""
        self.config.load_from_json()
    
    def check_known_pattern(self, error_text: str) -> Optional[str]:
        """
        Check if an error pattern is known.
        
        Args:
            error_text: The error text to check
            
        Returns:
            The category if known, None otherwise
        """
        return self.config.check_known_pattern(error_text)
    
    def get_recovery_for_error(self, error_type: str) -> Optional[str]:
        """
        Get recovery suggestion for an error type.
        
        Args:
            error_type: The error category
            
        Returns:
            Recovery suggestion if known
        """
        return self.config.get_recovery_for_error(error_type)
    
    def export_learned_rules(self) -> List[str]:
        """
        Export learned rules for COMPASS prompt evolution.
        
        Returns:
            List of rule strings
        """
        return self.config.export_learned_rules()
    
    def get_stats(self) -> Dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            "executor_stats": self.executor.get_stats(),
            "processor_stats": self.processor.get_stats(),
            "config_stats": self.config.get_stats(),
            "patterns_learned": len(self.learned_patterns),
            "recoveries_learned": len(self.learned_recoveries),
            "total_executions": len(self.execution_history),
            "persisted_stats": self._persisted_stats,
        }
    
    def get_data_paths(self) -> Dict[str, str]:
        """
        Get all data persistence paths.
        
        Returns:
            Dictionary with paths for all persisted data
        """
        return {
            "data_dir": str(self.data_dir),
            "coding_config": str(self.config._config_path),
            "execution_stats": str(self.data_dir / "execution_stats.json"),
            "experiences_json": str(self.data_dir / "experiences.json"),
            "chroma_db": str(self.data_dir / "chroma_db"),
        }
    
    def get_persisted_stats(self) -> Dict:
        """
        Get persisted statistics across all sessions.
        
        Returns:
            Dictionary with total executions, errors, etc.
        """
        return {
            **self._persisted_stats,
            "patterns_learned": len(self.config.learned_error_patterns),
            "recoveries_learned": len(self.config.learned_recovery_solutions),
            "experiences_count": len(self.experiences),
        }
    
    def get_experience_stats(self) -> Dict:
        """
        Get statistics about stored experiences.
        
        Returns:
            Dictionary with experience statistics
        """
        stats = {
            "total_experiences": len(self.experiences),
            "vector_store_available": self.vector_store is not None,
        }
        
        # Count by outcome
        outcomes = {}
        for exp in self.experiences:
            outcome = exp.get("outcome", "unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        stats["by_outcome"] = outcomes
        
        # Count by error type
        error_types = {}
        for exp in self.experiences:
            err = exp.get("error_type")
            if err:
                error_types[err] = error_types.get(err, 0) + 1
        stats["by_error_type"] = error_types
        
        return stats
    
    async def cleanup(self) -> None:
        """Clean up resources (Docker executor)."""
        await self.executor.cleanup()
