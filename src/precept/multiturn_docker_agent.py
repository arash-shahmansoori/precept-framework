"""
Multi-Turn Docker Agent for PRECEPT.

Combines multi-turn conversation capabilities with real Docker code execution
and dynamic learning from execution feedback.

This agent supports:
1. Multi-turn conversations with stateful context
2. Real Docker-based code execution (via CodeExecutionManager)
3. Dynamic learning from execution errors (DynamicCodingConfig)
4. Recovery suggestion learning
5. Pattern persistence and loading
6. Conversation history persistence
7. Integration with PRECEPT data directory structure

Data Directory Structure:
    data/
    ├── precept_multiturn/
    │   ├── coding_config.json      # Learned patterns & recoveries
    │   ├── conversation_history/   # Saved conversation sessions
    │   │   └── {session_id}.json
    │   └── execution_stats.json    # Execution statistics
    └── chroma_precept/               # Vector store (if integrated)

Usage:
    from precept import MultiTurnDockerAgent
    
    # Create agent with default data directory
    agent = MultiTurnDockerAgent(enable_docker=True)
    
    # Or specify custom data directory
    agent = MultiTurnDockerAgent(
        enable_docker=True,
        data_dir=Path("./my_data")
    )
    
    # Start conversation session
    agent.start_session("my-session")
    
    # Execute code in conversation turns
    result, response = await agent.chat_with_code(
        message="Let me try importing this package",
        code="import some_package"
    )
    
    # Get session summary
    summary = agent.get_session_summary()
    
    # Save learned patterns (auto-saved to data directory)
    agent.save_learning()
"""

import json
import time
from dataclasses import dataclass, asdict
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
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "precept_multiturn"


@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""
    turn_number: int
    user_message: str
    code_executed: Optional[str]
    success: bool
    output: Optional[str]
    error: Optional[str]
    error_category: Optional[str]
    execution_time: float


class MultiTurnDockerAgent:
    """
    Agent that supports multi-turn conversations with real Docker code execution.
    
    Combines:
    1. Conversation history (stateful multi-turn)
    2. Docker code execution (CodeExecutionManager)
    3. Dynamic learning (DynamicCodingConfig)
    4. Recovery solution learning
    
    Features:
    - Execute Python code in Docker containers
    - Learn error patterns dynamically
    - Learn recovery suggestions
    - Persist and load learned patterns
    - Track conversation history
    - Export learned rules for COMPASS prompt evolution
    
    Usage:
        agent = MultiTurnDockerAgent(enable_docker=True)
        agent.start_session("session-1")
        
        result, response = await agent.chat_with_code(
            "I'm trying to import numpy",
            "import numpy as np\\nprint(np.__version__)"
        )
        
        agent.save_learning()
    """
    
    def __init__(
        self,
        enable_docker: bool = True,
        data_dir: Optional[Path] = None,
        timeout: int = 30,
        auto_save: bool = True,
        enable_vector_store: bool = True,
    ):
        """
        Initialize the multi-turn Docker agent.
        
        Args:
            enable_docker: Whether to enable Docker execution (fallback to subprocess if False)
            data_dir: Directory for persisting data (defaults to data/precept_multiturn/)
            timeout: Default timeout for code execution in seconds
            auto_save: Whether to auto-save after each session ends
            enable_vector_store: Whether to enable ChromaDB vector store for semantic search
        """
        self.executor = CodeExecutionManager(enable_docker=enable_docker)
        self.processor = ExecutionFeedbackProcessor(enable_llm_categorization=False)
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
        
        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.session_id: Optional[str] = None
        self._session_active: bool = False
        self._session_start_time: float = 0.0
        
        # Learning state
        self.learned_patterns: Dict[str, str] = {}  # pattern -> category
        self.learned_recoveries: Dict[str, str] = {}  # category -> recovery suggestion
        
        # Load execution stats
        self._load_execution_stats()
    
    def _setup_data_directory(self) -> None:
        """Setup the data directory structure."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "conversation_history").mkdir(exist_ok=True)
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
            # Use a timeout for embeddings initialization
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                timeout=10,  # 10 second timeout
            )
            self.vector_store = Chroma(
                collection_name="precept_multiturn_experiences",
                embedding_function=self.embeddings,
                persist_directory=str(self.data_dir / "chroma_db"),
            )
            return True
        except Exception as e:
            # Silently fail - vector store is optional
            self.vector_store = None
            self.embeddings = None
            return False
    
    def _load_experiences(self) -> None:
        """Load experiences from JSON (fallback for vector store)."""
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
                self._persisted_stats = {"total_sessions": 0, "total_turns": 0}
        else:
            self._persisted_stats = {"total_sessions": 0, "total_turns": 0}
    
    def _save_execution_stats(self) -> None:
        """Save execution stats to disk."""
        stats_path = self.data_dir / "execution_stats.json"
        try:
            with open(stats_path, 'w') as f:
                json.dump(self._persisted_stats, f, indent=2)
        except Exception:
            pass  # Silently ignore save errors
    
    @property
    def is_docker_available(self) -> bool:
        """Check if Docker is available for code execution."""
        return self.executor.is_docker_available()
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional session identifier (auto-generated if not provided)
            
        Returns:
            The session ID
        """
        import uuid
        
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._session_active = True
        self._session_start_time = time.time()
        self.conversation_history = []
        
        # Update stats
        self._persisted_stats["total_sessions"] = self._persisted_stats.get("total_sessions", 0) + 1
        
        return self.session_id
    
    def end_session(self, save_conversation: bool = True) -> Dict:
        """
        End the current conversation session.
        
        Args:
            save_conversation: Whether to save conversation history to disk
        
        Returns:
            Session summary with statistics
        """
        summary = self.get_session_summary()
        summary["duration"] = time.time() - self._session_start_time
        
        # Update persisted stats
        self._persisted_stats["total_turns"] = self._persisted_stats.get("total_turns", 0) + len(self.conversation_history)
        
        # Save conversation history
        if save_conversation and self.conversation_history:
            self._save_conversation_history()
        
        # Auto-save learning and stats
        if self.auto_save:
            self.save_learning()
            self._save_execution_stats()
        
        self._session_active = False
        return summary
    
    def _save_conversation_history(self) -> None:
        """Save current conversation history to disk."""
        if not self.session_id or not self.conversation_history:
            return
        
        history_dir = self.data_dir / "conversation_history"
        history_dir.mkdir(exist_ok=True)
        
        history_file = history_dir / f"{self.session_id}.json"
        
        history_data = {
            "session_id": self.session_id,
            "start_time": self._session_start_time,
            "end_time": time.time(),
            "turns": [asdict(turn) for turn in self.conversation_history],
            "patterns_learned": self.learned_patterns,
            "recoveries_learned": self.learned_recoveries,
        }
        
        try:
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
        except Exception:
            pass  # Silently ignore save errors
    
    def load_conversation_history(self, session_id: str) -> Optional[Dict]:
        """
        Load a previous conversation session.
        
        Args:
            session_id: The session ID to load
            
        Returns:
            The conversation history dict or None if not found
        """
        history_file = self.data_dir / "conversation_history" / f"{session_id}.json"
        
        if not history_file.exists():
            return None
        
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def list_saved_sessions(self) -> List[str]:
        """
        List all saved conversation sessions.
        
        Returns:
            List of session IDs
        """
        history_dir = self.data_dir / "conversation_history"
        if not history_dir.exists():
            return []
        
        return [f.stem for f in history_dir.glob("*.json")]
    
    # =========================================================================
    # EXPERIENCE STORAGE & RETRIEVAL (Vector Store + JSON)
    # =========================================================================
    
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
        
        Stores in both:
        1. ChromaDB vector store (for semantic search)
        2. JSON file (for persistence fallback)
        
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
            "session_id": self.session_id,
        }
        
        # Store in JSON (always works)
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
                        "session_id": self.session_id or "unknown",
                    }],
                )
                return True
            except Exception:
                pass  # Vector store is optional
        
        return True
    
    def retrieve_similar_experiences(
        self,
        query: str,
        top_k: int = 3,
        filter_outcome: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve similar past experiences using semantic search.
        
        Args:
            query: What to search for
            top_k: Number of results to return
            filter_outcome: Optional filter by outcome (success/failure)
            
        Returns:
            List of similar experiences
        """
        results = []
        
        # Try vector store first (semantic search)
        if self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=top_k)
                for doc in docs:
                    if filter_outcome and doc.metadata.get("outcome") != filter_outcome:
                        continue
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": "vector_store",
                    })
                if results:
                    return results
            except Exception:
                pass  # Fall back to JSON search
        
        # Fallback: Simple keyword search in JSON experiences
        query_lower = query.lower()
        for exp in self.experiences:
            score = 0
            if query_lower in exp.get("task", "").lower():
                score += 2
            if query_lower in exp.get("lessons", "").lower():
                score += 1
            if query_lower in exp.get("strategy", "").lower():
                score += 1
            
            if score > 0:
                if filter_outcome and exp.get("outcome") != filter_outcome:
                    continue
                results.append({
                    "content": f"Task: {exp['task']}\nLessons: {exp['lessons']}",
                    "metadata": exp,
                    "source": "json",
                    "score": score,
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:top_k]
    
    def get_experience_stats(self) -> Dict:
        """
        Get statistics about stored experiences.
        
        Returns:
            Dictionary with experience statistics
        """
        stats = {
            "total_experiences": len(self.experiences),
            "vector_store_available": self.vector_store is not None,
            "experiences_path": str(self.data_dir / "experiences.json"),
            "chroma_path": str(self.data_dir / "chroma_db"),
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
    
    async def chat_with_code(
        self,
        message: str,
        code: str,
        timeout: Optional[int] = None,
    ) -> Tuple[ExecutionResult, str]:
        """
        Process a conversation turn that includes code execution.
        
        This method:
        1. Executes code in Docker
        2. Processes feedback for learning
        3. Learns new error patterns dynamically
        4. Learns recovery suggestions
        5. Records execution in config
        6. Adds turn to conversation history
        
        Args:
            message: The user's message/explanation
            code: Python code to execute in Docker
            timeout: Optional timeout override for this execution
            
        Returns:
            Tuple of (ExecutionResult, response_message)
        """
        # Ensure session is active
        if not self._session_active:
            self.start_session()
        
        # Execute code in Docker
        result = await self.executor.execute_python(
            code, 
            timeout=timeout or self.timeout
        )
        
        # Process feedback for learning
        feedback = await self.processor.process_result(result)
        
        # ═══════════════════════════════════════════════════════════════════════
        # DYNAMIC CONFIG UPDATE
        # ═══════════════════════════════════════════════════════════════════════
        
        error_category = None
        
        if not result.success and result.stderr:
            error_lines = [l for l in result.stderr.split('\n') if l.strip()]
            if error_lines:
                pattern = error_lines[-1][:80]
                category = feedback.error_category.value if feedback.error_category else "UNKNOWN"
                error_category = category
                
                # 1. Learn error pattern
                was_new_pattern = self.config.add_error_pattern(pattern, category)
                if was_new_pattern:
                    self.learned_patterns[pattern] = category
                
                # 2. Learn recovery solution
                if feedback.suggested_recovery:
                    was_new_recovery = self.config.add_recovery_solution(
                        category, 
                        feedback.suggested_recovery
                    )
                    if was_new_recovery:
                        self.learned_recoveries[category] = feedback.suggested_recovery
        
        # Record in conversation history
        turn = ConversationTurn(
            turn_number=len(self.conversation_history) + 1,
            user_message=message,
            code_executed=code[:500] if len(code) > 500 else code,
            success=result.success,
            output=result.stdout[:500] if result.stdout else None,
            error=result.stderr[:500] if result.stderr else None,
            error_category=error_category,
            execution_time=result.execution_time,
        )
        self.conversation_history.append(turn)
        
        # 3. Record execution in config
        self.config.record_execution(
            code=code[:100],
            success=result.success,
            error_type=error_category,
            execution_time=result.execution_time,
        )
        
        # 4. Store experience for future retrieval (Vector Store + JSON)
        outcome = "success" if result.success else "failure"
        lessons = ""
        if result.success:
            lessons = f"Code executed successfully. Output: {result.stdout[:100] if result.stdout else 'None'}"
        else:
            lessons = f"Error occurred: {error_category}. "
            if feedback.suggested_recovery:
                lessons += f"Recovery: {feedback.suggested_recovery}"
        
        self.store_experience(
            task=message,
            outcome=outcome,
            strategy="docker_execution",
            lessons=lessons,
            error_type=error_category,
            code_snippet=code[:200],
        )
        
        # Generate response
        if result.success:
            response = f"✅ Code executed successfully!"
            if result.stdout:
                response += f"\nOutput: {result.stdout[:300]}"
        else:
            error_type = error_category or "Unknown"
            response = f"❌ Error ({error_type})"
            if result.stderr:
                response += f": {result.stderr[:200]}"
            
            # Check if we have a learned recovery for this error type
            if error_type in self.learned_recoveries:
                response += f"\n💡 Previously learned fix: {self.learned_recoveries[error_type]}"
            elif feedback.suggested_recovery:
                response += f"\n💡 Suggestion: {feedback.suggested_recovery}"
        
        return result, response
    
    async def chat(
        self,
        message: str,
    ) -> str:
        """
        Process a conversation turn without code execution.
        
        For turns that are just discussion without code.
        
        Args:
            message: The user's message
            
        Returns:
            Acknowledgment response
        """
        if not self._session_active:
            self.start_session()
        
        turn = ConversationTurn(
            turn_number=len(self.conversation_history) + 1,
            user_message=message,
            code_executed=None,
            success=True,
            output=None,
            error=None,
            error_category=None,
            execution_time=0.0,
        )
        self.conversation_history.append(turn)
        
        return f"Received: {message[:100]}..."
    
    def get_session_summary(self) -> Dict:
        """
        Get summary of current session.
        
        Returns:
            Dictionary with session statistics
        """
        if not self.conversation_history:
            return {
                "session_id": self.session_id,
                "turns": 0,
                "success_rate": 0.0,
                "patterns_learned": 0,
                "recoveries_learned": 0,
            }
        
        code_turns = [t for t in self.conversation_history if t.code_executed]
        successful = sum(1 for t in code_turns if t.success)
        
        return {
            "session_id": self.session_id,
            "turns": len(self.conversation_history),
            "code_executions": len(code_turns),
            "successful": successful,
            "success_rate": successful / len(code_turns) if code_turns else 1.0,
            "patterns_learned": len(self.learned_patterns),
            "recoveries_learned": len(self.learned_recoveries),
            "total_execution_time": sum(t.execution_time for t in self.conversation_history),
        }
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get the conversation history as a list of dictionaries.
        
        Returns:
            List of turn dictionaries
        """
        return [
            {
                "turn": t.turn_number,
                "message": t.user_message,
                "code": t.code_executed,
                "success": t.success,
                "output": t.output,
                "error": t.error,
                "error_category": t.error_category,
            }
            for t in self.conversation_history
        ]
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation while keeping the session active.
        
        Clears conversation history but preserves learned patterns.
        """
        self.conversation_history = []
    
    def save_learning(self, path: Optional[Path] = None) -> None:
        """
        Save learned patterns and recoveries to disk.
        
        Args:
            path: Optional path override for saving
        """
        if path:
            self.config._config_path = path
        self.config.save_to_json()
    
    def load_learning(self, path: Optional[Path] = None) -> None:
        """
        Load learned patterns from disk.
        
        Args:
            path: Optional path override for loading
        """
        if path:
            self.config._config_path = path
        self.config.load_from_json()
    
    def check_known_pattern(self, error_text: str) -> Optional[str]:
        """
        Check if an error pattern is already known.
        
        Args:
            error_text: The error text to check
            
        Returns:
            The error category if known, None otherwise
        """
        return self.config.get_error_code(error_text)
    
    def get_recovery_for_error(self, error_code: str) -> List[str]:
        """
        Get learned recovery solutions for an error code.
        
        Args:
            error_code: The error category code
            
        Returns:
            List of recovery suggestions
        """
        return self.config.get_recovery_solutions(error_code)
    
    def export_learned_rules(self) -> List[str]:
        """
        Export learned rules for COMPASS prompt evolution.
        
        Returns:
            List of human-readable rules
        """
        return self.config.export_learned_rules()
    
    def get_config_stats(self) -> Dict:
        """
        Get dynamic config statistics.
        
        Returns:
            Dictionary with config statistics
        """
        return self.config.get_stats()
    
    def get_executor_stats(self) -> Dict:
        """
        Get code execution statistics.
        
        Returns:
            Dictionary with executor statistics
        """
        return self.executor.get_stats()
    
    def get_data_paths(self) -> Dict[str, str]:
        """
        Get all data persistence paths.
        
        Returns:
            Dictionary with paths for all persisted data
        """
        return {
            "data_dir": str(self.data_dir),
            "coding_config": str(self.config._config_path),
            "conversation_history": str(self.data_dir / "conversation_history"),
            "execution_stats": str(self.data_dir / "execution_stats.json"),
            "experiences_json": str(self.data_dir / "experiences.json"),
            "chroma_db": str(self.data_dir / "chroma_db"),
        }
    
    def get_persisted_stats(self) -> Dict:
        """
        Get persisted statistics across all sessions.
        
        Returns:
            Dictionary with total sessions, turns, etc.
        """
        return {
            **self._persisted_stats,
            "saved_sessions": len(self.list_saved_sessions()),
            "patterns_learned": len(self.config.learned_error_patterns),
            "recoveries_learned": len(self.config.learned_recovery_solutions),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources (Docker executor)."""
        await self.executor.cleanup()
