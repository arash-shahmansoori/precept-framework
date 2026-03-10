"""
Code Execution Manager for PRECEPT Framework.

Provides Docker-based sandboxed Python code execution using AutoGen's
DockerCommandLineCodeExecutor with graceful fallback when Docker is unavailable.

Features:
- Sandboxed Python execution in Docker containers
- Captures stdout, stderr, exit code, and execution time
- Graceful fallback to simulated execution when Docker unavailable
- Structured ExecutionResult for downstream processing

Usage:
    from precept.code_executor import CodeExecutionManager

    executor = CodeExecutionManager(enable_docker=True)
    result = await executor.execute_python("print('Hello, World!')")

    if result.success:
        print(f"Output: {result.stdout}")
    else:
        print(f"Error: {result.stderr}")
"""

import asyncio
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# AutoGen Docker executor imports
try:
    from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
    from autogen_core.code_executor import CodeBlock
    from autogen_core import CancellationToken

    DOCKER_EXECUTOR_AVAILABLE = True
except ImportError:
    DOCKER_EXECUTOR_AVAILABLE = False
    DockerCommandLineCodeExecutor = None
    CodeBlock = None
    CancellationToken = None


@dataclass
class ExecutionResult:
    """
    Structured result from code execution.

    Contains all information needed for learning and error categorization.
    """

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    code_executed: str = ""
    error_type: Optional[str] = None  # Auto-categorized error type
    error_message: Optional[str] = None  # Extracted error message
    warnings: List[str] = field(default_factory=list)
    traceback: Optional[str] = None  # Full traceback if available

    def has_warnings(self) -> bool:
        """Check if execution produced warnings."""
        return len(self.warnings) > 0

    def has_error(self) -> bool:
        """Check if execution produced an error."""
        return not self.success or self.exit_code != 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "execution_time": self.execution_time,
            "code_executed": self.code_executed,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "traceback": self.traceback,
        }


class CodeExecutionManager:
    """
    Manages code execution with Docker sandboxing.

    Wraps AutoGen's DockerCommandLineCodeExecutor for sandboxed Python
    execution. Provides graceful fallback when Docker is unavailable.

    Features:
    - Docker-based sandboxed execution (when available)
    - Graceful fallback to local subprocess execution
    - Configurable timeout and work directory
    - Structured execution results

    Usage:
        executor = CodeExecutionManager(enable_docker=True)

        if executor.is_docker_available():
            result = await executor.execute_python("print('Hello!')")
        else:
            print("Docker not available, using fallback")
            result = await executor.execute_python("print('Hello!')")
    """

    # Default Docker image for Python execution
    # Use full registry path to avoid docker-py resolution issues
    DEFAULT_IMAGE = "docker.io/library/python:3.11-slim"

    # Default timeout in seconds
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        enable_docker: bool = False,
        work_dir: Optional[Path] = None,
        docker_image: str = DEFAULT_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Code Execution Manager.

        Args:
            enable_docker: Whether to attempt Docker execution
            work_dir: Working directory for code execution
            docker_image: Docker image to use for execution
            timeout: Default timeout for code execution in seconds
        """
        self.enable_docker = enable_docker
        self.work_dir = work_dir or Path(tempfile.gettempdir()) / "precept_execution"
        self.docker_image = docker_image
        self.timeout = timeout

        # Ensure work directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Docker executor (initialized lazily)
        self._docker_executor: Optional[DockerCommandLineCodeExecutor] = None
        self._docker_available: Optional[bool] = None
        self._docker_started: bool = False

        # Execution statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.docker_executions = 0
        self.fallback_executions = 0

    def is_docker_available(self) -> bool:
        """
        Check if Docker is available for execution.

        Checks both:
        1. AutoGen Docker executor module is installed
        2. Docker daemon is running and accessible

        Returns:
            True if Docker execution is possible
        """
        if self._docker_available is not None:
            return self._docker_available

        # Check if module is available
        if not DOCKER_EXECUTOR_AVAILABLE:
            self._docker_available = False
            return False

        # Check if Docker daemon is running
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            self._docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            self._docker_available = False

        return self._docker_available

    async def _get_docker_executor(self) -> Optional[DockerCommandLineCodeExecutor]:
        """
        Get or create the Docker executor instance.

        The executor is started on first use and kept running for efficiency.

        Returns:
            DockerCommandLineCodeExecutor instance or None if unavailable
        """
        if not self.enable_docker or not self.is_docker_available():
            return None

        if self._docker_executor is None:
            self._docker_executor = DockerCommandLineCodeExecutor(
                image=self.docker_image,
                timeout=self.timeout,
                work_dir=str(self.work_dir),
            )
            # Start the executor (required before use)
            await self._docker_executor.start()
            self._docker_started = True

        return self._docker_executor

    async def execute_python(
        self,
        code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.

        Attempts Docker execution first (if enabled and available),
        falls back to local subprocess execution otherwise.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds (overrides default)

        Returns:
            ExecutionResult with execution details
        """
        self.total_executions += 1
        timeout = timeout or self.timeout
        start_time = time.time()

        # Check if Docker is available
        use_docker = self.enable_docker and self.is_docker_available()

        if use_docker:
            result = await self._execute_with_docker_context(code, timeout)
            
            # If Docker failed with infrastructure error, fall back to subprocess
            if not result.success and result.error_type == "DOCKER-ERROR":
                # Docker infrastructure failed, try subprocess
                result = await self._execute_with_subprocess(code, timeout)
                self.fallback_executions += 1
            else:
                self.docker_executions += 1
        else:
            # Fallback to local subprocess execution
            result = await self._execute_with_subprocess(code, timeout)
            self.fallback_executions += 1

        # Update statistics
        result.execution_time = time.time() - start_time
        result.code_executed = code

        if result.success:
            self.successful_executions += 1

        # Extract warnings from output
        result.warnings = self._extract_warnings(result.stdout + result.stderr)

        return result

    async def _execute_with_docker_context(
        self,
        code: str,
        timeout: int,
    ) -> ExecutionResult:
        """
        Execute code using Docker CLI directly.

        Uses subprocess to call docker CLI which is more reliable than docker-py
        with Docker Desktop due to socket/context issues.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            ExecutionResult from Docker execution
        """
        # Use Docker CLI directly (more reliable than docker-py with Docker Desktop)
        return await self._execute_with_docker_cli(code, timeout)

    async def _execute_with_docker_cli(
        self,
        code: str,
        timeout: int,
    ) -> ExecutionResult:
        """
        Execute code using Docker CLI via subprocess.

        This is more reliable than docker-py with Docker Desktop due to
        socket/context compatibility issues.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            ExecutionResult from Docker execution
        """
        try:
            # Write code to a temp file
            code_file = self.work_dir / "temp_code.py"
            code_file.write_text(code)

            # Build docker command
            # Use short image name (CLI handles resolution correctly)
            image = "python:3.11-slim"
            cmd = [
                "docker", "run",
                "--rm",  # Remove container after execution
                "-v", f"{self.work_dir}:/code",  # Mount code directory
                "-w", "/code",  # Working directory
                "--network", "none",  # No network for security
                image,
                "python", "/code/temp_code.py"
            ]

            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                exit_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=f"Docker execution timed out after {timeout} seconds",
                    exit_code=-1,
                    execution_time=float(timeout),
                    error_type="TIMEOUT",
                    error_message=f"Code execution exceeded {timeout}s timeout",
                )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            success = exit_code == 0

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time=0.0,  # Will be set by caller
            )

        except FileNotFoundError:
            # Docker CLI not found
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="Docker CLI not found. Please install Docker.",
                exit_code=-1,
                execution_time=0.0,
                error_type="DOCKER-ERROR",
                error_message="Docker CLI not installed",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                error_type="DOCKER-ERROR",
                error_message=f"Docker execution failed: {str(e)}",
            )
        finally:
            # Cleanup temp file
            try:
                if code_file.exists():
                    code_file.unlink()
            except Exception:
                pass

    async def _execute_with_docker(
        self,
        code: str,
        executor: DockerCommandLineCodeExecutor,
        timeout: int,
    ) -> ExecutionResult:
        """
        Execute code using Docker container.

        Args:
            code: Python code to execute
            executor: DockerCommandLineCodeExecutor instance
            timeout: Timeout in seconds

        Returns:
            ExecutionResult from Docker execution
        """
        try:
            # Create code block for AutoGen executor
            code_block = CodeBlock(language="python", code=code)

            # Create cancellation token (required by AutoGen 0.7+)
            cancellation_token = CancellationToken()

            # Execute in Docker with cancellation token
            result = await executor.execute_code_blocks(
                code_blocks=[code_block],
                cancellation_token=cancellation_token,
            )

            # Parse AutoGen result
            output = result.output if hasattr(result, "output") else str(result)
            exit_code = result.exit_code if hasattr(result, "exit_code") else 0

            # Determine success
            success = exit_code == 0

            # Split output into stdout/stderr (AutoGen combines them)
            stdout, stderr = self._split_output(output, success)

            return ExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                execution_time=0.0,  # Will be set by caller
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {timeout} seconds",
                exit_code=-1,
                execution_time=float(timeout),
                error_type="TIMEOUT",
                error_message=f"Code execution exceeded {timeout}s timeout",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                error_type="DOCKER-ERROR",
                error_message=f"Docker execution failed: {str(e)}",
            )

    async def _execute_with_subprocess(
        self,
        code: str,
        timeout: int,
    ) -> ExecutionResult:
        """
        Execute code using local subprocess (fallback).

        WARNING: This is less secure than Docker execution.
        Only used when Docker is unavailable.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds

        Returns:
            ExecutionResult from subprocess execution
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=self.work_dir,
                delete=False,
            ) as f:
                f.write(code)
                temp_file = Path(f.name)

            try:
                # Execute with subprocess
                process = await asyncio.create_subprocess_exec(
                    "python3",
                    str(temp_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.work_dir),
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )

                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                    exit_code = process.returncode or 0

                    return ExecutionResult(
                        success=exit_code == 0,
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=exit_code,
                        execution_time=0.0,
                        traceback=self._extract_traceback(stderr)
                        if exit_code != 0
                        else None,
                    )

                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ExecutionResult(
                        success=False,
                        stdout="",
                        stderr=f"Execution timed out after {timeout} seconds",
                        exit_code=-1,
                        execution_time=float(timeout),
                        error_type="TIMEOUT",
                        error_message=f"Code execution exceeded {timeout}s timeout",
                    )

            finally:
                # Clean up temp file
                temp_file.unlink(missing_ok=True)

        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time=0.0,
                error_type="SUBPROCESS-ERROR",
                error_message=f"Subprocess execution failed: {str(e)}",
            )

    def _split_output(self, output: str, success: bool) -> tuple:
        """
        Split combined output into stdout and stderr.

        AutoGen's Docker executor combines stdout/stderr.
        This attempts to separate them based on error patterns.

        Args:
            output: Combined output string
            success: Whether execution was successful

        Returns:
            Tuple of (stdout, stderr)
        """
        if success:
            return output, ""

        # Look for error indicators
        error_indicators = [
            "Traceback (most recent call last):",
            "Error:",
            "Exception:",
            "SyntaxError:",
            "NameError:",
            "TypeError:",
            "ValueError:",
            "ImportError:",
            "ModuleNotFoundError:",
            "AttributeError:",
            "KeyError:",
            "IndexError:",
            "RuntimeError:",
            "OSError:",
            "FileNotFoundError:",
        ]

        for indicator in error_indicators:
            if indicator in output:
                idx = output.find(indicator)
                return output[:idx].strip(), output[idx:].strip()

        # If no indicator found, treat all as stderr on failure
        return "", output

    def _extract_warnings(self, output: str) -> List[str]:
        """
        Extract warning messages from execution output.

        Args:
            output: Combined stdout/stderr output

        Returns:
            List of warning messages
        """
        warnings = []

        for line in output.split("\n"):
            line_lower = line.lower()
            if "warning:" in line_lower or "deprecationwarning" in line_lower:
                warnings.append(line.strip())
            elif line.startswith("UserWarning:") or line.startswith("FutureWarning:"):
                warnings.append(line.strip())

        return warnings

    def _extract_traceback(self, stderr: str) -> Optional[str]:
        """
        Extract Python traceback from stderr.

        Args:
            stderr: Standard error output

        Returns:
            Extracted traceback or None
        """
        if "Traceback (most recent call last):" in stderr:
            idx = stderr.find("Traceback (most recent call last):")
            return stderr[idx:]
        return None

    async def execute_with_packages(
        self,
        code: str,
        packages: List[str],
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute code after installing required packages.

        Installs packages in the Docker container before running code.

        Args:
            code: Python code to execute
            packages: List of packages to install
            timeout: Timeout in seconds

        Returns:
            ExecutionResult from execution
        """
        if not packages:
            return await self.execute_python(code, timeout)

        # Build pip install command
        install_code = f"""
import subprocess
import sys

# Install required packages
packages = {packages}
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Run the actual code
{code}
"""

        # Use longer timeout for package installation
        install_timeout = (timeout or self.timeout) + len(packages) * 30

        return await self.execute_python(install_code, install_timeout)

    def get_stats(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": (
                self.successful_executions / self.total_executions
                if self.total_executions > 0
                else 0.0
            ),
            "docker_executions": self.docker_executions,
            "fallback_executions": self.fallback_executions,
            "docker_available": self.is_docker_available(),
            "docker_enabled": self.enable_docker,
        }

    async def cleanup(self):
        """
        Clean up resources.

        Should be called when done with the executor.
        """
        if self._docker_executor is not None and self._docker_started:
            try:
                await self._docker_executor.stop()
            except Exception:
                pass
            self._docker_executor = None
            self._docker_started = False
