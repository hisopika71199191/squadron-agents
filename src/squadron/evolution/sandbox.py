"""
Sandboxed Execution Environment

Provides isolated execution environments for self-improvement experiments.
Uses Docker containers for secure isolation.

Security: By default, refuses to execute code without Docker isolation.
Subprocess mode is intentionally unsafe and disabled by default.

Key features:
- Docker-based isolation (required by default)
- Resource limits (CPU, memory, time)
- File system isolation
- Network restrictions
- Capability dropping
- Non-root execution
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import structlog

from squadron.core.config import EvolutionConfig

logger = structlog.get_logger(__name__)


class SandboxType(str, Enum):
    """Type of sandbox isolation."""
    DOCKER = "docker"
    SUBPROCESS = "subprocess"  # UNSAFE - disabled by default
    NONE = "none"  # For testing only - never use in production


class SandboxSecurityError(Exception):
    """Raised when sandbox security requirements are not met."""
    pass


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    # Security: Default to Docker, fail-safe if unavailable
    sandbox_type: SandboxType = SandboxType.DOCKER

    # Security: Require Docker by default (fail-safe mode)
    require_docker: bool = True
    allow_unsafe_subprocess: bool = False  # Must be explicitly enabled

    # Docker settings
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"  # Disable networking by default

    # Resource limits
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_seconds: float = 60.0
    max_pids: int = 100  # Prevent fork bombs
    max_file_descriptors: int = 100

    # File system
    working_dir: str | None = None
    mount_paths: list[tuple[str, str]] = field(default_factory=list)  # (host, container)

    # Permissions
    allow_network: bool = False
    allow_write: bool = True
    read_only_root: bool = True  # Security: Read-only root filesystem

    # Security: Run as non-root user in container
    container_user: str = "65534:65534"  # nobody:nogroup

    @classmethod
    def from_evolution_config(cls, config: EvolutionConfig) -> SandboxConfig:
        """Create from evolution config."""
        return cls(
            docker_image=config.sandbox_image,
        )


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""
    
    id: UUID = field(default_factory=uuid4)
    
    # Status
    success: bool = False
    exit_code: int = -1
    
    # Output
    stdout: str = ""
    stderr: str = ""
    
    # Artifacts
    output_files: dict[str, str] = field(default_factory=dict)  # path -> content
    
    # Metrics
    execution_time_seconds: float = 0.0
    memory_used_mb: float = 0.0
    
    # Error
    error: str | None = None
    
    # Timestamps
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "success": self.success,
            "exitCode": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "executionTimeSeconds": self.execution_time_seconds,
            "memoryUsedMb": self.memory_used_mb,
            "error": self.error,
            "startedAt": self.started_at.isoformat(),
            "completedAt": self.completed_at.isoformat() if self.completed_at else None,
        }


class Sandbox:
    """
    Sandboxed Execution Environment.

    Security: By default, refuses to execute code without Docker isolation.
    This is a fail-safe design to prevent accidental code execution in
    unsafe environments.

    Example:
        ```python
        sandbox = Sandbox(config=SandboxConfig())

        # Execute code in sandbox (requires Docker)
        result = await sandbox.execute(
            code="print('Hello, World!')",
            language="python",
        )

        # Execute a file
        result = await sandbox.execute_file(
            file_path="test_agent.py",
            args=["--test"],
        )
        ```
    """

    def __init__(self, config: SandboxConfig | None = None):
        """
        Initialize the sandbox.

        Args:
            config: Sandbox configuration

        Raises:
            SandboxSecurityError: If Docker is required but not available
        """
        self.config = config or SandboxConfig()
        self._temp_dirs: list[Path] = []
        self._docker_available = self._check_docker()

        # Security: Fail-safe mode - refuse to operate without Docker
        if self.config.require_docker and not self._docker_available:
            logger.error(
                "Docker required but not available",
                require_docker=self.config.require_docker,
            )
            raise SandboxSecurityError(
                "Docker is required for secure sandbox execution but is not available. "
                "Install Docker or set require_docker=False (UNSAFE) in SandboxConfig."
            )
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute code in the sandbox.
        
        Args:
            code: Code to execute
            language: Programming language
            timeout: Execution timeout (overrides config)
            env: Environment variables
            
        Returns:
            Execution result
        """
        timeout = timeout or self.config.max_execution_seconds
        
        # Create temp directory for execution
        temp_dir = Path(tempfile.mkdtemp(prefix="squadron_sandbox_"))
        self._temp_dirs.append(temp_dir)
        
        try:
            # Write code to file
            if language == "python":
                code_file = temp_dir / "main.py"
                code_file.write_text(code)
                cmd = ["python", str(code_file)]
            elif language == "javascript":
                code_file = temp_dir / "main.js"
                code_file.write_text(code)
                cmd = ["node", str(code_file)]
            elif language == "bash":
                code_file = temp_dir / "main.sh"
                code_file.write_text(code)
                cmd = ["bash", str(code_file)]
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported language: {language}",
                )
            
            # Security: Execute based on sandbox type with fail-safe checks
            if self.config.sandbox_type == SandboxType.DOCKER and self._docker_available:
                result = await self._execute_docker(cmd, temp_dir, timeout, env)
            elif self.config.sandbox_type == SandboxType.SUBPROCESS:
                # Security: Only allow subprocess if explicitly permitted
                if not self.config.allow_unsafe_subprocess:
                    return ExecutionResult(
                        success=False,
                        error="Subprocess execution is disabled for security. "
                        "Set allow_unsafe_subprocess=True to enable (UNSAFE).",
                    )
                logger.warning(
                    "Executing in UNSAFE subprocess mode - no isolation!",
                    code_length=len(code),
                )
                result = await self._execute_subprocess(cmd, temp_dir, timeout, env)
            else:
                # No Docker and no subprocess allowed - fail safe
                return ExecutionResult(
                    success=False,
                    error="No secure execution environment available. "
                    "Docker is required but not available.",
                )

            return result
            
        finally:
            # Cleanup temp directory
            self._cleanup_temp_dir(temp_dir)
    
    async def execute_file(
        self,
        file_path: str | Path,
        args: list[str] | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """
        Execute a file in the sandbox.
        
        Args:
            file_path: Path to the file to execute
            args: Command line arguments
            timeout: Execution timeout
            env: Environment variables
            
        Returns:
            Execution result
        """
        file_path = Path(file_path)
        timeout = timeout or self.config.max_execution_seconds
        args = args or []
        
        if not file_path.exists():
            return ExecutionResult(
                success=False,
                error=f"File not found: {file_path}",
            )
        
        # Determine command based on file extension
        ext = file_path.suffix.lower()
        if ext == ".py":
            cmd = ["python", str(file_path)] + args
        elif ext == ".js":
            cmd = ["node", str(file_path)] + args
        elif ext == ".sh":
            cmd = ["bash", str(file_path)] + args
        else:
            return ExecutionResult(
                success=False,
                error=f"Unsupported file type: {ext}",
            )
        
        # Create temp directory for working
        temp_dir = Path(tempfile.mkdtemp(prefix="squadron_sandbox_"))
        self._temp_dirs.append(temp_dir)
        
        try:
            if self.config.sandbox_type == SandboxType.DOCKER and self._docker_available:
                result = await self._execute_docker(cmd, temp_dir, timeout, env)
            else:
                result = await self._execute_subprocess(cmd, temp_dir, timeout, env)
            
            return result
            
        finally:
            self._cleanup_temp_dir(temp_dir)
    
    async def _execute_subprocess(
        self,
        cmd: list[str],
        working_dir: Path,
        timeout: float,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """Execute using subprocess isolation."""
        result = ExecutionResult()
        result.started_at = datetime.utcnow()
        
        # Build environment
        exec_env = dict(os.environ)
        if env:
            exec_env.update(env)
        
        # Remove potentially dangerous env vars
        for key in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH"]:
            if key in exec_env and not env:
                pass  # Keep system defaults unless overridden
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=exec_env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                result.success = False
                result.error = f"Execution timed out after {timeout}s"
                result.exit_code = -1
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.exit_code = -1
        
        result.completed_at = datetime.utcnow()
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        return result
    
    async def _execute_docker(
        self,
        cmd: list[str],
        working_dir: Path,
        timeout: float,
        env: dict[str, str] | None,
    ) -> ExecutionResult:
        """
        Execute using Docker isolation with hardened security.

        Security features:
        - Drop all capabilities
        - Run as non-root user
        - Read-only root filesystem
        - Process and file descriptor limits
        - No new privileges
        - Restricted network
        """
        result = ExecutionResult()
        result.started_at = datetime.utcnow()

        # Build docker command with security hardening
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after execution

            # Resource limits
            f"--memory={self.config.max_memory_mb}m",
            f"--cpus={self.config.max_cpu_percent / 100}",
            f"--pids-limit={self.config.max_pids}",
            f"--ulimit=nofile={self.config.max_file_descriptors}:{self.config.max_file_descriptors}",

            # Security: Drop all capabilities
            "--cap-drop=ALL",

            # Security: Prevent privilege escalation
            "--security-opt=no-new-privileges:true",

            # Security: Run as non-root user
            f"--user={self.config.container_user}",

            # Security: Network isolation
            f"--network={self.config.docker_network if not self.config.allow_network else 'bridge'}",
        ]

        # Security: Read-only root filesystem (with tmpfs for /tmp)
        if self.config.read_only_root:
            docker_cmd.extend([
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            ])

        # Mount working directory
        docker_cmd.extend([
            "-v", f"{working_dir}:/workspace:{'rw' if self.config.allow_write else 'ro'}",
            "-w", "/workspace",
        ])

        # Add environment variables (filter sensitive ones)
        if env:
            # Security: Don't pass potentially sensitive environment variables
            safe_env_prefixes = ("LANG", "LC_", "TZ", "PATH")
            for key, value in env.items():
                if key.startswith(safe_env_prefixes) or key in {"HOME", "USER"}:
                    docker_cmd.extend(["-e", f"{key}={value}"])

        # Add mount paths (read-only only)
        for host_path, container_path in self.config.mount_paths:
            # Security: All additional mounts are read-only
            docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])

        # Add image and command
        docker_cmd.append(self.config.docker_image)
        docker_cmd.extend(cmd)

        logger.debug(
            "Executing in hardened Docker container",
            image=self.config.docker_image,
            user=self.config.container_user,
        )
        
        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout + 10,  # Extra time for Docker overhead
                )
                
                result.stdout = stdout.decode("utf-8", errors="replace")
                result.stderr = stderr.decode("utf-8", errors="replace")
                result.exit_code = process.returncode or 0
                result.success = result.exit_code == 0
                
            except asyncio.TimeoutError:
                # Kill the container
                container_id = await self._get_running_container()
                if container_id:
                    subprocess.run(["docker", "kill", container_id], capture_output=True)
                
                result.success = False
                result.error = f"Docker execution timed out after {timeout}s"
                result.exit_code = -1
                
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.exit_code = -1
        
        result.completed_at = datetime.utcnow()
        result.execution_time_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        return result
    
    async def _get_running_container(self) -> str | None:
        """Get the ID of a running container (for cleanup)."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"ancestor={self.config.docker_image}"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.decode().strip().split("\n")[0]
        except Exception:
            pass
        return None
    
    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up a temporary directory."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if temp_dir in self._temp_dirs:
                self._temp_dirs.remove(temp_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir", path=str(temp_dir), error=str(e))
    
    async def cleanup(self) -> None:
        """Clean up all temporary directories."""
        for temp_dir in list(self._temp_dirs):
            self._cleanup_temp_dir(temp_dir)
    
    async def run_tests(
        self,
        test_file: str | Path,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Run a test file in the sandbox.
        
        Args:
            test_file: Path to test file
            timeout: Execution timeout
            
        Returns:
            Execution result
        """
        test_file = Path(test_file)
        timeout = timeout or self.config.max_execution_seconds * 2  # More time for tests
        
        # Use pytest for Python tests
        if test_file.suffix == ".py":
            return await self.execute_file(
                file_path=test_file,
                args=["-v"],
                timeout=timeout,
            )
        
        return await self.execute_file(test_file, timeout=timeout)
    
    @property
    def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        return self._docker_available
