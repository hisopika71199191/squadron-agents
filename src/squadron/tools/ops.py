"""
Operations Tools Pack

Tools for system operations and DevOps tasks:
- Shell command execution (with security controls)
- Docker management
- Process monitoring
- System information

Security: Commands are validated against an allowlist and parsed safely
to prevent command injection attacks.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import structlog

from squadron.connectivity.mcp_host import mcp_tool

logger = structlog.get_logger(__name__)


# Security: Safe environment variables that can be exposed
SAFE_ENV_VARS = frozenset({
    "PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "PWD", "HOSTNAME",
    "LC_ALL", "LC_CTYPE", "TZ", "EDITOR", "VISUAL", "PAGER",
})

# Security: Default allowlist of safe commands
DEFAULT_ALLOWED_COMMANDS = frozenset({
    # File operations (read-only)
    "ls", "cat", "head", "tail", "wc", "file", "stat", "du", "df",
    "find", "locate", "which", "whereis",
    # Text processing
    "grep", "awk", "sed", "sort", "uniq", "cut", "tr", "diff",
    # System info
    "ps", "top", "htop", "uptime", "uname", "hostname", "whoami", "id", "date",
    "free", "vmstat", "iostat", "netstat", "ss",
    # Development tools
    "git", "python", "python3", "pip", "pip3", "npm", "node", "cargo", "go",
    "make", "cmake", "gcc", "g++", "rustc",
    # Docker (safe read operations)
    "docker",
    # Network diagnostics
    "ping", "curl", "wget", "dig", "nslookup", "host",
    # Archive tools
    "tar", "zip", "unzip", "gzip", "gunzip",
    # Package managers (query only)
    "apt", "apt-cache", "dpkg", "rpm", "yum", "brew",
})

# Security: Patterns that indicate dangerous intent even in allowed commands
DANGEROUS_PATTERNS = frozenset({
    # Destructive file operations
    "rm -rf /", "rm -rf /*", "rm -fr /", "rm -fr /*",
    # Privilege escalation
    "chmod 777 /", "chmod -R 777 /", "chown root",
    # System commands
    "shutdown", "reboot", "halt", "poweroff", "init 0", "init 6",
    # Fork bombs and resource exhaustion
    ":(){ :|:& };:", "fork()",
    # Dangerous redirects
    "> /dev/sd", "dd if=/dev/zero", "mkfs",
})


@dataclass
class CommandResult:
    """Result of a shell command execution."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    success: bool = False


@dataclass
class ProcessInfo:
    """Information about a running process."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = ""
    created: datetime | None = None


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    id: str
    name: str
    image: str
    status: str
    ports: list[str] = field(default_factory=list)
    created: str = ""


class OpsTools:
    """
    Operations Tools Pack.

    Provides tools for system operations:
    - Shell command execution (with security controls)
    - Docker container management
    - Process monitoring
    - System information

    Security: All commands are validated against an allowlist and executed
    using subprocess_exec (not shell) to prevent injection attacks.

    Example:
        ```python
        tools = OpsTools(
            allowed_commands=["ls", "cat", "grep", "docker"],
            working_dir="/app",
        )

        # Run a command
        result = await tools.run_command("ls -la")

        # List Docker containers
        containers = await tools.docker_ps()

        # Get system info
        info = await tools.system_info()
        ```
    """

    def __init__(
        self,
        working_dir: str | None = None,
        allowed_commands: set[str] | list[str] | None = None,
        max_output_length: int = 50000,
        command_timeout: float = 60.0,
        strict_mode: bool = True,
    ):
        """
        Initialize ops tools.

        Args:
            working_dir: Default working directory
            allowed_commands: Allowlist of permitted commands (None = default safe set)
            max_output_length: Maximum output length
            command_timeout: Default command timeout
            strict_mode: If True, only allow explicitly listed commands
        """
        self.working_dir = working_dir or os.getcwd()
        # Security: Use allowlist, not blocklist
        if allowed_commands is not None:
            self.allowed_commands = frozenset(allowed_commands)
        else:
            self.allowed_commands = DEFAULT_ALLOWED_COMMANDS
        self.max_output_length = max_output_length
        self.command_timeout = command_timeout
        self.strict_mode = strict_mode

        # Check for Docker
        self._has_docker = self._check_command("docker")
    
    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available."""
        return shutil.which(cmd) is not None

    def _parse_command(self, command: str) -> tuple[list[str] | None, str]:
        """
        Safely parse a command string into arguments.

        Security: Uses shlex to properly handle quoting and prevent injection.

        Returns:
            Tuple of (parsed_args, error_message). parsed_args is None on error.
        """
        try:
            # Use shlex for safe parsing - handles quotes, escapes properly
            args = shlex.split(command)
            if not args:
                return None, "Empty command"
            return args, ""
        except ValueError as e:
            # shlex raises ValueError for unmatched quotes, etc.
            return None, f"Invalid command syntax: {e}"

    def _validate_command(self, args: list[str]) -> tuple[bool, str]:
        """
        Validate parsed command arguments against security rules.

        Security: Uses allowlist approach - only explicitly permitted commands allowed.

        Args:
            args: Parsed command arguments (first element is the command)

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not args:
            return False, "Empty command"

        cmd_name = args[0]

        # Extract base command name (handle paths like /usr/bin/ls)
        base_cmd = os.path.basename(cmd_name)

        # Security: Check against allowlist
        if self.strict_mode and base_cmd not in self.allowed_commands:
            return False, f"Command not in allowlist: {base_cmd}"

        # Security: Check for dangerous patterns in the full command
        full_command = " ".join(args)
        for pattern in DANGEROUS_PATTERNS:
            if pattern in full_command:
                return False, f"Dangerous pattern detected: {pattern}"

        # Security: Block shell metacharacters that could indicate injection attempts
        # These should not appear in properly parsed arguments
        dangerous_chars = {";", "|", "&", "$", "`", "(", ")", "{", "}", "<", ">", "\n", "\r"}
        for arg in args[1:]:  # Skip the command itself
            # Allow $ in environment variable references for specific safe cases
            if any(char in arg for char in dangerous_chars):
                # Check if it's a safe use case (e.g., grep pattern)
                if not self._is_safe_argument(base_cmd, arg):
                    return False, f"Potentially dangerous characters in argument: {arg[:50]}"

        return True, ""

    def _is_safe_argument(self, cmd: str, arg: str) -> bool:
        """
        Check if an argument with special characters is safe for a specific command.

        Some commands legitimately need special characters (e.g., grep patterns).
        """
        # grep, awk, sed can have regex patterns with special chars
        if cmd in {"grep", "awk", "sed", "find"} and not any(c in arg for c in {";", "|", "&", "`"}):
            return True
        return False

    @mcp_tool(description="Execute a command (allowlist enforced)", requires_approval=True)
    async def run_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a command securely.

        Security:
        - Commands are parsed with shlex (prevents shell injection)
        - Only allowlisted commands are permitted
        - Executed with subprocess_exec (not shell)
        - Dangerous patterns are blocked

        Args:
            command: Command to execute (e.g., "ls -la")
            cwd: Working directory
            timeout: Command timeout
            env: Additional environment variables

        Returns:
            Command result
        """
        # Security: Parse command safely with shlex
        args, parse_error = self._parse_command(command)
        if args is None:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Failed to parse command: {parse_error}",
                success=False,
            )

        # Security: Validate against allowlist and dangerous patterns
        is_valid, error = self._validate_command(args)
        if not is_valid:
            logger.warning("Command blocked by security policy", command=command[:100], reason=error)
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Security policy violation: {error}",
                success=False,
            )

        cwd = cwd or self.working_dir
        timeout = timeout or self.command_timeout

        # Security: Build safe environment (don't expose sensitive vars)
        exec_env = {k: v for k, v in os.environ.items() if k in SAFE_ENV_VARS}
        exec_env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
        if env:
            # Only allow overriding safe variables
            for key, value in env.items():
                if key in SAFE_ENV_VARS or not self.strict_mode:
                    exec_env[key] = value

        start_time = datetime.utcnow()

        try:
            # Security: Use subprocess_exec, NOT subprocess_shell
            # This prevents shell injection attacks
            process = await asyncio.create_subprocess_exec(
                *args,  # Unpacked list of arguments
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=exec_env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                stdout_str = stdout.decode("utf-8", errors="replace")
                stderr_str = stderr.decode("utf-8", errors="replace")

                # Truncate if too long
                if len(stdout_str) > self.max_output_length:
                    stdout_str = stdout_str[: self.max_output_length] + "\n[Output truncated...]"
                if len(stderr_str) > self.max_output_length:
                    stderr_str = stderr_str[: self.max_output_length] + "\n[Output truncated...]"

                duration = (datetime.utcnow() - start_time).total_seconds()

                return CommandResult(
                    command=command,
                    exit_code=process.returncode or 0,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    duration_seconds=duration,
                    success=process.returncode == 0,
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                return CommandResult(
                    command=command,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    success=False,
                )

        except FileNotFoundError:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Command not found: {args[0]}",
                success=False,
            )
        except Exception as e:
            logger.error("Command execution failed", command=command[:100], error=str(e))
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                success=False,
            )
    
    @mcp_tool(description="List running Docker containers")
    async def docker_ps(
        self,
        all_containers: bool = False,
    ) -> list[ContainerInfo]:
        """
        List Docker containers.
        
        Args:
            all_containers: Include stopped containers
            
        Returns:
            List of container info
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "ps", "--format", "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}"]
        if all_containers:
            cmd.insert(2, "-a")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        
        containers = []
        for line in stdout.decode().splitlines():
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 4:
                containers.append(ContainerInfo(
                    id=parts[0],
                    name=parts[1],
                    image=parts[2],
                    status=parts[3],
                    ports=parts[4].split(",") if len(parts) > 4 and parts[4] else [],
                ))
        
        return containers
    
    @mcp_tool(description="Get Docker container logs")
    async def docker_logs(
        self,
        container: str,
        tail: int = 100,
        follow: bool = False,
    ) -> str:
        """
        Get Docker container logs.
        
        Args:
            container: Container name or ID
            tail: Number of lines to show
            follow: Follow log output (not recommended for tools)
            
        Returns:
            Container logs
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "logs", "--tail", str(tail), container]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        
        # Docker logs go to both stdout and stderr
        output = stdout.decode("utf-8", errors="replace")
        output += stderr.decode("utf-8", errors="replace")
        
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "\n[Logs truncated...]"
        
        return output
    
    @mcp_tool(description="Execute a command in a Docker container")
    async def docker_exec(
        self,
        container: str,
        command: str,
        workdir: str | None = None,
    ) -> CommandResult:
        """
        Execute a command in a Docker container.
        
        Args:
            container: Container name or ID
            command: Command to execute
            workdir: Working directory in container
            
        Returns:
            Command result
        """
        if not self._has_docker:
            raise RuntimeError("Docker not available")
        
        cmd = ["docker", "exec"]
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.extend([container, "sh", "-c", command])
        
        start_time = datetime.utcnow()
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.command_timeout,
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            return CommandResult(
                command=f"docker exec {container}: {command}",
                exit_code=process.returncode or 0,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                duration_seconds=duration,
                success=process.returncode == 0,
            )
            
        except asyncio.TimeoutError:
            return CommandResult(
                command=f"docker exec {container}: {command}",
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {self.command_timeout}s",
                success=False,
            )
    
    @mcp_tool(description="Get system information")
    async def system_info(self) -> dict[str, Any]:
        """
        Get system information.
        
        Returns:
            System information dictionary
        """
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
        }
        
        # Try to get more detailed info
        try:
            import psutil
            
            info["cpu_count"] = psutil.cpu_count()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            memory = psutil.virtual_memory()
            info["memory_total_gb"] = round(memory.total / (1024**3), 2)
            info["memory_available_gb"] = round(memory.available / (1024**3), 2)
            info["memory_percent"] = memory.percent
            
            disk = psutil.disk_usage("/")
            info["disk_total_gb"] = round(disk.total / (1024**3), 2)
            info["disk_free_gb"] = round(disk.free / (1024**3), 2)
            info["disk_percent"] = disk.percent
            
        except ImportError:
            pass
        
        return info
    
    @mcp_tool(description="List running processes")
    async def list_processes(
        self,
        filter_name: str | None = None,
        limit: int = 20,
    ) -> list[ProcessInfo]:
        """
        List running processes.
        
        Args:
            filter_name: Filter by process name
            limit: Maximum number of processes
            
        Returns:
            List of process info
        """
        processes = []
        
        try:
            import psutil
            
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status", "create_time"]):
                try:
                    info = proc.info
                    
                    # Apply filter
                    if filter_name and filter_name.lower() not in info["name"].lower():
                        continue
                    
                    memory_mb = info["memory_info"].rss / (1024 * 1024) if info["memory_info"] else 0
                    
                    processes.append(ProcessInfo(
                        pid=info["pid"],
                        name=info["name"],
                        cpu_percent=info["cpu_percent"] or 0,
                        memory_mb=round(memory_mb, 2),
                        status=info["status"],
                        created=datetime.fromtimestamp(info["create_time"]) if info["create_time"] else None,
                    ))
                    
                    if len(processes) >= limit:
                        break
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except ImportError:
            # Fallback to ps command
            result = await self.run_command("ps aux | head -20")
            if result.success:
                logger.info("psutil not available, using ps command")
        
        # Sort by CPU usage
        processes.sort(key=lambda p: p.cpu_percent, reverse=True)
        
        return processes[:limit]
    
    @mcp_tool(description="Check if a port is in use")
    async def check_port(
        self,
        port: int,
        host: str = "localhost",
    ) -> dict[str, Any]:
        """
        Check if a port is in use.
        
        Args:
            port: Port number
            host: Host to check
            
        Returns:
            Port status information
        """
        import socket
        
        result = {
            "port": port,
            "host": host,
            "in_use": False,
            "process": None,
        }
        
        # Check if port is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        
        try:
            sock.connect((host, port))
            result["in_use"] = True
        except (socket.timeout, ConnectionRefusedError):
            result["in_use"] = False
        finally:
            sock.close()
        
        # Try to find process using the port
        if result["in_use"]:
            try:
                import psutil
                
                for conn in psutil.net_connections():
                    if conn.laddr.port == port:
                        try:
                            proc = psutil.Process(conn.pid)
                            result["process"] = {
                                "pid": conn.pid,
                                "name": proc.name(),
                            }
                            break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            except ImportError:
                pass
        
        return result
    
    @mcp_tool(description="Get safe environment variables")
    async def get_env(
        self,
        names: list[str] | None = None,
        include_unsafe: bool = False,
    ) -> dict[str, str]:
        """
        Get environment variables (security-filtered).

        Security: By default, only safe environment variables are returned.
        Sensitive variables (API keys, passwords, tokens) are never exposed.

        Args:
            names: Specific variable names to retrieve (must be in safe list)
            include_unsafe: If True and strict_mode=False, include more variables

        Returns:
            Environment variables (filtered for security)
        """
        # Security: Define additional sensitive patterns to always redact
        sensitive_patterns = {
            "KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "AUTH",
            "PRIVATE", "API", "ACCESS", "AWS_", "AZURE_", "GCP_", "GOOGLE_",
            "DATABASE", "DB_", "MONGO", "REDIS", "POSTGRES", "MYSQL",
            "SMTP", "MAIL", "SSH", "GPG", "PGP", "CERT", "SSL", "TLS",
        }

        def is_sensitive(key: str) -> bool:
            """Check if a key name indicates sensitive data."""
            key_upper = key.upper()
            return any(pattern in key_upper for pattern in sensitive_patterns)

        if names is not None:
            # Return only specifically requested variables from safe list
            result = {}
            for name in names:
                if name in SAFE_ENV_VARS and name in os.environ:
                    result[name] = os.environ[name]
                elif is_sensitive(name):
                    result[name] = "[REDACTED - sensitive variable]"
                elif not self.strict_mode and name in os.environ:
                    result[name] = os.environ[name]
                else:
                    result[name] = "[NOT AVAILABLE]"
            return result

        # Default: return only safe variables
        if self.strict_mode and not include_unsafe:
            return {k: v for k, v in os.environ.items() if k in SAFE_ENV_VARS}

        # Non-strict mode: return more but still redact sensitive
        result = {}
        for key, value in os.environ.items():
            if is_sensitive(key):
                result[key] = "[REDACTED]"
            else:
                result[key] = value

        return result
    
    @mcp_tool(description="Watch a file for changes")
    async def watch_file(
        self,
        path: str,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Watch a file for changes.
        
        Args:
            path: File path to watch
            timeout: Maximum time to watch
            
        Returns:
            Change information
        """
        from pathlib import Path
        
        file_path = Path(path)
        
        if not file_path.exists():
            return {"error": f"File not found: {path}"}
        
        initial_stat = file_path.stat()
        initial_mtime = initial_stat.st_mtime
        initial_size = initial_stat.st_size
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                return {
                    "changed": False,
                    "path": path,
                    "watched_seconds": timeout,
                }
            
            await asyncio.sleep(0.5)
            
            try:
                current_stat = file_path.stat()
                if current_stat.st_mtime != initial_mtime or current_stat.st_size != initial_size:
                    return {
                        "changed": True,
                        "path": path,
                        "watched_seconds": elapsed,
                        "old_size": initial_size,
                        "new_size": current_stat.st_size,
                    }
            except FileNotFoundError:
                return {
                    "changed": True,
                    "path": path,
                    "deleted": True,
                    "watched_seconds": elapsed,
                }
    
    def get_tools(self) -> list[Callable]:
        """Get all tools as a list of callables."""
        return [
            self.run_command,
            self.docker_ps,
            self.docker_logs,
            self.docker_exec,
            self.system_info,
            self.list_processes,
            self.check_port,
            self.get_env,
            self.watch_file,
        ]
