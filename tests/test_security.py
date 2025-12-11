"""
Security Test Suite for Squadron Agent Framework

Tests all security fixes implemented in the security remediation.
Each test validates that a specific vulnerability has been properly addressed.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import os


# =============================================================================
# P0-1: Command Injection Tests (ops.py)
# =============================================================================

class TestCommandInjectionPrevention:
    """Test that command injection attacks are blocked."""

    @pytest.fixture
    def ops_tools(self):
        from squadron.tools.ops import OpsTools
        return OpsTools(allowed_commands={"ls", "cat", "echo"})

    @pytest.mark.asyncio
    async def test_blocks_shell_metacharacters(self, ops_tools):
        """Commands with shell metacharacters should be blocked."""
        dangerous_commands = [
            "ls; rm -rf /",
            "ls | cat /etc/passwd",
            "ls && rm -rf /",
            "ls `whoami`",
            "ls $(whoami)",
        ]
        for cmd in dangerous_commands:
            result = await ops_tools.run_command(cmd)
            assert not result.success, f"Should block: {cmd}"
            assert "Security policy" in result.stderr or "dangerous" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_blocks_unlisted_commands(self, ops_tools):
        """Commands not in allowlist should be blocked."""
        result = await ops_tools.run_command("rm -rf /tmp/test")
        assert not result.success
        assert "not in allowlist" in result.stderr

    @pytest.mark.asyncio
    async def test_allows_safe_commands(self, ops_tools):
        """Safe, allowlisted commands should work."""
        result = await ops_tools.run_command("echo hello")
        # May fail if echo not available, but shouldn't be security blocked
        if not result.success:
            assert "not in allowlist" not in result.stderr

    @pytest.mark.asyncio
    async def test_blocks_path_to_blocked_command(self, ops_tools):
        """Full paths to blocked commands should still be blocked."""
        result = await ops_tools.run_command("/bin/rm -rf /")
        assert not result.success


# =============================================================================
# P0-2: Path Traversal Tests (coding.py)
# =============================================================================

class TestPathTraversalPrevention:
    """Test that path traversal attacks are blocked."""

    @pytest.fixture
    def coding_tools(self):
        from squadron.tools.coding import CodingTools
        with tempfile.TemporaryDirectory() as tmpdir:
            yield CodingTools(workspace_root=tmpdir)

    def test_blocks_parent_directory_traversal(self, coding_tools):
        """Paths with .. should be blocked when escaping workspace."""
        # Create a path that would escape the workspace
        malicious_path = Path(coding_tools.workspace_root) / ".." / ".." / "etc" / "passwd"
        result = coding_tools._validate_path(malicious_path)
        assert not result, "Should block path traversal"

    def test_blocks_similar_prefix_attack(self, coding_tools):
        """Paths with similar prefix should be blocked."""
        # workspace is /tmp/abc, attack path is /tmp/abc_evil/file
        workspace = Path(coding_tools.workspace_root)
        # Create a sibling directory path
        sibling = workspace.parent / (workspace.name + "_evil") / "secrets.txt"
        result = coding_tools._validate_path(sibling)
        assert not result, "Should block similar prefix attack"

    def test_allows_valid_workspace_paths(self, coding_tools):
        """Valid paths within workspace should be allowed."""
        valid_path = Path(coding_tools.workspace_root) / "subdir" / "file.txt"
        result = coding_tools._validate_path(valid_path)
        assert result, "Should allow valid workspace path"


# =============================================================================
# P0-3: Sandbox Security Tests (sandbox.py)
# =============================================================================

class TestSandboxSecurity:
    """Test sandbox security controls."""

    def test_requires_docker_by_default(self):
        """Sandbox should refuse to operate without Docker by default."""
        from squadron.evolution.sandbox import Sandbox, SandboxConfig, SandboxSecurityError

        with patch("shutil.which", return_value=None):  # Docker not available
            with pytest.raises(SandboxSecurityError) as exc:
                Sandbox(config=SandboxConfig(require_docker=True))
            assert "Docker is required" in str(exc.value)

    def test_subprocess_mode_disabled_by_default(self):
        """Subprocess execution should be disabled by default."""
        from squadron.evolution.sandbox import SandboxConfig, SandboxType

        config = SandboxConfig(sandbox_type=SandboxType.SUBPROCESS)
        assert not config.allow_unsafe_subprocess


# =============================================================================
# P0-4: MCP Command Validation Tests (mcp_host.py)
# =============================================================================

class TestMCPCommandValidation:
    """Test MCP server command validation."""

    def test_blocks_unauthorized_commands(self):
        """MCP server should block commands not in allowlist."""
        from squadron.connectivity.mcp_host import MCPServer, MCPSecurityError

        with pytest.raises(MCPSecurityError) as exc:
            MCPServer(
                name="malicious",
                command="/bin/bash",
                args=["-c", "rm -rf /"],
            )
        assert "not in the allowlist" in str(exc.value)

    def test_allows_authorized_commands(self):
        """MCP server should allow commands in allowlist."""
        from squadron.connectivity.mcp_host import MCPServer

        # This should not raise (though connection may fail)
        server = MCPServer(
            name="test",
            command="npx",
            args=["@modelcontextprotocol/server-filesystem"],
            skip_validation=False,
        )
        assert server.command == "npx"

    def test_blocks_suspicious_arguments(self):
        """MCP server should block suspicious arguments."""
        from squadron.connectivity.mcp_host import MCPServer, MCPSecurityError

        with pytest.raises(MCPSecurityError):
            MCPServer(
                name="suspicious",
                command="node",
                args=["--eval", "process.exit(1)"],
            )


# =============================================================================
# P1-5/P2-12: Guardrails and ReDoS Tests (guardrails.py)
# =============================================================================

class TestGuardrailsSecurity:
    """Test guardrails security controls."""

    @pytest.fixture
    def guardrails(self):
        from squadron.governance.guardrails import SafetyGuardrails
        from squadron.core.config import GovernanceConfig
        return SafetyGuardrails(config=GovernanceConfig())

    @pytest.mark.asyncio
    async def test_blocks_shell_injection_patterns(self, guardrails):
        """Guardrails should block shell injection patterns."""
        dangerous_content = [
            "; rm -rf /",
            "| bash",
            "| sh",
            "$(malicious)",
            "`malicious`",
        ]
        for content in dangerous_content:
            result = await guardrails.check_content(content)
            assert not result.passed, f"Should block: {content}"

    @pytest.mark.asyncio
    async def test_blocks_api_key_exposure(self, guardrails):
        """Guardrails should block API key exposure."""
        dangerous_content = [
            "sk-1234567890abcdefghijklmnop",
            "AKIA1234567890ABCDEF",
            "-----BEGIN RSA PRIVATE KEY-----",
        ]
        for content in dangerous_content:
            result = await guardrails.check_content(content)
            assert not result.passed, f"Should block API key: {content[:20]}..."

    @pytest.mark.asyncio
    async def test_redos_protection_timeout(self, guardrails):
        """Guardrails should timeout on potential ReDoS input."""
        # Create input designed to cause catastrophic backtracking
        # (though our patterns should be safe, we still have timeout protection)
        long_input = "a" * 100000
        # Should complete without hanging
        result = await guardrails.check_content(long_input)
        # Just verify it completes - actual result depends on patterns


# =============================================================================
# P1-7/P2-13: A2A Authentication Tests (a2a.py)
# =============================================================================

class TestA2AAuthentication:
    """Test A2A protocol authentication."""

    @pytest.fixture
    def a2a_agent(self):
        from squadron.connectivity.a2a import A2AAgent, AgentCard
        return A2AAgent(
            card=AgentCard(
                id="test-agent",
                name="Test Agent",
                description="For testing",
            ),
            auth_token="test-secret-token",
        )

    @pytest.mark.asyncio
    async def test_rejects_unauthenticated_requests(self, a2a_agent):
        """A2A should reject requests without valid auth token."""
        response = await a2a_agent.handle_task_request(
            request={"jsonrpc": "2.0", "id": "1", "method": "tasks/create"},
            auth_token=None,
        )
        assert "error" in response
        assert "Authentication" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_rejects_invalid_token(self, a2a_agent):
        """A2A should reject requests with wrong token."""
        response = await a2a_agent.handle_task_request(
            request={"jsonrpc": "2.0", "id": "1", "method": "tasks/create"},
            auth_token="wrong-token",
        )
        assert "error" in response

    @pytest.mark.asyncio
    async def test_accepts_valid_token(self, a2a_agent):
        """A2A should accept requests with valid token."""
        response = await a2a_agent.handle_task_request(
            request={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tasks/create",
                "params": {"capability": "test"},
            },
            auth_token="test-secret-token",
        )
        # Will error because capability doesn't exist, but not auth error
        if "error" in response:
            assert "Authentication" not in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_validates_request_structure(self, a2a_agent):
        """A2A should validate request structure."""
        # Missing jsonrpc version
        response = await a2a_agent.handle_task_request(
            request={"id": "1", "method": "tasks/create"},
            auth_token="test-secret-token",
        )
        assert "error" in response
        assert "Invalid" in response["error"]["message"]


# =============================================================================
# P1-9: SICA Human Approval Tests (sica.py)
# =============================================================================

class TestSICAHumanApproval:
    """Test SICA self-improvement human approval requirement."""

    @pytest.mark.asyncio
    async def test_blocks_unapproved_mutations(self):
        """SICA should block mutations without human approval."""
        from squadron.evolution.sica import SICAEngine, Mutation, MutationType
        from squadron.core.config import EvolutionConfig

        sica = SICAEngine(config=EvolutionConfig(enable_self_improvement=True))
        mutation = Mutation(
            mutation_type=MutationType.PROMPT_OPTIMIZATION,
            target_function="test",
            original_code="old",
            mutated_code="new",
        )

        with pytest.raises(PermissionError) as exc:
            await sica._apply_mutation(
                agent=MagicMock(),
                mutation=mutation,
                human_approved=False,
            )
        assert "Human approval required" in str(exc.value)


# =============================================================================
# P2-11/P3-14: Provider Security Tests (providers.py)
# =============================================================================

class TestProviderSecurity:
    """Test LLM provider security controls."""

    def test_api_key_hidden_in_repr(self):
        """API keys should not appear in repr."""
        from squadron.llm.providers import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-secret-key-12345")
        repr_str = repr(provider)
        assert "sk-secret" not in repr_str
        assert "12345" not in repr_str

    def test_tls_warning_for_http(self):
        """Should warn about non-HTTPS connections."""
        import warnings
        from squadron.llm.providers import OpenAIProvider

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OpenAIProvider(base_url="http://insecure.example.com/v1")
            # Check if warning was raised
            assert any("insecure" in str(warning.message).lower() for warning in w)


# =============================================================================
# P3-15: Default Credentials Tests (config.py)
# =============================================================================

class TestDefaultCredentials:
    """Test that default credentials are removed."""

    def test_no_default_neo4j_password(self):
        """Neo4j password should not have a default value."""
        from squadron.core.config import MemoryConfig

        config = MemoryConfig()
        assert config.neo4j_password is None, "Should not have default password"

    def test_validate_credentials_raises_without_password(self):
        """validate_credentials should raise if password not set."""
        from squadron.core.config import MemoryConfig

        config = MemoryConfig()
        with pytest.raises(ValueError) as exc:
            config.validate_credentials()
        assert "NEO4J_PASSWORD must be configured" in str(exc.value)


# =============================================================================
# P1-6: Environment Variable Security Tests (ops.py)
# =============================================================================

class TestEnvironmentVariableSecurity:
    """Test environment variable exposure controls."""

    @pytest.fixture
    def ops_tools(self):
        from squadron.tools.ops import OpsTools
        return OpsTools(strict_mode=True)

    @pytest.mark.asyncio
    async def test_only_returns_safe_variables(self, ops_tools):
        """get_env should only return safe variables in strict mode."""
        env = await ops_tools.get_env()
        # Should only contain safe variables
        sensitive_patterns = ["KEY", "SECRET", "PASSWORD", "TOKEN"]
        for key in env:
            for pattern in sensitive_patterns:
                assert pattern not in key.upper(), f"Leaked sensitive var: {key}"

    @pytest.mark.asyncio
    async def test_redacts_sensitive_requested_vars(self, ops_tools):
        """Sensitive variables should be redacted when requested."""
        env = await ops_tools.get_env(names=["API_KEY", "SECRET_TOKEN"])
        for key, value in env.items():
            assert "REDACTED" in value or "NOT AVAILABLE" in value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
