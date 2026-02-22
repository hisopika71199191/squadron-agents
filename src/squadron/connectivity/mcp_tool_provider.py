"""
MCP Tool Provider

Bridges MCP servers to the Squadron Agent tool system. Provides seamless
integration so that MCP-discovered tools can be used exactly like predefined
tool packs (CodingTools, PresentationTools, etc.).

The MCPToolProvider connects to one or more MCP servers, discovers their
tools, and exposes them as:
- Callable async functions (for Agent._tool_registry)
- ToolDefinition objects (for LATS/LLM action generation)
- OpenAI/Anthropic formatted tool descriptions (for LLM prompts)

Example:
    ```python
    from squadron import Agent, LATSReasoner
    from squadron.connectivity import MCPToolProvider

    # From a config file
    mcp_tools = MCPToolProvider(config_path="mcp_servers.json")
    await mcp_tools.connect()

    # Or from an existing MCPHost
    mcp_tools = MCPToolProvider(mcp_host=existing_host)

    # Use as tool source for Agent (just like PresentationTools)
    agent = Agent(name="dev", tools=mcp_tools, llm=llm, reasoner=reasoner)
    result = await agent.run("Search for recent AI papers")

    # Clean up
    await mcp_tools.close()
    ```
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from squadron.connectivity.mcp_host import MCPHost, MCPServer, MCPTool, MCPTransport
from squadron.connectivity.mcp_client import MCPClient, MCPClientConfig
from squadron.llm.base import ToolDefinition

logger = structlog.get_logger(__name__)


@dataclass
class MCPToolWrapper:
    """
    Wraps an MCP tool as a callable async function.

    This wrapper is what gets registered in Agent._tool_registry.
    When called, it routes the execution through the MCPHost or MCPClient.
    """
    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str
    _host: MCPHost | None = None
    _client: MCPClient | None = None

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the MCP tool with the given arguments."""
        if self._host:
            result = await self._host.call_tool(self.name, kwargs)
            # MCP results often come as {"content": [...]} format
            if isinstance(result, dict):
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    # Extract text from content blocks
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                    if texts:
                        return "\n".join(texts)
                return json.dumps(result) if result else "Tool executed successfully"
            return str(result) if result else "Tool executed successfully"
        elif self._client:
            result = await self._client.call_tool(self.name, kwargs)
            if isinstance(result, dict):
                content = result.get("content", [])
                if isinstance(content, list) and content:
                    texts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            texts.append(block["text"])
                    if texts:
                        return "\n".join(texts)
                return json.dumps(result) if result else "Tool executed successfully"
            return str(result) if result else "Tool executed successfully"
        else:
            raise RuntimeError(f"No MCP host or client configured for tool '{self.name}'")

    def to_tool_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM/LATS integration."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema or {
                "type": "object",
                "properties": {},
            },
        )


class MCPToolProvider:
    """
    MCP Tool Provider — use MCP servers as a tool source for Squadron Agents.

    This class acts as a bridge between MCP (Model Context Protocol) servers
    and the Squadron Agent tool system. It discovers tools from MCP servers
    and exposes them in a format compatible with Agent and LATS.

    Supports three modes of operation:

    1. **Config file** — Load MCP servers from a JSON config file:
       ```python
       provider = MCPToolProvider(config_path="mcp_servers.json")
       await provider.connect()
       ```

    2. **Existing MCPHost** — Use an already-connected MCPHost:
       ```python
       provider = MCPToolProvider(mcp_host=host)
       ```

    3. **Inline server definitions** — Add servers programmatically:
       ```python
       provider = MCPToolProvider()
       await provider.add_server("fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
       ```

    The provider implements `get_tools()`, matching the tool pack interface
    (like CodingTools, PresentationTools), so it can be passed directly to
    Agent(tools=...).

    Example:
        ```python
        from squadron import Agent, LATSReasoner
        from squadron.connectivity import MCPToolProvider
        from squadron.llm import OpenAICompatibleProvider

        llm = OpenAICompatibleProvider(...)

        # Connect to MCP servers
        mcp_tools = MCPToolProvider(config_path="mcp_servers.json")
        await mcp_tools.connect()

        # Create agent with MCP tools (just like using PresentationTools)
        reasoner = LATSReasoner()
        agent = Agent(name="assistant", tools=mcp_tools, llm=llm, reasoner=reasoner)
        result = await agent.run("Search for Python tutorials")

        # Clean up
        await mcp_tools.close()
        ```
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        mcp_host: MCPHost | None = None,
        mcp_client: MCPClient | None = None,
    ):
        """
        Initialize the MCP Tool Provider.

        Args:
            config_path: Path to MCP servers JSON config file
            mcp_host: Existing MCPHost instance (already connected or will be connected)
            mcp_client: Existing MCPClient instance (for remote MCP servers)
        """
        self._config_path = Path(config_path) if config_path else None
        self._host = mcp_host
        self._client = mcp_client
        self._owns_host = False  # Whether we created the host (and need to close it)
        self._tool_wrappers: list[MCPToolWrapper] = []
        self._tool_definitions: list[ToolDefinition] = []
        self._connected = False

    async def connect(self) -> None:
        """
        Connect to MCP servers and discover tools.

        If a config_path was provided, creates an MCPHost and loads servers.
        If an mcp_host was provided, discovers tools from it.
        If an mcp_client was provided, discovers tools from it.
        """
        if self._connected:
            return

        # Create MCPHost from config if needed
        if self._config_path and not self._host:
            self._host = MCPHost()
            self._owns_host = True
            await self._host.load_servers(str(self._config_path))

        # Connect client if provided
        if self._client:
            await self._client.connect()

        # Discover and wrap tools
        await self._discover_tools()
        self._connected = True

        logger.info(
            "MCPToolProvider connected",
            tools=len(self._tool_wrappers),
            tool_names=[t.name for t in self._tool_wrappers],
        )

    async def add_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: str = "stdio",
        url: str | None = None,
    ) -> int:
        """
        Add and connect to an MCP server, then discover its tools.

        Args:
            name: Unique server identifier
            command: Server command (e.g. "npx", "python")
            args: Command arguments
            env: Environment variables
            transport: Transport type ("stdio", "sse", "http")
            url: Server URL (for SSE/HTTP transport)

        Returns:
            Number of tools discovered from this server
        """
        if not self._host:
            self._host = MCPHost()
            self._owns_host = True

        transport_enum = MCPTransport(transport)

        await self._host.add_server(
            name=name,
            command=command,
            args=args,
            env=env,
            transport=transport_enum,
            url=url,
        )

        # Re-discover all tools
        old_count = len(self._tool_wrappers)
        await self._discover_tools()
        new_tools = len(self._tool_wrappers) - old_count

        self._connected = True
        logger.info("Added MCP server", name=name, new_tools=new_tools)
        return new_tools

    async def add_remote_server(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> int:
        """
        Add a remote MCP server via HTTP/SSE.

        Args:
            base_url: Server base URL
            api_key: Optional API key
            timeout: Request timeout in seconds

        Returns:
            Number of tools discovered
        """
        self._client = MCPClient(
            config=MCPClientConfig(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        )
        await self._client.connect()

        old_count = len(self._tool_wrappers)
        await self._discover_tools()
        new_tools = len(self._tool_wrappers) - old_count

        self._connected = True
        logger.info("Added remote MCP server", url=base_url, new_tools=new_tools)
        return new_tools

    async def _discover_tools(self) -> None:
        """Discover tools from all connected MCP sources."""
        self._tool_wrappers.clear()
        self._tool_definitions.clear()

        # Discover from MCPHost
        if self._host:
            for mcp_tool in self._host.get_all_tools():
                wrapper = MCPToolWrapper(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    input_schema=mcp_tool.input_schema,
                    server_name=mcp_tool.server_name,
                    _host=self._host,
                )
                self._tool_wrappers.append(wrapper)
                self._tool_definitions.append(wrapper.to_tool_definition())

        # Discover from MCPClient
        if self._client:
            for mcp_tool in self._client.tools:
                wrapper = MCPToolWrapper(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    input_schema=mcp_tool.input_schema,
                    server_name=mcp_tool.server_name,
                    _client=self._client,
                )
                self._tool_wrappers.append(wrapper)
                self._tool_definitions.append(wrapper.to_tool_definition())

    def get_tools(self) -> list[MCPToolWrapper]:
        """
        Get all discovered MCP tools as callables.

        This method matches the tool pack interface (CodingTools, PresentationTools, etc.)
        so MCPToolProvider can be passed directly to Agent(tools=...).

        Returns:
            List of MCPToolWrapper callable objects
        """
        return list(self._tool_wrappers)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """
        Get all tool definitions for LLM/LATS integration.

        Returns ToolDefinition objects that describe each tool's name,
        description, and parameter schema — used by LATS to generate
        candidate tool calls.

        Returns:
            List of ToolDefinition objects
        """
        return list(self._tool_definitions)

    def get_tool(self, name: str) -> MCPToolWrapper | None:
        """Get a specific tool wrapper by name."""
        for wrapper in self._tool_wrappers:
            if wrapper.name == name:
                return wrapper
        return None

    @property
    def tool_count(self) -> int:
        """Number of discovered tools."""
        return len(self._tool_wrappers)

    @property
    def tool_names(self) -> list[str]:
        """Names of all discovered tools."""
        return [t.name for t in self._tool_wrappers]

    @property
    def is_connected(self) -> bool:
        """Whether the provider has been connected."""
        return self._connected

    async def close(self) -> None:
        """Close all MCP connections."""
        if self._owns_host and self._host:
            await self._host.close()
        if self._client:
            await self._client.disconnect()
        self._connected = False
        self._tool_wrappers.clear()
        self._tool_definitions.clear()
        logger.info("MCPToolProvider closed")

    async def __aenter__(self) -> MCPToolProvider:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"MCPToolProvider(tools={len(self._tool_wrappers)}, "
            f"connected={self._connected})"
        )
