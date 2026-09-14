"""Agent-side clients and tool-using agent for termux-mcp."""

__version__ = "0.1.0"

from .agent import Agent, AgentError, AgentProvider, ProviderResponse, ToolCall
from .mcp_client import MCPClient, HTTPMCPClient, MCPClientError, MCPTransport
from .providers import OpenAICompatibleProvider, ProviderError

__all__ = [
    "Agent",
    "AgentError",
    "AgentProvider",
    "ProviderResponse",
    "ToolCall",
    "MCPClient",
    "HTTPMCPClient",
    "MCPClientError",
    "MCPTransport",
    "OpenAICompatibleProvider",
    "ProviderError",
]
