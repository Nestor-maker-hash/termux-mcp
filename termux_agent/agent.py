"""Tool-using agent loop built on top of MCP transports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .mcp_client import MCPTransport


class AgentError(RuntimeError):
    """Raised when the agent cannot complete an operation."""


@dataclass
class ToolCall:
    """A tool invocation requested by the model/provider."""

    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class ProviderResponse:
    """Provider-neutral response from an LLM."""

    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)


class AgentProvider(Protocol):
    """Interface an LLM provider must implement."""

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Sequence[Dict[str, Any]],
    ) -> ProviderResponse:
        ...


class Agent:
    """General-purpose tool-using agent backed by an MCP transport."""

    def __init__(
        self,
        client: MCPTransport,
        provider: AgentProvider,
        *,
        max_iterations: int = 20,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        self.client = client
        self.provider = provider
        self.max_iterations = max_iterations

        self._tools: Optional[List[Dict[str, Any]]] = None
        self.last_run_trace: List[Dict[str, Any]] = []

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Return the discovered MCP tools."""
        if self._tools is None:
            raise AgentError("Agent has not discovered MCP tools yet.")
        return list(self._tools)

    def connect(self) -> Dict[str, Any]:
        """Connect to MCP and discover available tools."""
        result = self.client.connect()
        self._tools = self.client.list_tools()
        return result

    def close(self) -> None:
        """Close the underlying MCP transport."""
        self.client.close()
        self._tools = None

    def run(self, task: str) -> str:
        """Run a task through the provider/tool execution loop."""
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string.")

        self.last_run_trace = []

        if not self.client.running:
            raise AgentError("Agent is not connected.")

        if self._tools is None:
            self._tools = self.client.list_tools()

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": task,
            }
        ]

        for _ in range(self.max_iterations):
            response = self.provider.complete(
                messages,
                self._tools,
            )

            if response.tool_calls:
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [],
                }

                for index, call in enumerate(response.tool_calls):
                    call_id = call.id or "call_{}".format(index + 1)

                    assistant_message["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                    )

                messages.append(assistant_message)

                for index, call in enumerate(response.tool_calls):
                    call_id = call.id or "call_{}".format(index + 1)

                    trace_entry: Dict[str, Any] = {
                        "type": "tool_call",
                        "id": call_id,
                        "name": call.name,
                        "arguments": dict(call.arguments),
                        "success": False,
                    }

                    try:
                        result = self.client.call_tool(
                            call.name,
                            call.arguments,
                        )
                        result_text = self._tool_result_text(result)
                    except Exception as exc:
                        trace_entry["error"] = str(exc)
                        self.last_run_trace.append(trace_entry)
                        raise

                    trace_entry["success"] = True
                    trace_entry["result"] = result_text
                    self.last_run_trace.append(trace_entry)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": call.name,
                            "content": result_text,
                        }
                    )

                continue

            if response.content is not None:
                return response.content

            raise AgentError(
                "Provider returned neither content nor tool calls."
            )

        raise AgentError(
            "Agent exceeded the maximum number of iterations ({}).".format(
                self.max_iterations
            )
        )

    @staticmethod
    def _tool_result_text(result: Dict[str, Any]) -> str:
        """Convert an MCP tool result into text for an LLM tool message."""
        content = result.get("content")

        if isinstance(content, list):
            parts: List[str] = []

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "text":
                    text = item.get("text")
                    if text is not None:
                        parts.append(str(text))

            if parts:
                return "\n".join(parts)

        if "text" in result:
            return str(result["text"])

        return json.dumps(result, ensure_ascii=False)
