"""LLM provider adapters for the Termux agent."""

from __future__ import annotations

import json
import re
import http.client
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from .agent import AgentProvider, ProviderResponse, ToolCall


class ProviderError(RuntimeError):
    """Raised when an LLM provider request fails."""


class OpenAICompatibleProvider:
    """OpenAI-compatible chat-completions provider.

    The adapter deliberately implements the common OpenAI tool-calling
    interface so it can work with compatible API endpoints as well.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        system_prompt: Optional[str] = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")

        if timeout < 1:
            raise ValueError("timeout must be at least 1.")

        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL.")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        tools: Sequence[Dict[str, Any]],
    ) -> ProviderResponse:
        """Send one model request and normalize its tool calls."""
        payload_messages: List[Dict[str, Any]] = []

        if self.system_prompt:
            payload_messages.append(
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            )

        payload_messages.extend(dict(message) for message in messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
        }

        if tools:
            payload["tools"] = [
                self._normalize_tool(tool)
                for tool in tools
            ]

        response = self._post("/chat/completions", payload)

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderError("Provider response contains no choices.")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ProviderError("Provider response contains no message.")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = str(content)

        tool_calls: List[ToolCall] = []

        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ProviderError("Invalid tool_calls in provider response.")

        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue

            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if not isinstance(name, str) or not name:
                raise ProviderError("Provider returned a tool call without a name.")

            raw_arguments = function.get("arguments", "{}")

            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    repaired = re.sub(
                        r",\\s*([}\\]])",
                        r"\\1",
                        raw_arguments,
                    )
                    if repaired == raw_arguments:
                        raise ProviderError(
                            "Provider returned invalid JSON tool arguments "
                            "for {!r}: {}".format(name, exc)
                        ) from exc
                    try:
                        arguments = json.loads(repaired)
                    except json.JSONDecodeError:
                        raise ProviderError(
                            "Provider returned invalid JSON tool arguments "
                            "for {!r}: {}".format(name, exc)
                        ) from exc
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                raise ProviderError(
                    "Provider returned invalid arguments for {!r}.".format(name)
                )

            if not isinstance(arguments, dict):
                raise ProviderError(
                    "Tool arguments for {!r} must be an object.".format(name)
                )

            call_id = raw_call.get("id")
            if call_id is not None and not isinstance(call_id, str):
                call_id = str(call_id)

            metadata = raw_call.get("extra_content")
            if metadata is not None and not isinstance(metadata, dict):
                metadata = None

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    id=call_id,
                    metadata=metadata,
                )
            )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _normalize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an MCP tool or OpenAI tool to OpenAI function format."""
        if not isinstance(tool, dict):
            raise ProviderError("Tool definition must be an object.")

        # Already normalized for OpenAI-compatible APIs.
        if (
            tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
        ):
            return dict(tool)

        name = tool.get("name")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema")

        if not isinstance(name, str) or not name:
            raise ProviderError("MCP tool is missing a valid name.")

        if not isinstance(description, str):
            description = str(description)

        if not isinstance(input_schema, dict):
            raise ProviderError(
                "MCP tool {!r} is missing a valid inputSchema.".format(name)
            )

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema,
            },
        }

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        parsed = urlparse(self.base_url)

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Authorization": "Bearer {}".format(self.api_key),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        connection_class = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )

        connection = connection_class(
            parsed.netloc,
            timeout=self.timeout,
        )

        request_path = (parsed.path.rstrip("/") or "") + path

        try:
            connection.request(
                "POST",
                request_path,
                body=body,
                headers=headers,
            )

            response = connection.getresponse()
            raw_body = response.read()
        except (OSError, http.client.HTTPException) as exc:
            raise ProviderError(
                "Provider request failed: {}".format(exc)
            ) from exc
        finally:
            connection.close()

        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(
                "Provider returned invalid JSON (HTTP {}).".format(
                    response.status
                )
            ) from exc

        if response.status < 200 or response.status >= 300:
            error = decoded.get("error") if isinstance(decoded, dict) else None

            if isinstance(error, dict):
                message = error.get("message") or str(error)
            else:
                message = str(decoded)

            raise ProviderError(
                "Provider returned HTTP {}: {}".format(
                    response.status,
                    message,
                )
            )

        if not isinstance(decoded, dict):
            raise ProviderError("Provider response must be a JSON object.")

        return decoded


__all__ = [
    "ProviderError",
    "OpenAICompatibleProvider",
]
