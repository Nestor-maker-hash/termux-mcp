"""Command-line interface for the Termux tool-using agent."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from .agent import Agent, AgentError
from .mcp_client import HTTPMCPClient, MCPClient
from .providers import OpenAICompatibleProvider, ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="termux-agent",
        description="Run a tool-using AI agent through Termux MCP.",
    )

    parser.add_argument(
        "task",
        nargs="?",
        help="Task for the agent to perform.",
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key. Defaults to TERMUX_AGENT_API_KEY.",
    )

    parser.add_argument(
        "--model",
        default=os.environ.get(
            "TERMUX_AGENT_MODEL",
            "gpt-4o-mini",
        ),
        help="LLM model name.",
    )

    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "TERMUX_AGENT_BASE_URL",
            "https://api.openai.com/v1",
        ),
        help="OpenAI-compatible API base URL.",
    )

    parser.add_argument(
        "--http",
        action="store_true",
        help="Connect to MCP over HTTP instead of stdio.",
    )

    parser.add_argument(
        "--mcp-url",
        default=os.environ.get(
            "TERMUX_AGENT_MCP_URL",
            "http://127.0.0.1:8081/mcp",
        ),
        help="MCP HTTP endpoint.",
    )

    parser.add_argument(
        "--mcp-token",
        default=os.environ.get(
            "TERMUX_AGENT_MCP_TOKEN",
        ),
        help="MCP HTTP bearer token.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum agent tool-loop iterations.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="LLM request timeout in seconds.",
    )

    parser.add_argument(
        "--system-prompt",
        default=os.environ.get(
            "TERMUX_AGENT_SYSTEM_PROMPT",
        ),
        help="Optional system prompt.",
    )

    return parser


def create_client(args: argparse.Namespace):
    if args.http:
        return HTTPMCPClient(
            args.mcp_url,
            auth_token=args.mcp_token,
            timeout=args.timeout,
        )

    return MCPClient()


def run_agent(
    task: str,
    *,
    api_key: str,
    model: str,
    base_url: str,
    http: bool = False,
    mcp_url: str = "http://127.0.0.1:8081/mcp",
    mcp_token: Optional[str] = None,
    max_iterations: int = 20,
    timeout: int = 60,
    system_prompt: Optional[str] = None,
) -> str:
    client = (
        HTTPMCPClient(
            mcp_url,
            auth_token=mcp_token,
            timeout=timeout,
        )
        if http
        else MCPClient()
    )

    provider = OpenAICompatibleProvider(
        api_key,
        model=model,
        base_url=base_url,
        timeout=timeout,
        system_prompt=system_prompt,
    )

    agent = Agent(
        client,
        provider,
        max_iterations=max_iterations,
    )

    try:
        agent.connect()
        return agent.run(task)
    finally:
        agent.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.task or not args.task.strip():
        parser.error("a task is required")

    api_key = args.api_key or os.environ.get("TERMUX_AGENT_API_KEY")

    if not api_key:
        parser.error(
            "API key required; use --api-key or "
            "TERMUX_AGENT_API_KEY"
        )

    try:
        result = run_agent(
            args.task,
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            http=args.http,
            mcp_url=args.mcp_url,
            mcp_token=args.mcp_token,
            max_iterations=args.max_iterations,
            timeout=args.timeout,
            system_prompt=args.system_prompt,
        )
    except (AgentError, ProviderError, RuntimeError, ValueError) as exc:
        print("termux-agent: {}".format(exc), file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
