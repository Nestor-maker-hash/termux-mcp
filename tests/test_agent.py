import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from termux_agent.agent import (
    Agent,
    AgentError,
    ProviderResponse,
    ToolCall,
)


class FakeMCPClient:
    def __init__(self):
        self.running = False
        self.connect_calls = 0
        self.list_tools_calls = 0
        self.tool_calls = []

    def connect(self):
        self.connect_calls += 1
        self.running = True
        return {
            "serverInfo": {
                "name": "fake-mcp",
            }
        }

    def list_tools(self):
        self.list_tools_calls += 1
        return [
            {
                "name": "echo",
                "description": "Echo text.",
                "inputSchema": {
                    "type": "object",
                },
            }
        ]

    def call_tool(self, name, arguments=None):
        self.tool_calls.append((name, arguments or {}))
        return {
            "content": [
                {
                    "type": "text",
                    "text": arguments.get("text", ""),
                }
            ]
        }

    def close(self):
        self.running = False


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
            }
        )

        if not self.responses:
            raise AssertionError("Provider ran out of responses.")

        return self.responses.pop(0)


class AgentTests(unittest.TestCase):

    def test_connect_discovers_tools(self):
        client = FakeMCPClient()
        provider = FakeProvider([])

        agent = Agent(client, provider)

        result = agent.connect()

        self.assertEqual(result["serverInfo"]["name"], "fake-mcp")
        self.assertEqual(client.connect_calls, 1)
        self.assertEqual(client.list_tools_calls, 1)
        self.assertEqual(agent.tools[0]["name"], "echo")

    def test_run_returns_provider_content(self):
        client = FakeMCPClient()
        provider = FakeProvider(
            [
                ProviderResponse(content="Done."),
            ]
        )

        agent = Agent(client, provider)
        agent.connect()

        result = agent.run("Say hello.")

        self.assertEqual(result, "Done.")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            provider.calls[0]["messages"][0]["content"],
            "Say hello.",
        )

    def test_run_executes_tool_then_returns_final_response(self):
        client = FakeMCPClient()
        provider = FakeProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall(
                            name="echo",
                            arguments={"text": "hello"},
                        )
                    ]
                ),
                ProviderResponse(content="The tool returned hello."),
            ]
        )

        agent = Agent(client, provider)
        agent.connect()

        result = agent.run("Echo hello.")

        self.assertEqual(result, "The tool returned hello.")
        self.assertEqual(
            client.tool_calls,
            [
                (
                    "echo",
                    {
                        "text": "hello",
                    },
                )
            ],
        )
        self.assertEqual(len(provider.calls), 2)

        second_messages = provider.calls[1]["messages"]

        self.assertEqual(second_messages[0]["role"], "user")
        self.assertEqual(second_messages[1]["role"], "assistant")
        self.assertEqual(
            second_messages[1]["tool_calls"][0]["function"]["name"],
            "echo",
        )

        # The tool result follows the assistant tool call.
        self.assertEqual(second_messages[2]["role"], "tool")
        self.assertEqual(second_messages[2]["tool_call_id"], "call_1")
        self.assertEqual(second_messages[2]["name"], "echo")
        self.assertEqual(second_messages[2]["content"], "hello")

    def test_run_can_execute_multiple_tools_in_one_response(self):
        client = FakeMCPClient()
        provider = FakeProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall("echo", {"text": "one"}),
                        ToolCall("echo", {"text": "two"}),
                    ]
                ),
                ProviderResponse(content="Finished."),
            ]
        )

        agent = Agent(client, provider)
        agent.connect()

        result = agent.run("Do both.")

        self.assertEqual(result, "Finished.")
        self.assertEqual(
            client.tool_calls,
            [
                ("echo", {"text": "one"}),
                ("echo", {"text": "two"}),
            ],
        )

    def test_run_requires_connection(self):
        client = FakeMCPClient()
        provider = FakeProvider([])

        agent = Agent(client, provider)

        with self.assertRaises(AgentError):
            agent.run("Do something.")

    def test_run_rejects_empty_task(self):
        client = FakeMCPClient()
        provider = FakeProvider([])

        agent = Agent(client, provider)

        with self.assertRaises(ValueError):
            agent.run("")

    def test_provider_empty_response_is_error(self):
        client = FakeMCPClient()
        provider = FakeProvider(
            [
                ProviderResponse(),
            ]
        )

        agent = Agent(client, provider)
        agent.connect()

        with self.assertRaises(AgentError):
            agent.run("Do something.")

    def test_iteration_limit(self):
        client = FakeMCPClient()
        provider = FakeProvider(
            [
                ProviderResponse(
                    tool_calls=[
                        ToolCall("echo", {"text": "loop"})
                    ]
                ),
                ProviderResponse(
                    tool_calls=[
                        ToolCall("echo", {"text": "loop"})
                    ]
                ),
            ]
        )

        agent = Agent(
            client,
            provider,
            max_iterations=2,
        )
        agent.connect()

        with self.assertRaises(AgentError):
            agent.run("Loop.")

        self.assertEqual(len(client.tool_calls), 2)


if __name__ == "__main__":
    unittest.main()
