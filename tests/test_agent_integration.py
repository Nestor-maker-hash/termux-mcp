import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from termux_agent.agent import Agent
from termux_agent.providers import OpenAICompatibleProvider


class FakeMCPClient:
    def __init__(self):
        self.running = False
        self.calls = []

    def connect(self):
        self.running = True
        return {
            "serverInfo": {
                "name": "fake-mcp",
            }
        }

    def list_tools(self):
        return [
            {
                "name": "echo",
                "description": "Echo text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                        }
                    },
                    "required": ["text"],
                },
            }
        ]

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments or {}))

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


class MockLLMProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__(
            "test-key",
            model="mock-model",
            base_url="https://example.com/v1",
        )
        self.requests = []

    def _post(self, path, payload):
        self.requests.append((path, payload))

        if len(self.requests) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": json.dumps(
                                            {
                                                "text": "hello from integration test"
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The MCP tool returned the requested text.",
                    }
                }
            ]
        }


class AgentIntegrationTests(unittest.TestCase):

    def test_provider_agent_mcp_tool_loop(self):
        client = FakeMCPClient()
        provider = MockLLMProvider()

        agent = Agent(
            client,
            provider,
            max_iterations=5,
        )

        agent.connect()

        result = agent.run(
            "Echo hello from the integration test."
        )

        self.assertEqual(
            result,
            "The MCP tool returned the requested text.",
        )

        self.assertEqual(
            client.calls,
            [
                (
                    "echo",
                    {
                        "text": "hello from integration test",
                    },
                )
            ],
        )

        self.assertEqual(len(provider.requests), 2)

        first_path, first_payload = provider.requests[0]

        self.assertEqual(first_path, "/chat/completions")
        self.assertEqual(first_payload["model"], "mock-model")
        self.assertEqual(
            first_payload["messages"],
            [
                {
                    "role": "user",
                    "content": "Echo hello from the integration test.",
                }
            ],
        )

        self.assertEqual(
            first_payload["tools"][0]["function"]["name"],
            "echo",
        )

        _, second_payload = provider.requests[1]

        messages = second_payload["messages"]

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(
            messages[1]["tool_calls"][0]["id"],
            "call_1",
        )
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["name"],
            "echo",
        )

        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["name"], "echo")
        self.assertEqual(
            messages[2]["content"],
            "hello from integration test",
        )


if __name__ == "__main__":
    unittest.main()
