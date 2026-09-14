import json
import unittest

from termux_agent.agent import ProviderResponse, ToolCall
from termux_agent.providers import OpenAICompatibleProvider, ProviderError


class FakeProvider(OpenAICompatibleProvider):
    def __init__(self, response):
        super().__init__(
            "test-key",
            base_url="https://example.com/v1",
        )
        self.response = response
        self.received_payload = None

    def _post(self, path, payload):
        self.received_payload = (path, payload)
        return self.response


class ProviderTests(unittest.TestCase):
    def test_text_response(self):
        provider = FakeProvider({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    }
                }
            ]
        })

        result = provider.complete(
            [{"role": "user", "content": "Hi"}],
            [],
        )

        self.assertIsInstance(result, ProviderResponse)
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.tool_calls, [])

    def test_tool_call(self):
        provider = FakeProvider({
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
                                    "name": "read_project_file",
                                    "arguments": json.dumps({
                                        "path": "src/app.tsx",
                                    }),
                                },
                            }
                        ],
                    }
                }
            ]
        })

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_project_file",
                    "description": "Read a project file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            }
        ]

        result = provider.complete(
            [{"role": "user", "content": "Read src/app.tsx"}],
            tools,
        )

        self.assertIsNone(result.content)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(
            result.tool_calls[0],
            ToolCall(
                name="read_project_file",
                arguments={"path": "src/app.tsx"},
                id="call_1",
            ),
        )

        path, payload = provider.received_payload

        self.assertEqual(path, "/chat/completions")
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["messages"][0]["content"], "Read src/app.tsx")
        self.assertEqual(payload["tools"], tools)

    def test_mcp_tool_is_normalized(self):
        provider = FakeProvider({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        })

        mcp_tools = [
            {
                "name": "inspect_project",
                "description": "Inspect a project.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            }
        ]

        provider.complete(
            [{"role": "user", "content": "Inspect ."}],
            mcp_tools,
        )

        tools = provider.received_payload[1]["tools"]

        self.assertEqual(
            tools,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "inspect_project",
                        "description": "Inspect a project.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                            },
                            "required": ["path"],
                        },
                    },
                }
            ],
        )

    def test_system_prompt(self):
        provider = FakeProvider({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "done",
                    }
                }
            ]
        })
        provider.system_prompt = "You are a Termux developer assistant."

        provider.complete(
            [{"role": "user", "content": "Hello"}],
            [],
        )

        messages = provider.received_payload[1]["messages"]

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"],
            "You are a Termux developer assistant.",
        )

    def test_invalid_tool_arguments(self):
        provider = FakeProvider({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "run",
                                    "arguments": "{not-json",
                                }
                            }
                        ],
                    }
                }
            ]
        })

        with self.assertRaises(ProviderError):
            provider.complete([], [])

    def test_missing_choices(self):
        provider = FakeProvider({})

        with self.assertRaises(ProviderError):
            provider.complete([], [])

    def test_invalid_api_key(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider("")

    def test_invalid_base_url(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider(
                "key",
                base_url="not-a-url",
            )


if __name__ == "__main__":
    unittest.main()
