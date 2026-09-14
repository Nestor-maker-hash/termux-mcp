import json
import os
import sys
import unittest
from pathlib import Path
import threading


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from termux_agent.mcp_client import (
    HTTPMCPClient,
    MCPClient,
    MCPClientError,
    MCPTransport,
)


class MCPClientTests(unittest.TestCase):

    def test_connect_and_initialize(self):
        with MCPClient(
            command=[
                sys.executable,
                "-m",
                "termux_mcp.mcp_main",
                "--stdio",
            ],
            cwd=str(ROOT),
        ) as client:
            self.assertTrue(client.running)

            tools = client.list_tools()
            names = {tool["name"] for tool in tools}

            self.assertIn("run", names)
            self.assertIn("ls", names)
            self.assertIn("inspect_project", names)
            self.assertIn("edit_project", names)

    def test_call_tool(self):
        with MCPClient(
            command=[
                sys.executable,
                "-m",
                "termux_mcp.mcp_main",
                "--stdio",
            ],
            cwd=str(ROOT),
        ) as client:
            result = client.call_tool("cancel", {})

            self.assertIn("content", result)
            self.assertEqual(
                result["content"][0]["text"],
                "Nothing running.",
            )

    def test_tool_error_becomes_client_error(self):
        with MCPClient(
            command=[
                sys.executable,
                "-m",
                "termux_mcp.mcp_main",
                "--stdio",
            ],
            cwd=str(ROOT),
        ) as client:
            with self.assertRaises(MCPClientError):
                client.call_tool("does_not_exist", {})

    def test_transports_share_interface(self):
        clients = [
            MCPClient(),
            HTTPMCPClient(
                "http://127.0.0.1:{}/mcp".format(
                    HTTPMCPClientTests.port
                )
            ),
        ]

        for client in clients:
            self.assertIsInstance(client, MCPTransport)

    def test_not_connected(self):
        client = MCPClient()

        with self.assertRaises(MCPClientError):
            client.list_tools()

        with self.assertRaises(MCPClientError):
            client.call_tool("cancel", {})


class HTTPMCPClientTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from termux_mcp.mcp_server import MCPHTTPServer
        from termux_mcp.mcp_transport_http import MCPRequestHandler

        cls.server = MCPHTTPServer(
            ("127.0.0.1", 0),
            MCPRequestHandler,
        )
        cls.port = cls.server.server_address[1]

        cls.thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_connect_and_list_tools(self):
        client = HTTPMCPClient(
            "http://127.0.0.1:{}/mcp".format(self.port)
        )

        try:
            result = client.connect()

            self.assertEqual(
                result["serverInfo"]["name"],
                "termux-native-mcp",
            )
            self.assertTrue(client.session_id)

            tools = client.list_tools()
            names = {tool["name"] for tool in tools}

            self.assertIn("run", names)
            self.assertIn("inspect_project", names)
            self.assertIn("edit_project", names)
        finally:
            client.close()

    def test_call_tool(self):
        client = HTTPMCPClient(
            "http://127.0.0.1:{}/mcp".format(self.port)
        )

        try:
            client.connect()
            result = client.call_tool("cancel", {})

            self.assertEqual(
                result["content"][0]["text"],
                "Nothing running.",
            )
        finally:
            client.close()

    def test_not_connected(self):
        client = HTTPMCPClient(
            "http://127.0.0.1:{}/mcp".format(self.port)
        )

        with self.assertRaises(MCPClientError):
            client.list_tools()

        with self.assertRaises(MCPClientError):
            client.call_tool("cancel", {})


if __name__ == "__main__":
    unittest.main()
