"""Minimal MCP client for communicating with termux-native-mcp over stdio."""

from __future__ import annotations

import json
import subprocess
import threading
import http.client
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class MCPClientError(RuntimeError):
    """Raised when MCP communication fails."""



@runtime_checkable
class MCPTransport(Protocol):
    """Common interface implemented by MCP client transports."""

    @property
    def running(self) -> bool:
        ...

    def connect(self) -> Dict[str, Any]:
        ...

    def list_tools(self) -> List[Dict[str, Any]]:
        ...

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class MCPClient:
    """JSON-RPC MCP client using the server's newline-delimited stdio transport."""

    def __init__(
        self,
        command: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        self.command = command or ["termux-native-mcp", "--stdio"]
        self.cwd = cwd

        self._process: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def connect(self) -> Dict[str, Any]:
        """Start the MCP server and perform the MCP initialize handshake."""
        if self.running:
            raise MCPClientError("Client is already connected.")

        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self._process = None
            raise MCPClientError(
                "Failed to start MCP server: {}".format(exc)
            ) from exc

        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "termux-agent",
                        "version": "0.1.0",
                    },
                },
            )
            self._initialized = True

            self._notify("notifications/initialized", {})

            return result
        except Exception:
            self.close()
            raise

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the tools exposed by the MCP server."""
        self._require_connected()
        result = self._request("tools/list", {})
        return result.get("tools", [])

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call an MCP tool and return its complete MCP result."""
        self._require_connected()

        return self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )

    def close(self) -> None:
        """Close the MCP server process and its pipes."""
        process = self._process
        self._process = None
        self._initialized = False

        if process is None:
            return

        if process.poll() is None:
            try:
                process.stdin.close()
            except Exception:
                pass

            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=2)
                except Exception:
                    pass

        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except Exception:
                pass

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_connected(self) -> None:
        if not self.running or not self._initialized:
            raise MCPClientError("MCP client is not connected.")

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1

            self._write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )

            while True:
                message = self._read_message()

                if message.get("id") != request_id:
                    continue

                if "error" in message:
                    error = message["error"] or {}
                    code = error.get("code", "unknown")
                    text = error.get("message", "Unknown MCP error")
                    raise MCPClientError(
                        "MCP error {}: {}".format(code, text)
                    )

                result = message.get("result")
                if not isinstance(result, dict):
                    raise MCPClientError(
                        "Invalid MCP result for {!r}.".format(method)
                    )

                return result

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def _write_message(self, message: Dict[str, Any]) -> None:
        process = self._process

        if process is None or process.stdin is None:
            raise MCPClientError("MCP server is not running.")

        if process.poll() is not None:
            raise MCPClientError(
                "MCP server exited with code {}.".format(process.returncode)
            )

        try:
            process.stdin.write(
                json.dumps(message, ensure_ascii=False) + "\n"
            )
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPClientError(
                "Failed to write to MCP server: {}".format(exc)
            ) from exc

    def _read_message(self) -> Dict[str, Any]:
        process = self._process

        if process is None or process.stdout is None:
            raise MCPClientError("MCP server is not running.")

        try:
            line = process.stdout.readline()
        except OSError as exc:
            raise MCPClientError(
                "Failed to read from MCP server: {}".format(exc)
            ) from exc

        if not line:
            stderr = ""
            if process.stderr is not None:
                try:
                    stderr = process.stderr.read().strip()
                except Exception:
                    pass

            detail = (
                " MCP stderr: {}".format(stderr)
                if stderr
                else ""
            )

            raise MCPClientError(
                "MCP server closed stdout unexpectedly.{}".format(detail)
            )

        try:
            message = json.loads(line)
        except ValueError as exc:
            raise MCPClientError(
                "MCP server returned invalid JSON: {}".format(line.strip())
            ) from exc

        if not isinstance(message, dict):
            raise MCPClientError("MCP server returned a non-object message.")

        return message


class HTTPMCPClient:
    """MCP client using the server's Streamable HTTP transport."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8081/mcp",
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError("MCP URL must use http:// or https://.")

        if not parsed.hostname:
            raise ValueError("MCP URL must include a hostname.")

        self.url = url.rstrip("/")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.path = parsed.path or "/mcp"
        self.auth_token = auth_token
        self.timeout = timeout

        self._next_id = 1
        self._session_id: Optional[str] = None
        self._initialized = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._session_id is not None and self._initialized

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def connect(self) -> Dict[str, Any]:
        """Perform the MCP initialize handshake."""
        if self.running:
            raise MCPClientError("Client is already connected.")

        result, session_id = self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "termux-agent",
                    "version": "0.1.0",
                },
            },
            include_session=False,
        )

        if not session_id:
            raise MCPClientError(
                "MCP initialize response did not provide a session ID."
            )

        self._session_id = session_id
        self._initialized = True

        try:
            self._notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

        return result

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the tools exposed by the MCP server."""
        self._require_connected()
        result, _ = self._request("tools/list", {})
        return result.get("tools", [])

    def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call an MCP tool."""
        self._require_connected()

        result, _ = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )
        return result

    def close(self) -> None:
        """Delete the MCP HTTP session."""
        session_id = self._session_id

        self._session_id = None
        self._initialized = False

        if not session_id:
            return

        try:
            self._request_raw(
                "DELETE",
                None,
                session_id=session_id,
            )
        except Exception:
            pass

    def _require_connected(self) -> None:
        if not self.running:
            raise MCPClientError("MCP client is not connected.")

    def _headers(self, session_id: Optional[str]) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if session_id:
            headers["Mcp-Session-Id"] = session_id

        if self.auth_token:
            headers["Authorization"] = "Bearer {}".format(
                self.auth_token
            )

        return headers

    def _connection(self):
        if self.scheme == "https":
            return http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=self.timeout,
            )

        return http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=self.timeout,
        )

    def _request(
        self,
        method: str,
        params: Dict[str, Any],
        include_session: bool = True,
    ):
        with self._lock:
            session_id = self._session_id if include_session else None

            request_id = self._next_id
            self._next_id += 1

            body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                ensure_ascii=False,
            ).encode("utf-8")

            status, headers, raw = self._request_raw(
                "POST",
                body,
                session_id=session_id,
            )

            if status != 200:
                raise MCPClientError(
                    "MCP HTTP request failed with status {}: {}".format(
                        status,
                        raw.decode("utf-8", errors="replace"),
                    )
                )

            try:
                message = json.loads(raw.decode("utf-8"))
            except ValueError as exc:
                raise MCPClientError(
                    "MCP HTTP server returned invalid JSON."
                ) from exc

            if not isinstance(message, dict):
                raise MCPClientError(
                    "MCP HTTP server returned a non-object message."
                )

            if "error" in message:
                error = message["error"] or {}
                code = error.get("code", "unknown")
                text = error.get("message", "Unknown MCP error")
                raise MCPClientError(
                    "MCP error {}: {}".format(code, text)
                )

            result = message.get("result")
            if not isinstance(result, dict):
                raise MCPClientError(
                    "Invalid MCP result for {!r}.".format(method)
                )

            response_session = headers.get("Mcp-Session-Id")
            return result, response_session

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            },
            ensure_ascii=False,
        ).encode("utf-8")

        status, _, raw = self._request_raw(
            "POST",
            body,
            session_id=self._session_id,
        )

        if status not in (200, 202):
            raise MCPClientError(
                "MCP notification failed with status {}: {}".format(
                    status,
                    raw.decode("utf-8", errors="replace"),
                )
            )

    def _request_raw(
        self,
        method: str,
        body: Optional[bytes],
        session_id: Optional[str] = None,
    ):
        conn = self._connection()

        try:
            conn.request(
                method,
                self.path,
                body=body,
                headers=self._headers(session_id),
            )

            response = conn.getresponse()
            raw = response.read()

            return (
                response.status,
                {k: v for k, v in response.getheaders()},
                raw,
            )
        except (OSError, http.client.HTTPException) as exc:
            raise MCPClientError(
                "MCP HTTP connection failed: {}".format(exc)
            ) from exc
        finally:
            conn.close()
