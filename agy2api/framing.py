from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional


def encode_http_chunk(payload: bytes) -> bytes:
    """Format payload as a standard HTTP/1.1 chunk: <hex_size>\r\n<payload>\r\n."""
    size_hex = hex(len(payload))[2:].encode("ascii")
    return size_hex + b"\r\n" + payload + b"\r\n"


TERMINAL_CHUNK = b"0\r\n\r\n"


@dataclass
class ServerSentEvent:
    event: str = "message"
    data: str = ""
    id: Optional[str] = None
    retry: Optional[int] = None

    def encode(self) -> bytes:
        """Encode to wire-format SSE lines."""
        lines = []
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.event != "message":
            lines.append(f"event: {self.event}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        for part in self.data.split("\n"):
            lines.append(f"data: {part}")
        return ("\n".join(lines) + "\n\n").encode("utf-8")


class SSEAccumulator:
    """Resilient accumulator for streaming text chunks into ServerSentEvent items."""

    def __init__(self):
        self._buffer: str = ""
        self._current: ServerSentEvent = ServerSentEvent()

    def _parse_line(self, line: str) -> None:
        line = line.strip("\r")
        if not line:
            return
        if line.startswith(":"):
            return
        if ":" in line:
            field, val = line.split(":", 1)
            field = field.strip()
            val = val.lstrip(" ")
            if field == "data":
                self._current.data = (
                    (self._current.data + "\n" + val).lstrip("\n")
                )
            elif field == "event":
                self._current.event = val
            elif field == "id":
                self._current.id = val
            elif field == "retry":
                try:
                    self._current.retry = int(val)
                except ValueError:
                    pass

    def feed(self, text_chunk: str) -> List[ServerSentEvent]:
        events: List[ServerSentEvent] = []
        self._buffer += text_chunk

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip("\r")

            if not line:
                # Dispatch event on blank line
                if self._current.data:
                    events.append(self._current)
                    self._current = ServerSentEvent()
                continue

            self._parse_line(line)

        return events

    def flush(self) -> List[ServerSentEvent]:
        events: List[ServerSentEvent] = []
        if self._buffer.strip():
            self._parse_line(self._buffer)
        if self._current.data:
            events.append(self._current)
            self._current = ServerSentEvent()
        self._buffer = ""
        return events


def format_gemini_sse(payload: dict[str, Any]) -> str:
    """Serialize Gemini candidate chunk to SSE data line."""
    serialized = json.dumps(payload, ensure_ascii=False)
    return f"data: {serialized}\n\n"
