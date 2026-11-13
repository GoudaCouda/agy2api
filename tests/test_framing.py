from agy2api.framing import (
    encode_http_chunk,
    TERMINAL_CHUNK,
    ServerSentEvent,
    SSEAccumulator,
    format_gemini_sse,
)


def test_encode_http_chunk():
    data = b"Hello, world!"
    chunk = encode_http_chunk(data)
    # len("Hello, world!") == 13 == 0xd
    assert chunk.startswith(b"d\r\n")
    assert chunk.endswith(b"\r\n")
    assert b"Hello, world!" in chunk


def test_terminal_chunk():
    assert TERMINAL_CHUNK == b"0\r\n\r\n"


def test_server_sent_event_encode():
    sse = ServerSentEvent(event="message", data='{"text": "hi"}', id="123")
    encoded = sse.encode()
    assert b"id: 123\n" in encoded
    assert b'data: {"text": "hi"}\n\n' in encoded


def test_sse_accumulator_normal_and_split():
    acc = SSEAccumulator()

    # Part 1: Partial line
    events = acc.feed('data: {"choices":')
    assert len(events) == 0

    # Part 2: Complete line and blank separator
    events = acc.feed(' [{"delta": {"content": "Hello"}}]}\n\n')
    assert len(events) == 1
    assert events[0].data == '{"choices": [{"delta": {"content": "Hello"}}]}'

    # Part 3: Multiple events in single chunk
    raw = 'data: {"a": 1}\n\ndata: {"b": 2}\n\n'
    events = acc.feed(raw)
    assert len(events) == 2
    assert events[0].data == '{"a": 1}'
    assert events[1].data == '{"b": 2}'


def test_sse_accumulator_flush():
    acc = SSEAccumulator()
    acc.feed('data: final chunk without trailing blank')
    flushed = acc.flush()
    assert len(flushed) == 1
    assert flushed[0].data == 'final chunk without trailing blank'


def test_format_gemini_sse():
    payload = {"candidates": [{"content": {"parts": [{"text": "response"}]}}]}
    formatted = format_gemini_sse(payload)
    assert formatted.startswith("data: {")
    assert formatted.endswith("\n\n")
    assert '"text": "response"' in formatted
