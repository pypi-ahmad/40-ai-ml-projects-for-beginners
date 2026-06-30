"""App-level utility tests."""

from __future__ import annotations

import socket

import pytest

import app


def test_resolve_launch_port_skips_busy_port() -> None:
    """Port resolver should move to next port when preferred is occupied."""

    host = "127.0.0.1"
    try:
        busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pytest.skip("Socket creation is blocked in this environment.")

    with busy:
        busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        busy.bind((host, 0))
        busy_port = busy.getsockname()[1]

        resolved = app.resolve_launch_port(host, busy_port, max_attempts=10)

    assert resolved != busy_port
    assert resolved > busy_port
