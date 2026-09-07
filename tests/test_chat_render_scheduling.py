"""Exercise the actual chat methods with the shipped Alpine reactivity engine."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_streaming_render_scheduler():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for Alpine runtime regression tests")
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, str(root / "tests/js/chat_render_scheduling.cjs")],
        cwd=root, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_final_message_uses_full_received_content_before_timer_cleanup():
    root = Path(__file__).resolve().parents[1]
    html = (root / "omlx/admin/templates/chat.html").read_text()
    stream = html.split("async streamResponse(streamContext = null, depth = 0)", 1)[1]
    stream = stream.split("    stopStreaming()", 1)[0]
    assert "stream.streamingContent" in stream
    assert stream.index("this.saveCurrentChat(context.chatId, chatSession.messages") < stream.rindex(
        "this.resetStreamSession(stream, { preserveFinalContent: true })"
    )
