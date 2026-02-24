import base64
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["USE_DUMMY_GEMINI"] = "true"
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import clean_assistant_text, clean_user_transcript, create_app  # noqa: E402


def test_health_dummy_mode() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["dummy_mode"] is True


def test_stt_dummy_transcript() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)
    response = client.post(
        "/stt",
        files={"file": ("clip.webm", b"dummy-bytes", "audio/webm")},
    )
    assert response.status_code == 200
    text = response.json()["text"]
    assert isinstance(text, str)
    assert len(text) > 5


def test_websocket_text_and_audio_stream() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)

    with client.websocket_connect("/ws/chat?session_id=test-session") as ws:
        first = ws.receive_json()
        assert first["type"] == "session_ready"

        ws.send_json({"type": "user_text", "text": "Pair wine with salmon"})

        got_text = False
        got_audio = False
        got_end = False
        last_text = ""

        for _ in range(500):
            evt = ws.receive_json()
            evt_type = evt.get("type")
            if evt_type == "assistant_text_delta":
                got_text = True
                last_text += evt.get("text", "")
            elif evt_type == "assistant_audio_chunk":
                got_audio = True
                assert "audio_base64" in evt
            elif evt_type == "assistant_turn_end":
                got_end = True
                break

        assert got_text is True
        assert got_audio is True
        assert got_end is True
        lowered = last_text.lower()
        assert any(token in lowered for token in ["wine", "sauvignon", "pinot", "cabernet", "merlot"])


def test_chat_continuity_persists_with_same_session() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)

    with client.websocket_connect("/ws/chat?session_id=continuity") as ws:
        ws.receive_json()  # session_ready
        ws.send_json({"type": "user_text", "text": "I like Pinot Noir"})
        for _ in range(500):
            evt = ws.receive_json()
            if evt.get("type") == "assistant_turn_end":
                break

        ws.send_json({"type": "user_text", "text": "suggest next bottle"})
        final_text = ""
        for _ in range(500):
            evt = ws.receive_json()
            if evt.get("type") == "assistant_text_delta":
                final_text += evt.get("text", "")
            if evt.get("type") == "assistant_turn_end":
                break

        assert "earlier" in final_text.lower() or "continuing" in final_text.lower()


def test_live_audio_stream_path() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)

    with client.websocket_connect("/ws/chat?session_id=audio-live") as ws:
        ws.receive_json()  # session_ready
        ws.send_json({"type": "user_audio_start"})

        audio_bytes = b"\x00\x00" * 2048
        ws.send_json(
            {
                "type": "user_audio_chunk",
                "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                "sample_rate": 16000,
            }
        )
        ws.send_json({"type": "user_audio_end"})

        got_user_text = False
        got_assistant_text = False
        got_assistant_audio = False
        got_end = False

        for _ in range(600):
            evt = ws.receive_json()
            evt_type = evt.get("type")
            if evt_type == "user_transcript_delta":
                got_user_text = True
            elif evt_type == "assistant_text_delta":
                got_assistant_text = True
            elif evt_type == "assistant_audio_chunk":
                got_assistant_audio = True
            elif evt_type == "assistant_turn_end":
                got_end = True
                break

        assert got_user_text is True
        assert got_assistant_text is True
        assert got_assistant_audio is True
        assert got_end is True


def test_reset_clears_continuity() -> None:
    app = create_app(dummy_mode=True)
    client = TestClient(app)

    with client.websocket_connect("/ws/chat?session_id=reset-check") as ws:
        ws.receive_json()  # session_ready
        ws.send_json({"type": "user_text", "text": "I like Pinot Noir"})
        for _ in range(500):
            evt = ws.receive_json()
            if evt.get("type") == "assistant_turn_end":
                break

        ws.send_json({"type": "reset"})
        reset_evt = ws.receive_json()
        assert reset_evt.get("type") == "session_reset"

        ws.send_json({"type": "user_text", "text": "suggest next bottle"})
        final_text = ""
        for _ in range(500):
            evt = ws.receive_json()
            if evt.get("type") == "assistant_text_delta":
                final_text += evt.get("text", "")
            if evt.get("type") == "assistant_turn_end":
                break

        lowered = final_text.lower()
        assert "continuing from your earlier preferences" not in lowered


def test_assistant_meta_cleanup() -> None:
    raw = (
        "**Initiating Wine Conversation** I've processed the user's request. "
        "Hello there! Try a Pinot Noir with mushroom pasta. "
        "Would you like a lighter red option too?"
    )
    cleaned = clean_assistant_text(raw)
    assert "initiating" not in cleaned.lower()
    assert "processed" not in cleaned.lower()
    assert "pinot noir" in cleaned.lower()


def test_user_transcript_enforces_english_bias() -> None:
    non_english = "ഹേ ഹായ് ആം ഐ ബി"
    assert clean_user_transcript(non_english) == ""
