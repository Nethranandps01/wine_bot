"""Test that the Gemini Live API works with gemini-2.0-flash-live-001 + IPv4 patch."""
import asyncio
import socket
import os
from dotenv import load_dotenv

# Force IPv4 (same as main.py)
_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return _orig(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4

load_dotenv(".env")
from google import genai
from google.genai import types as t

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
MODEL = "models/gemini-2.5-flash-native-audio-latest"

async def main():
    client = genai.Client(api_key=api_key, http_options=t.HttpOptions(api_version="v1alpha"))
    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
        },
        "system_instruction": "You are a wine sommelier. Be brief.",
    }
    print(f"Connecting to Live API with {MODEL} ...")
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("✅ Connected! Sending test message...")
            await session.send_client_content(
                turns=t.Content(role="user", parts=[t.Part(text="Name one Burgundy wine in 5 words.")])
            )
            print("Waiting for response...")
            audio_bytes = 0
            text_out = ""
            async with asyncio.timeout(20):
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if not sc:
                        continue
                    mt = getattr(sc, "model_turn", None)
                    if mt and mt.parts:
                        for part in mt.parts:
                            if getattr(part, "text", None):
                                text_out += part.text
                            id_ = getattr(part, "inline_data", None)
                            if id_ and getattr(id_, "data", None):
                                audio_bytes += len(id_.data)
                    if getattr(sc, "turn_complete", False):
                        break
            print(f"✅ Text: {text_out!r}")
            print(f"✅ Audio bytes received: {audio_bytes}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")

asyncio.run(main())
