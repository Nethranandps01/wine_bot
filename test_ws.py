import asyncio
import websockets
import os
import socket
from dotenv import load_dotenv

# Monkey-patch socket to force IPv4
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

async def test():
    uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={api_key}"
    print(f"Connecting to {uri[:60]}...")
    try:
        async with websockets.connect(uri) as ws:
            print("Connected successfully with IPv4 Monkey Patch!")
            await ws.close()
    except Exception as e:
        print(f"IPv4 Error: {type(e).__name__}: {e}")

asyncio.run(test())
