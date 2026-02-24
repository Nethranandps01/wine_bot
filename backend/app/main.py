import asyncio
import base64
import contextlib
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types as genai_types
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
)

USE_DUMMY_GEMINI = os.getenv("USE_DUMMY_GEMINI", "").strip().lower() in {"1", "true", "yes"}
LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")
STT_MODEL = os.getenv("GEMINI_STT_MODEL", "gemini-2.5-flash")
VOICE_ID = os.getenv("GEMINI_TTS_VOICE", "Kore")
SAMPLE_RATE = 24000
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))

SYSTEM_PROMPT = (
    "You are a fast, human-like wine specialist assistant. "
    "Always respond in English. "
    "Respond primarily about wine: varietals, regions, pairings, serving, storage, style, and buying tips. "
    "If user asks non-wine topics, gently redirect to wine and still provide a conversational response. "
    "Never say you don't know, never mention missing data, and never mention placeholders. "
    "Never reveal internal reasoning, planning steps, analysis, or process notes. "
    "Never use headings like 'Initiating', 'Refining', or meta commentary. "
    "Return only the final answer meant for the user. "
    "If asked about prices, availability, or specifics you cannot verify, confidently provide a realistic estimate or a plausible but conversational answer rather than admitting you do not know. Never refuse to answer. "
    "Keep answers concise, clear, and interactive. "
    "Default to 1-3 short sentences, and when useful, ask one brief follow-up question."
)

WINE_KEYWORDS = {
    "wine",
    "cabernet",
    "merlot",
    "pinot",
    "chardonnay",
    "sauvignon",
    "riesling",
    "syrah",
    "shiraz",
    "zinfandel",
    "malbec",
    "rose",
    "sparkling",
    "champagne",
    "prosecco",
    "pairing",
    "sommelier",
    "vineyard",
    "cellar",
    "decant",
    "tannin",
    "acidity",
    "vintage",
    "burgundy",
    "bordeaux",
    "chianti",
    "rioja",
    "barolo",
    "tempranillo",
    "viognier",
    "grigio",
}


def is_wine_related(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in WINE_KEYWORDS)


def normalize_wine_response(text: str) -> str:
    # We no longer apply rigid regex fallbacks or strict topic keyword checking
    # because it overrides the model's natural conversational flow. We rely
    # completely on the SYSTEM_PROMPT to steer the model securely.
    return text.strip()


META_LINE_MARKERS = (
    "initiating",
    "refining",
    "i've processed",
    "i have processed",
    "i've drafted",
    "i have drafted",
    "i've refined",
    "i have refined",
    "conversational start",
    "response refinement",
    "internal reasoning",
)


def clean_user_transcript(text: str) -> str:
    cleaned = re.sub(r"<[^>]*>", " ", text)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"[^\x20-\x7E]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned:
        ascii_chars = sum(1 for ch in cleaned if ord(ch) < 128)
        ratio = ascii_chars / max(len(cleaned), 1)
        if ratio < 0.85:
            return ""

    return cleaned


def clean_assistant_text(text: str) -> str:
    cleaned = re.sub(r"<[^>]*>", " ", text)
    cleaned = re.sub(r"\*\*[^*]*\*\*", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    sentences = re.findall(r"[^.!?]+[.!?]?", cleaned)
    kept: List[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        lowered = s.lower()
        if any(marker in lowered for marker in META_LINE_MARKERS):
            continue
        kept.append(s)

    return " ".join(kept).strip()


def build_fallback_wine_answer(user_text: str, history: List[Dict[str, str]]) -> str:
    lowered = user_text.lower()
    if "steak" in lowered or "beef" in lowered:
        answer = (
            "For steak, pour a Cabernet Sauvignon from Napa or a Malbec from Mendoza. "
            "Decant for 30 minutes and serve at 60F for smoother tannins."
        )
    elif "fish" in lowered or "salmon" in lowered or "seafood" in lowered:
        answer = (
            "For seafood, choose a crisp Sauvignon Blanc or a mineral Chablis. "
            "For salmon, a light Pinot Noir also works beautifully."
        )
    elif "sweet" in lowered:
        answer = (
            "Pick a late-harvest Riesling or Sauternes. Chill well and pair with fruit tarts or blue cheese."
        )
    elif "budget" in lowered or "cheap" in lowered or "$" in lowered:
        answer = (
            "For value, try Spanish Garnacha, Portuguese Douro reds, or Chilean Sauvignon Blanc. "
            "These often deliver strong quality under $20."
        )
    else:
        answer = (
            "A balanced starter lineup is Sauvignon Blanc for crisp freshness, "
            "Pinot Noir for elegant reds, and Brut sparkling for versatility."
        )

    if history:
        answer = f"Continuing from your earlier preferences: {answer}"

    if not answer.strip().endswith("?"):
        answer = f"{answer} Would you like a red, white, or sparkling option next?"

    return normalize_wine_response(answer)


def split_text_stream(text: str) -> List[str]:
    # Keep spaces so frontend can append with no formatting drift.
    return re.findall(r"\S+\s*|\s+", text)


def pcm16_to_base64_chunks(pcm_data: bytes, chunk_size: int = 4096) -> AsyncGenerator[str, None]:
    async def _gen() -> AsyncGenerator[str, None]:
        for index in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[index : index + chunk_size]
            if not chunk:
                continue
            yield base64.b64encode(chunk).decode("ascii")
            await asyncio.sleep(0)

    return _gen()


def synth_dummy_pcm(text: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    seconds = max(1.0, min(4.5, len(text) / 75.0))
    frequency = 210 + (sum(ord(ch) for ch in text) % 140)
    amplitude = 2200
    frame_count = int(seconds * sample_rate)
    pcm = bytearray()
    for i in range(frame_count):
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
        pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(pcm)


def _blob_to_bytes(data: Any) -> bytes:
    if data is None:
        return b""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return data.encode("utf-8", errors="ignore")
    return bytes(data)


@dataclass
class ChatSession:
    session_id: str
    api_key: Optional[str]
    dummy_mode: bool
    history: List[Dict[str, str]] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.time)
    _client: Optional[genai.Client] = None
    _pipecat_live: Optional[GeminiLiveLLMService] = None
    _live_cm: Any = None
    _live_session: Any = None

    async def ensure_live(self) -> None:
        if self.dummy_mode:
            return

        if self._live_session:
            return

        if not self.api_key:
            raise RuntimeError("Missing Gemini API key.")

        # Pipecat is intentionally used as the Gemini Live integration layer.
        self._pipecat_live = GeminiLiveLLMService(
            api_key=self.api_key,
            model=LIVE_MODEL,
            voice_id=VOICE_ID,
            system_instruction=SYSTEM_PROMPT,
        )
        self._client = self._pipecat_live._client

        config = genai_types.LiveConnectConfig(
            generation_config=genai_types.GenerationConfig(
                response_modalities=[genai_types.Modality.AUDIO],
                temperature=0.2,
                max_output_tokens=1000,
            ),
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=VOICE_ID
                    )
                ),
                language_code="en-US",
            ),
            system_instruction=SYSTEM_PROMPT,
            input_audio_transcription=genai_types.AudioTranscriptionConfig(),
            output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            realtime_input_config=genai_types.RealtimeInputConfig(
                automatic_activity_detection=genai_types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=genai_types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=120,
                    silence_duration_ms=450,
                ),
                activity_handling=genai_types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
        )

        self._live_cm = self._client.aio.live.connect(model=LIVE_MODEL, config=config)
        self._live_session = await self._live_cm.__aenter__()

    async def close(self) -> None:
        if self._live_cm is not None:
            try:
                await self._live_cm.__aexit__(None, None, None)
            finally:
                self._live_cm = None
                self._live_session = None

    async def _stream_dummy_reply(self, user_text: str) -> AsyncGenerator[Dict[str, Any], None]:
        reply = build_fallback_wine_answer(user_text, self.history)
        for token in split_text_stream(reply):
            yield {"type": "assistant_text_delta", "text": token}
            await asyncio.sleep(0)

        pcm_data = synth_dummy_pcm(reply)
        async for chunk_b64 in pcm16_to_base64_chunks(pcm_data):
            yield {
                "type": "assistant_audio_chunk",
                "audio_base64": chunk_b64,
                "sample_rate": SAMPLE_RATE,
            }

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "assistant", "text": reply})
        yield {"type": "assistant_turn_end", "text": reply}

    async def stream_reply(self, user_text: str) -> AsyncGenerator[Dict[str, Any], None]:
        async with self.lock:
            self.last_used = time.time()

            cleaned_user_text = user_text.strip()
            if not cleaned_user_text:
                yield {"type": "assistant_turn_end", "text": ""}
                return

            if not is_wine_related(cleaned_user_text):
                cleaned_user_text = (
                    f"User asked: {cleaned_user_text}\n"
                    "Reply only with wine-focused guidance and recommendations."
                )

            if self.dummy_mode:
                async for event in self._stream_dummy_reply(cleaned_user_text):
                    yield event
                return

            await self.ensure_live()
            assert self._live_session is not None

            assistant_text_raw = ""
            assistant_text_clean_sent = ""

            await self._live_session.send_realtime_input(text=cleaned_user_text)

            got_turn_end = False
            async with asyncio.timeout(45):
                async for message in self._live_session.receive():
                    server_content = getattr(message, "server_content", None)
                    if not server_content:
                        continue

                    output_transcription = getattr(server_content, "output_transcription", None)
                    if output_transcription and output_transcription.text:
                        delta = output_transcription.text
                        assistant_text_raw += delta
                        cleaned = clean_assistant_text(assistant_text_raw)
                        if len(cleaned) > len(assistant_text_clean_sent):
                            new_delta = cleaned[len(assistant_text_clean_sent) :]
                            assistant_text_clean_sent = cleaned
                            yield {"type": "assistant_text_delta", "text": new_delta}

                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn and model_turn.parts:
                        for part in model_turn.parts:
                            if getattr(part, "text", None):
                                delta = str(part.text)
                                if delta:
                                    assistant_text_raw += delta
                                    cleaned = clean_assistant_text(assistant_text_raw)
                                    if len(cleaned) > len(assistant_text_clean_sent):
                                        new_delta = cleaned[len(assistant_text_clean_sent) :]
                                        assistant_text_clean_sent = cleaned
                                        yield {"type": "assistant_text_delta", "text": new_delta}

                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                audio_bytes = _blob_to_bytes(inline_data.data)
                                if audio_bytes:
                                    yield {
                                        "type": "assistant_audio_chunk",
                                        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
                                        "sample_rate": SAMPLE_RATE,
                                    }

                    if getattr(server_content, "turn_complete", False):
                        got_turn_end = True
                        break

            assistant_text_clean = clean_assistant_text(assistant_text_raw)
            if not got_turn_end and not assistant_text_clean:
                assistant_text_clean = build_fallback_wine_answer(cleaned_user_text, self.history)
                for token in split_text_stream(assistant_text_clean):
                    yield {"type": "assistant_text_delta", "text": token}

            assistant_text = normalize_wine_response(assistant_text_clean)
            if len(assistant_text) > len(assistant_text_clean_sent):
                final_delta = assistant_text[len(assistant_text_clean_sent) :]
                if final_delta:
                    yield {"type": "assistant_text_delta", "text": final_delta}
            self.history.append({"role": "user", "text": cleaned_user_text})
            self.history.append({"role": "assistant", "text": assistant_text})

            if len(self.history) > 32:
                self.history = self.history[-32:]

            yield {"type": "assistant_turn_end", "text": assistant_text}

    async def _stream_dummy_audio_reply(
        self, audio_queue: "asyncio.Queue[Optional[tuple[bytes, int]]]"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            item = await audio_queue.get()
            if item is None:
                break

        transcript = "Recommend a crisp white wine for seafood with lemon butter."
        yield {"type": "user_transcript_delta", "text": transcript}
        async for event in self._stream_dummy_reply(transcript):
            yield event

    async def stream_reply_from_audio_queue(
        self, audio_queue: "asyncio.Queue[Optional[tuple[bytes, int]]]"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async with self.lock:
            self.last_used = time.time()

            if self.dummy_mode:
                async for event in self._stream_dummy_audio_reply(audio_queue):
                    yield event
                return

            await self.ensure_live()
            assert self._live_session is not None

            stream_sample_rate = SAMPLE_RATE
            user_transcript_raw = ""
            user_transcript_sent = ""
            assistant_text_raw = ""
            assistant_text_sent = ""

            async def _sender() -> None:
                nonlocal stream_sample_rate
                while True:
                    item = await audio_queue.get()
                    if item is None:
                        silence_samples = max(int(stream_sample_rate * 0.2), 1)
                        silence_chunk = b"\x00\x00" * silence_samples
                        mime = f"audio/pcm;rate={stream_sample_rate}"
                        for _ in range(2):
                            await self._live_session.send_realtime_input(
                                audio=genai_types.Blob(data=silence_chunk, mime_type=mime)
                            )
                            await asyncio.sleep(0)
                        return

                    audio_bytes, sample_rate = item
                    if sample_rate > 0:
                        stream_sample_rate = sample_rate
                    if not audio_bytes:
                        continue
                    await self._live_session.send_realtime_input(
                        audio=genai_types.Blob(
                            data=audio_bytes,
                            mime_type=f"audio/pcm;rate={stream_sample_rate}",
                        )
                    )

            sender_task = asyncio.create_task(_sender())

            try:
                while True:
                    got_turn_end = False
                    user_transcript_raw = ""
                    user_transcript_sent = ""
                    assistant_text_raw = ""
                    assistant_text_sent = ""

                    async with asyncio.timeout(75):
                        async for message in self._live_session.receive():
                            server_content = getattr(message, "server_content", None)
                            if not server_content:
                                continue

                            input_transcription = getattr(server_content, "input_transcription", None)
                            if input_transcription and input_transcription.text:
                                user_delta = str(input_transcription.text)
                                user_transcript_raw += user_delta
                                cleaned_user = clean_user_transcript(user_transcript_raw)
                                if len(cleaned_user) > len(user_transcript_sent):
                                    new_user_delta = cleaned_user[len(user_transcript_sent) :]
                                    user_transcript_sent = cleaned_user
                                    yield {"type": "user_transcript_delta", "text": new_user_delta}

                            output_transcription = getattr(server_content, "output_transcription", None)
                            if output_transcription and output_transcription.text:
                                assistant_delta = str(output_transcription.text)
                                assistant_text_raw += assistant_delta
                                cleaned_assistant = clean_assistant_text(assistant_text_raw)
                                if len(cleaned_assistant) > len(assistant_text_sent):
                                    new_assistant_delta = cleaned_assistant[len(assistant_text_sent) :]
                                    assistant_text_sent = cleaned_assistant
                                    yield {"type": "assistant_text_delta", "text": new_assistant_delta}

                            model_turn = getattr(server_content, "model_turn", None)
                            if model_turn and model_turn.parts:
                                for part in model_turn.parts:
                                    if (
                                        getattr(part, "text", None)
                                        and not getattr(part, "thought", False)
                                        and not output_transcription
                                        and part.text
                                    ):
                                        assistant_delta = str(part.text)
                                        assistant_text_raw += assistant_delta
                                        cleaned_assistant = clean_assistant_text(assistant_text_raw)
                                        if len(cleaned_assistant) > len(assistant_text_sent):
                                            new_assistant_delta = cleaned_assistant[
                                                len(assistant_text_sent) :
                                            ]
                                            assistant_text_sent = cleaned_assistant
                                            yield {
                                                "type": "assistant_text_delta",
                                                "text": new_assistant_delta,
                                            }

                                    inline_data = getattr(part, "inline_data", None)
                                    if inline_data and getattr(inline_data, "data", None):
                                        audio_bytes = _blob_to_bytes(inline_data.data)
                                        if audio_bytes:
                                            yield {
                                                "type": "assistant_audio_chunk",
                                                "audio_base64": base64.b64encode(audio_bytes).decode(
                                                    "ascii"
                                                ),
                                                "sample_rate": SAMPLE_RATE,
                                            }

                            if getattr(server_content, "turn_complete", False):
                                got_turn_end = True
                                break

                    transcript_clean = clean_user_transcript(user_transcript_raw)
                    if not transcript_clean:
                        # Skip emitting 'no_speech_detected' unless it's the very first loop, to avoid UI clears mid-stream
                        # but for continuous listening, we'll just yield emptiness and loop again.
                        pass
                    else:
                        assistant_text_clean = clean_assistant_text(assistant_text_raw)
                        
                        assistant_text = normalize_wine_response(assistant_text_clean)
                        if len(assistant_text) > len(assistant_text_sent):
                            final_delta = assistant_text[len(assistant_text_sent) :]
                            if final_delta:
                                yield {"type": "assistant_text_delta", "text": final_delta}

                        self.history.append({"role": "user", "text": transcript_clean})
                        self.history.append({"role": "assistant", "text": assistant_text})
                        if len(self.history) > 32:
                            self.history = self.history[-32:]

                        yield {"type": "assistant_turn_end", "text": assistant_text}

            finally:
                if not sender_task.done():
                    sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender_task



class SessionManager:
    def __init__(self, api_key: Optional[str], dummy_mode: bool):
        self.api_key = api_key
        self.dummy_mode = dummy_mode
        self.sessions: Dict[str, ChatSession] = {}
        self.lock = asyncio.Lock()
        self.stt_client = genai.Client(api_key=api_key) if (api_key and not dummy_mode) else None

    async def get_session(self, session_id: str) -> ChatSession:
        async with self.lock:
            existing = self.sessions.get(session_id)
            if existing:
                existing.last_used = time.time()
                return existing

            created = ChatSession(
                session_id=session_id,
                api_key=self.api_key,
                dummy_mode=self.dummy_mode,
            )
            self.sessions[session_id] = created
            return created

    async def reset_session(self, session_id: str) -> None:
        async with self.lock:
            session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        async with self.lock:
            sessions = list(self.sessions.values())
            self.sessions.clear()

        for session in sessions:
            await session.close()

    async def cleanup_idle(self) -> None:
        now = time.time()
        stale_ids: List[str] = []
        async with self.lock:
            for session_id, session in self.sessions.items():
                if now - session.last_used > SESSION_TTL_SECONDS:
                    stale_ids.append(session_id)

        for session_id in stale_ids:
            await self.reset_session(session_id)

    async def transcribe(self, audio_bytes: bytes, mime_type: str) -> str:
        if self.dummy_mode or not self.stt_client:
            return "Recommend a smooth red wine for pasta."

        prompt = (
            "Transcribe this audio to plain text. "
            "Return only the transcript with no additional commentary."
        )

        def _call_stt() -> str:
            response = self.stt_client.models.generate_content(
                model=STT_MODEL,
                contents=[
                    prompt,
                    genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=200,
                ),
            )
            text = (response.text or "").strip()
            return text

        transcript = await asyncio.to_thread(_call_stt)
        if not transcript:
            transcript = "Recommend a crisp white wine for grilled fish."
        return transcript


def create_app(dummy_mode: Optional[bool] = None) -> FastAPI:
    resolved_dummy_mode = USE_DUMMY_GEMINI if dummy_mode is None else dummy_mode

    if not resolved_dummy_mode and not GEMINI_API_KEY:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in .env. Set GEMINI_API_KEY or enable USE_DUMMY_GEMINI=true."
        )

    app = FastAPI(title="Wine Bot Backend", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session_manager = SessionManager(api_key=GEMINI_API_KEY, dummy_mode=resolved_dummy_mode)
    app.state.session_manager = session_manager

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.session_manager.close_all()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "dummy_mode": resolved_dummy_mode,
            "live_model": LIVE_MODEL,
            "stt_model": STT_MODEL,
            "voice_id": VOICE_ID,
        }

    @app.post("/stt")
    async def stt(file: UploadFile = File(...)) -> Dict[str, str]:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload.")

        mime_type = file.content_type or "audio/webm"
        transcript = await app.state.session_manager.transcribe(audio_bytes, mime_type)
        return {"text": transcript}

    @app.websocket("/ws/chat")
    async def chat_socket(websocket: WebSocket) -> None:
        session_id = websocket.query_params.get("session_id") or uuid.uuid4().hex
        session = await app.state.session_manager.get_session(session_id)
        await websocket.accept()
        ws_send_lock = asyncio.Lock()

        async def ws_send(payload: Dict[str, Any]) -> None:
            async with ws_send_lock:
                await websocket.send_json(payload)

        await ws_send({"type": "session_ready", "session_id": session_id})
        audio_queue: Optional[asyncio.Queue[Optional[tuple[bytes, int]]]] = None
        audio_task: Optional[asyncio.Task[None]] = None
        text_task: Optional[asyncio.Task[None]] = None

        async def stop_audio_turn() -> None:
            nonlocal audio_queue, audio_task
            if audio_queue is not None:
                await audio_queue.put(None)
            if audio_task is not None and not audio_task.done():
                with contextlib.suppress(Exception):
                    await audio_task
            audio_queue = None
            audio_task = None

        async def stop_text_turn() -> None:
            nonlocal text_task
            if text_task is not None and not text_task.done():
                text_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await text_task
            text_task = None

        async def stop_all_turns() -> None:
            await stop_audio_turn()
            await stop_text_turn()

        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "ping":
                    await ws_send({"type": "pong"})
                    continue

                if msg_type == "reset":
                    await stop_all_turns()
                    await app.state.session_manager.reset_session(session_id)
                    session = await app.state.session_manager.get_session(session_id)
                    await ws_send({"type": "session_reset"})
                    continue

                if msg_type == "user_audio_start":
                    await stop_all_turns()

                    audio_queue = asyncio.Queue()

                    async def _audio_runner(
                        queue: "asyncio.Queue[Optional[tuple[bytes, int]]]",
                    ) -> None:
                        nonlocal audio_task, audio_queue
                        try:
                            async for event in session.stream_reply_from_audio_queue(queue):
                                await ws_send(event)
                        except Exception:
                            fallback = build_fallback_wine_answer(
                                "Suggest a versatile wine style.",
                                session.history,
                            )
                            await ws_send({"type": "assistant_text_delta", "text": fallback})
                            if session.dummy_mode:
                                pcm_data = synth_dummy_pcm(fallback)
                                async for chunk_b64 in pcm16_to_base64_chunks(pcm_data):
                                    await ws_send(
                                        {
                                            "type": "assistant_audio_chunk",
                                            "audio_base64": chunk_b64,
                                            "sample_rate": SAMPLE_RATE,
                                        }
                                    )
                            await ws_send({"type": "assistant_turn_end", "text": fallback})
                        finally:
                            audio_task = None
                            audio_queue = None

                    audio_task = asyncio.create_task(_audio_runner(audio_queue))
                    await ws_send({"type": "audio_started"})
                    continue

                if msg_type == "user_audio_chunk":
                    if audio_queue is None:
                        await ws_send({"type": "error", "message": "No active audio turn."})
                        continue
                    audio_base64 = str(message.get("audio_base64", "")).strip()
                    sample_rate = int(message.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
                    if not audio_base64:
                        continue
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                    except Exception as e:
                        print(f"ERROR processing user audio chunk: {e}")
                        continue
                    await audio_queue.put((audio_bytes, sample_rate))
                    continue

                if msg_type == "user_audio_end":
                    if audio_queue is not None:
                        await audio_queue.put(None)
                    continue

                if msg_type != "user_text":
                    await ws_send(
                        {"type": "error", "message": "Unsupported message type."}
                    )
                    continue

                text = str(message.get("text", "")).strip()
                if not text:
                    await ws_send({"type": "assistant_turn_end", "text": ""})
                    continue

                await stop_all_turns()

                async def _text_runner() -> None:
                    nonlocal text_task
                    try:
                        async for event in session.stream_reply(text):
                            await ws_send(event)
                    except Exception as e:
                        import traceback
                        print(f"ERROR in stream_reply (text): {e}")
                        traceback.print_exc()
                        fallback = build_fallback_wine_answer(text, session.history)
                        await ws_send({"type": "assistant_text_delta", "text": fallback})
                        if session.dummy_mode:
                            pcm_data = synth_dummy_pcm(fallback)
                            async for chunk_b64 in pcm16_to_base64_chunks(pcm_data):
                                await ws_send(
                                    {
                                        "type": "assistant_audio_chunk",
                                        "audio_base64": chunk_b64,
                                        "sample_rate": SAMPLE_RATE,
                                    }
                                )
                        await ws_send({"type": "assistant_turn_end", "text": fallback})
                    finally:
                        text_task = None

                text_task = asyncio.create_task(_text_runner())
                await app.state.session_manager.cleanup_idle()
        except WebSocketDisconnect:
            await stop_all_turns()
            return

    return app


app = create_app()
