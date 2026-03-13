import os
import io
import json
import re
import struct
import time
import uuid
import logging
import asyncio
import wave
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import AsyncGroq

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Interview Practice Platform")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SMALLEST_API_KEY = os.getenv("SMALLEST_API_KEY")

GROQ_MODEL = "llama-3.1-8b-instant"
STT_REST_URL = "https://api.smallest.ai/waves/v1/pulse/get_text"
TTS_REST_URL = "https://api.smallest.ai/waves/v1/lightning-v3.1/get_speech"

# ---------------------------------------------------------------------------
# In-memory session storage
# ---------------------------------------------------------------------------
sessions: dict = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class JobContext(BaseModel):
    job_title: str
    company: str
    interview_type: str
    experience_level: str
    focus_areas: Optional[list[str]] = None


class StartSessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Helper: build the interviewer system prompt
# ---------------------------------------------------------------------------
def build_system_prompt(ctx: JobContext) -> str:
    focus = ""
    if ctx.focus_areas:
        focus = f" Focus especially on these areas: {', '.join(ctx.focus_areas)}."
    return (
        f"You are a professional interviewer at {ctx.company} interviewing a candidate "
        f"for the {ctx.job_title} position. This is a {ctx.interview_type} interview. "
        f"The candidate has {ctx.experience_level} experience.{focus} "
        "Your name is Alex. "
        "Ask one question at a time. Start with a brief warm greeting and an icebreaker question. "
        "Then proceed with 5-7 relevant questions. Follow up on vague or incomplete answers. "
        "Be professional but warm and encouraging. Keep responses concise — you're speaking out loud. "
        "When you've asked enough questions, say 'That wraps up our interview today' to signal the end."
    )


FEEDBACK_SYSTEM_PROMPT = """You are an expert interview coach analyzing a mock interview transcript.

Return a JSON object with this exact structure:
{
  "overall_score": <number 1-10>,
  "categories": {
    "clarity": <number 1-10>,
    "structure": <number 1-10>,
    "relevance": <number 1-10>,
    "confidence": <number 1-10>
  },
  "strengths": ["<specific strength with example from their answers>", ...],
  "improvements": ["<specific actionable improvement tip>", ...],
  "quotes": [
    {"quote": "<exact words the candidate said>", "suggestion": "<how they could rephrase or improve this specific answer>"},
    ...
  ],
  "action_plan": [
    "<concrete practice exercise or habit to build before the real interview>",
    ...
  ],
  "summary": "<2-3 sentence overall assessment — what stood out, biggest area to work on, and encouragement>"
}

Scoring guide:
- 1-3: Major issues, unclear or off-topic answers
- 4-5: Below average, vague answers lacking specifics
- 6-7: Solid but room for improvement, decent examples
- 8-9: Strong answers with clear structure and specifics
- 10: Exceptional, polished and memorable

Be specific — reference their actual words. For improvements, explain WHY it matters and HOW to fix it. Include at least 2-3 quotes with concrete rewording suggestions. The action_plan should have 2-4 practical exercises.

Return ONLY valid JSON, no markdown fences or extra text."""


# ---------------------------------------------------------------------------
# Helper: call Groq LLM (non-streaming, used for feedback)
# ---------------------------------------------------------------------------
async def call_llm(messages: list[dict]) -> str:
    client = AsyncGroq(api_key=GROQ_API_KEY)
    try:
        chat = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return chat.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq LLM error: {e}")
        raise


# ---------------------------------------------------------------------------
# Helper: Groq LLM streaming — yields sentence chunks for faster TTS
# ---------------------------------------------------------------------------
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

async def stream_llm_sentences(messages: list[dict]):
    """Stream LLM response and yield complete sentences as they form."""
    client = AsyncGroq(api_key=GROQ_API_KEY)
    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
        stream=True,
    )
    buffer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            buffer += delta.content
            # Check if we have a complete sentence
            parts = SENTENCE_END.split(buffer)
            if len(parts) > 1:
                # Yield all complete sentences, keep the remainder
                for sentence in parts[:-1]:
                    sentence = sentence.strip()
                    if sentence:
                        yield sentence
                buffer = parts[-1]
    # Yield whatever is left
    if buffer.strip():
        yield buffer.strip()


# ---------------------------------------------------------------------------
# Helper: Smallest.ai TTS (REST)
# ---------------------------------------------------------------------------
async def text_to_speech(text: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            TTS_REST_URL,
            headers={
                "Authorization": f"Bearer {SMALLEST_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "voice_id": "magnus",
                "sample_rate": 24000,
                "language": "en",
                "output_format": "pcm",
            },
        )
        if resp.status_code != 200:
            logger.error(f"TTS error {resp.status_code}: {resp.text}")
            raise RuntimeError(f"TTS request failed: {resp.status_code}")
        return resp.content


# ---------------------------------------------------------------------------
# Helper: Smallest.ai Pulse STT (REST)
# ---------------------------------------------------------------------------
def pcm_to_wav(pcm_data: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Wrap raw PCM bytes in a proper WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def speech_to_text(audio_data: bytes) -> str:
    wav_data = pcm_to_wav(audio_data)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{STT_REST_URL}?language=en",
                headers={
                    "Authorization": f"Bearer {SMALLEST_API_KEY}",
                    "Content-Type": "audio/wav",
                },
                content=wav_data,
            )
            if resp.status_code != 200:
                logger.error(f"STT error {resp.status_code}: {resp.text}")
                raise RuntimeError(f"STT request failed: {resp.status_code}")
            data = resp.json()
            if isinstance(data, dict):
                return data.get("transcription", data.get("text", data.get("transcript", ""))).strip()
            return str(data).strip()
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise


# ---------------------------------------------------------------------------
# Helper: generate interviewer response with streaming TTS
# ---------------------------------------------------------------------------
CHUNK = 16_384

async def generate_and_speak(ws: WebSocket, session: dict):
    """Stream LLM sentence-by-sentence, TTS each sentence, send audio immediately."""
    full_response = ""
    async for sentence in stream_llm_sentences(session["history"]):
        full_response += (" " if full_response else "") + sentence
        # Send text chunk so frontend can show it progressively
        await ws.send_json({"type": "interviewer_text", "text": sentence})
        # TTS this sentence and stream audio right away
        try:
            audio = await text_to_speech(sentence)
            for i in range(0, len(audio), CHUNK):
                await ws.send_bytes(audio[i : i + CHUNK])
        except Exception as e:
            logger.error(f"TTS failed for chunk: {e}")
    # Signal end of all audio
    await ws.send_json({"type": "audio_end"})
    # Store full response in history
    session["history"].append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/start-session", response_model=StartSessionResponse)
async def start_session(ctx: JobContext):
    session_id = str(uuid.uuid4())
    system_prompt = build_system_prompt(ctx)
    sessions[session_id] = {
        "job_context": ctx.model_dump(),
        "system_prompt": system_prompt,
        "history": [{"role": "system", "content": system_prompt}],
        "start_time": time.time(),
    }
    logger.info(f"Session created: {session_id}")
    return StartSessionResponse(session_id=session_id)


@app.post("/api/feedback/{session_id}")
async def get_feedback(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    transcript_lines = []
    for msg in session["history"]:
        if msg["role"] == "system":
            continue
        role_label = "Interviewer" if msg["role"] == "assistant" else "Candidate"
        transcript_lines.append(f"{role_label}: {msg['content']}")
    transcript = "\n".join(transcript_lines)

    if not transcript_lines:
        raise HTTPException(status_code=400, detail="No conversation to evaluate")

    messages = [
        {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the interview transcript:\n\n{transcript}"},
    ]

    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        chat = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        raw = chat.choices[0].message.content
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        feedback = json.loads(cleaned)
        return feedback
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON feedback, wrapping raw text")
        return {"raw_feedback": raw}
    except Exception as e:
        logger.error(f"Feedback generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback")


# ---------------------------------------------------------------------------
# WebSocket interview handler
# ---------------------------------------------------------------------------

@app.websocket("/ws/interview/{session_id}")
async def interview_ws(ws: WebSocket, session_id: str):
    await ws.accept()

    if session_id not in sessions:
        await ws.send_json({"type": "error", "text": "Invalid session ID"})
        await ws.close()
        return

    session = sessions[session_id]
    logger.info(f"WebSocket connected for session {session_id}")

    # ------ Step 1: Generate and send the interviewer greeting (streamed) ------
    try:
        await generate_and_speak(ws, session)
        logger.info("Greeting sent")
    except Exception as e:
        logger.error(f"Error sending greeting: {e}")
        await ws.send_json({"type": "error", "text": "Failed to start interview"})
        await ws.close()
        return

    # ------ Step 2: Main conversation loop ------
    audio_buffer = bytearray()
    SILENCE_TIMEOUT = 2.0

    async def process_audio():
        """Transcribe accumulated audio, stream LLM + TTS back."""
        nonlocal audio_buffer
        if len(audio_buffer) == 0:
            return

        audio_data = bytes(audio_buffer)
        audio_buffer = bytearray()

        # Notify client we're processing
        await ws.send_json({"type": "processing", "text": "Transcribing..."})

        # STT
        try:
            transcript = await speech_to_text(audio_data)
        except Exception as e:
            logger.error(f"STT failed: {e}")
            await ws.send_json({"type": "error", "text": "Speech recognition failed"})
            return

        if not transcript:
            logger.info("Empty transcript, skipping")
            await ws.send_json({"type": "processing", "text": ""})
            return

        logger.info(f"User said: {transcript}")
        await ws.send_json({"type": "transcript", "text": transcript})

        # Add user message to history
        session["history"].append({"role": "user", "content": transcript})

        # Stream LLM + TTS response
        try:
            await generate_and_speak(ws, session)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            await ws.send_json({"type": "error", "text": "Failed to get interviewer response"})

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=SILENCE_TIMEOUT)
            except asyncio.TimeoutError:
                if audio_buffer:
                    await process_audio()
                continue

            if msg["type"] == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                audio_buffer.extend(msg["bytes"])

            elif "text" in msg and msg["text"]:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "end_of_speech":
                    await process_audio()

                elif data.get("type") == "end_session":
                    logger.info(f"Session {session_id} ended by client")
                    await ws.send_json({"type": "session_ended"})
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Static file serving
# ---------------------------------------------------------------------------
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Interview Practice Platform API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
