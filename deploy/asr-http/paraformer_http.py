#!/usr/bin/env python3
"""把 OpenAI /v1/audio/transcriptions 转到本机 FunASR paraformer WebSocket。"""

import cgi
import json
import os
import subprocess
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.sync.client import connect

UPSTREAM = os.environ.get("FUNASR_WS", "ws://127.0.0.1:10095")
PORT = int(os.environ.get("PORT", "8000"))


MAX_AUDIO_SEC = float(os.environ.get("ASR_MAX_AUDIO_SEC", "120"))
MAX_UPLOAD_BYTES = 32 * 1024 * 1024
REQUEST_SLOT = threading.BoundedSemaphore(1)

class AudioTooLong(ValueError):
    pass


def to_wav(src: str) -> str:
    wav = src + ".16k.wav"
    subprocess.check_call(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", src,
            "-t", str(MAX_AUDIO_SEC + 1),
            "-ac", "1", "-ar", "16000",
            "-f", "wav", wav,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=60,
    )
    with wave.open(wav, "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if duration > MAX_AUDIO_SEC:
        # A bounded decode detects oversize input; truncated content is never successful.
        os.remove(wav)
        raise AudioTooLong(f"audio exceeds {MAX_AUDIO_SEC:g}s; split it before upload")
    return wav


def transcribe(wav_path: str) -> str:
    with wave.open(wav_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        audio_bytes = wav_file.readframes(wav_file.getnframes())
    if not audio_bytes:
        return ""
    stride = int(60 * 10 / 10 / 1000 * sample_rate * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1
    header = {
        "mode": "offline",
        "chunk_size": [5, 10, 5],
        "chunk_interval": 10,
        "encoder_chunk_look_back": 4,
        "decoder_chunk_look_back": 0,
        "audio_fs": sample_rate,
        "wav_name": os.path.basename(wav_path),
        "wav_format": "pcm",
        "is_speaking": True,
        "hotwords": "",
        "itn": True,
    }
    text = ""
    with connect(UPSTREAM, subprotocols=["binary"], open_timeout=15, close_timeout=5) as ws:
        ws.send(json.dumps(header))
        for i in range(chunk_num):
            beg = i * stride
            ws.send(audio_bytes[beg:beg + stride])
            if i == chunk_num - 1:
                ws.send(json.dumps({"is_speaking": False}))
        while True:
            raw = ws.recv(timeout=180)
            message = json.loads(raw)
            if message.get("mode") == "offline":
                text = message.get("text", "")
                break
    return text


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/health":
            self._send(200, {"status": "ok", "model": "paraformer-large", "max_audio_sec": MAX_AUDIO_SEC, "max_concurrent": 1})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/v1/audio/transcriptions":
            self._send(404, {"error": "not found"})
            return
        if not REQUEST_SLOT.acquire(blocking=False):
            self._send(429, {"error": "ASR busy; retry this chunk later"})
            return
        self.connection.settimeout(60)
        src = ""
        wav = ""
        try:
            length = self.headers.get("Content-Length", "")
            if not length.isdigit() or int(length) <= 0:
                self._send(411, {"error": "positive Content-Length required"})
                return
            if int(length) > MAX_UPLOAD_BYTES:
                self._send(413, {"error": "upload exceeds 32 MiB"})
                return
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            item = form["file"]
            raw = item.file.read()
            suffix = Path(item.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                src = tmp.name
            wav = src + ".16k.wav"
            to_wav(src)
            text = transcribe(wav)
            self._send(200, {"text": text, "model": "paraformer-large"})
        except AudioTooLong as exc:
            self._send(413, {"error": str(exc)})
        except Exception as exc:
            self._send(500, {"error": str(exc)})
        finally:
            try:
                for path in (src, wav):
                    if path and os.path.exists(path):
                        os.remove(path)
            finally:
                REQUEST_SLOT.release()

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


if __name__ == "__main__":
    print(f"paraformer http {PORT} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
