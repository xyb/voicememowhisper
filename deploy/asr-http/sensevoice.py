#!/usr/bin/env python3
"""SenseVoice Small ONNX，对外保持 OpenAI /v1/audio/transcriptions。"""

import cgi
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/opt/pydeps")
sys.path.insert(0, "/workspace/FunASR/runtime/python/onnxruntime")

from funasr_onnx import SenseVoiceSmall

MODEL_DIR = os.environ.get("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
CACHE = os.environ.get("SENSEVOICE_CACHE", "/models")
THREADS = int(os.environ.get("SENSEVOICE_THREADS", "4"))
PORT = int(os.environ.get("PORT", "8000"))
QUANTIZE = os.environ.get("SENSEVOICE_QUANT", "1") != "0"
TOKEN_RE = re.compile(r"<\|.*?\|>")

print(f"loading {MODEL_DIR} quantize={QUANTIZE} threads={THREADS}", flush=True)
MODEL = SenseVoiceSmall(
    MODEL_DIR,
    batch_size=1,
    quantize=QUANTIZE,
    intra_op_num_threads=THREADS,
    cache_dir=CACHE,
)
print("loaded", flush=True)
LOCK = threading.Lock()


def clean(text: str) -> str:
    return " ".join(TOKEN_RE.sub(" ", text or "").split())


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
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, {"status": "ok", "model": "sensevoice-small", "max_audio_sec": MAX_AUDIO_SEC, "max_concurrent": 1})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path != "/v1/audio/transcriptions":
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
            language = form.getfirst("language") or "auto"
            if language not in {"auto", "zh", "en", "ja", "ko", "yue"}:
                language = "auto"
            item = form["file"]
            raw = item.file.read()
            suffix = Path(item.filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw)
                src = tmp.name
            wav = src + ".16k.wav"
            to_wav(src)
            with LOCK:
                result = MODEL([wav], language=language, use_itn=True)
            text = clean(result[0] if result else "")
            self._send(200, {"text": text, "language": language, "model": "sensevoice-small"})
        except AudioTooLong as exc:
            self._send(413, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
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
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
