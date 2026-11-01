import concurrent.futures
import http.client
import importlib.util
import io
import json
import os
import sys
import threading
import types
import urllib.request
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "deploy" / "asr-http"


def pcm(seconds):
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        w.writeframes(b'\0\0' * int(seconds * 16000))
    return b.getvalue()


def body(seconds):
    return b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\n' + pcm(seconds) + b'\r\n--boundary--\r\n'


@pytest.fixture

def server(monkeypatch):
    calls = []
    entered, release = threading.Event(), threading.Event()
    class Model:
        def __init__(self, *args, **kwargs): pass
        def __call__(self, paths, **kwargs):
            calls.append(paths)
            entered.set()
            assert release.wait(5)
            return ['hello']
    monkeypatch.setitem(sys.modules, 'funasr_onnx', types.SimpleNamespace(SenseVoiceSmall=Model))
    monkeypatch.setenv('ASR_MAX_AUDIO_SEC', '2')
    spec = importlib.util.spec_from_file_location('guarded_service', ROOT / 'sensevoice.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    httpd = module.ThreadingHTTPServer(('127.0.0.1', 0), module.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True); t.start()
    yield httpd.server_address[1], calls, entered, release
    release.set(); httpd.shutdown(); httpd.server_close(); t.join(2)


def request(port, data):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    c.request('POST', '/v1/audio/transcriptions', data, {'Content-Type': 'multipart/form-data; boundary=boundary'})
    r = c.getresponse(); result = r.status, json.loads(r.read()); c.close(); return result


def test_long_decoded_audio_rejected_and_next_short_request_works(server):
    port, calls, entered, release = server
    assert request(port, body(3))[0] == 413
    assert not calls
    release.set()
    assert request(port, body(2))[0] == 200
    assert len(calls) == 1


def test_busy_rejected_before_reading_body(server):
    port, calls, entered, release = server
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(request, port, body(1))
        assert entered.wait(3)
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        c.putrequest('POST', '/v1/audio/transcriptions')
        c.putheader('Content-Length', '100'); c.endheaders()
        r = c.getresponse()
        assert r.status == 429
        assert r.getheader('Connection') == 'close'
        r.read(); c.close()
        release.set()
        assert first.result()[0] == 200
    assert len(calls) == 1


def test_upload_limit_checked_without_body(server):
    port, calls, _, _ = server
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    c.putrequest('POST', '/v1/audio/transcriptions')
    c.putheader('Content-Length', str(33 * 1024 * 1024)); c.endheaders()
    r = c.getresponse(); assert r.status == 413
    r.read(); c.close(); assert not calls
