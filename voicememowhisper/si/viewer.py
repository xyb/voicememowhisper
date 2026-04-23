"""Interactive word-aligned waveform + spectrogram viewer.

Produces a self-contained HTML page that shows, for a given audio clip:

  * a big waveform + log-spectrogram figure with colored bands per word,
  * the character text rendered as HTML labels above the figure,
  * a red playback cursor that tracks the ``<audio>`` element at 60fps
    via ``requestAnimationFrame``,
  * click-to-seek on every character label and every band.

Intended use cases:

  * auditioning speaker-library enrollment clips and eyeballing word
    boundaries vs actual speech energy,
  * debugging a transcript's word-level alignment on a candidate clip
    without spinning up an audio editor.

The word-level timestamps come from faster-whisper run with
``word_timestamps=True``. This sidesteps the FunASR / paraformer HTTP
backend which doesn't expose word-level granularity at the time of
this writing.

The default model is "small" for interactive iteration; bump to
"medium" or "large-v3" when recognition accuracy matters more than
turnaround time.
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# Fixed fractional plot-area layout. HTML uses the same constants to
# translate time → x-percent reliably. Do not let matplotlib's tight
# layout move these around.
_PLOT_LEFT = 0.040
_PLOT_RIGHT = 0.995
_WAVE_TOP = 0.985
_WAVE_BOTTOM = 0.620
_SPEC_TOP = 0.580
_SPEC_BOTTOM = 0.090

_TAB20_HEX = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
]

# CJK fonts matplotlib actually indexes on macOS. "PingFang SC" is
# present on every modern macOS but matplotlib's font_manager can't see
# it — use these instead to avoid tofu boxes.
_CJK_CANDIDATES = ("Hiragino Sans GB", "STHeiti", "Songti SC",
                   "Arial Unicode MS")


def _configure_mpl_fonts() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in _CJK_CANDIDATES:
        if cand in available:
            plt.rcParams["font.family"] = [cand, "sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def _transcribe_with_words(audio: Path, model_name: str, language: str) -> dict:
    """Run faster-whisper with word_timestamps=True and return a dict."""
    from faster_whisper import WhisperModel

    print(f"[viewer] loading faster-whisper {model_name} ...", file=sys.stderr)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"[viewer] transcribing {audio.name} ...", file=sys.stderr)
    segments_iter, info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        vad_filter=False,
    )
    out = {
        "audio": str(audio),
        "duration": info.duration,
        "model": model_name,
        "language": language,
        "segments": [],
    }
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": w.start,
                    "end": w.end,
                    "text": w.word,
                    "probability": getattr(w, "probability", None),
                })
        out["segments"].append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words,
        })
    return out


def _load_audio_mono_16k(audio: Path):
    """Decode audio to mono 16 kHz via ffmpeg and read as numpy array."""
    import numpy as np
    from scipy.io import wavfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(audio), "-ac", "1", "-ar", "16000", str(tmp_path)],
            check=True,
        )
        sr, samples = wavfile.read(str(tmp_path))
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        return sr, samples
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_figure(data: dict, audio: Path, out_png: Path) -> tuple[int, int]:
    """Render waveform + spectrogram PNG. Return (width_px, height_px)."""
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import spectrogram

    _configure_mpl_fonts()

    sr, samples = _load_audio_mono_16k(audio)
    t_audio = np.arange(len(samples)) / sr
    duration = len(samples) / sr

    words: list[dict] = []
    for seg in data["segments"]:
        words.extend(seg.get("words", []))

    cmap = plt.get_cmap("tab20")

    def band_color(i: int, alpha: float = 0.28) -> tuple:
        r, g, b, _ = cmap(i % cmap.N)
        return (r, g, b, alpha)

    fig_w_in = min(24, max(14, duration * 1.8))
    fig_h_in = 5.0
    dpi = 150
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax_wave = fig.add_axes([_PLOT_LEFT, _WAVE_BOTTOM,
                            _PLOT_RIGHT - _PLOT_LEFT,
                            _WAVE_TOP - _WAVE_BOTTOM])
    ax_spec = fig.add_axes([_PLOT_LEFT, _SPEC_BOTTOM,
                            _PLOT_RIGHT - _PLOT_LEFT,
                            _SPEC_TOP - _SPEC_BOTTOM])

    ax_wave.plot(t_audio, samples, color="#1f2937", linewidth=0.5)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_xlim(0, duration)
    ax_wave.set_xticks([])
    ax_wave.set_yticks([])

    f, t, Sxx = spectrogram(samples, fs=sr, nperseg=512, noverlap=384)
    Sdb = 10 * np.log10(Sxx + 1e-10)
    vmin = np.percentile(Sdb, 5)
    vmax = np.percentile(Sdb, 99)
    ax_spec.pcolormesh(t, f, Sdb, shading="auto", cmap="magma",
                       vmin=vmin, vmax=vmax)
    ax_spec.set_ylim(0, 4000)
    ax_spec.set_xlim(0, duration)
    ax_spec.set_ylabel("Hz", fontsize=8)
    ax_spec.set_xlabel("s", fontsize=8)
    ax_spec.tick_params(axis="both", labelsize=7)

    for i, w in enumerate(words):
        color = band_color(i)
        spec_color = (color[0], color[1], color[2], 0.16)
        ax_wave.axvspan(w["start"], w["end"], facecolor=color,
                        edgecolor="none", zorder=-1)
        ax_spec.axvspan(w["start"], w["end"], facecolor=spec_color,
                        edgecolor="none", zorder=3)
        for x in (w["start"], w["end"]):
            ax_wave.axvline(x, color="#6b7280", linewidth=0.4, alpha=0.5)
            ax_spec.axvline(x, color="#ffffff", linewidth=0.4, alpha=0.4)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return int(fig_w_in * dpi), int(fig_h_in * dpi)


def _build_html(
    data: dict,
    audio_name: str,
    figure_name: str,
    figure_size_px: tuple[int, int],
    out_html: Path,
) -> None:
    fig_w, fig_h = figure_size_px
    duration = data.get("duration", 1.0) or 1.0

    words: list[dict] = []
    for seg in data["segments"]:
        words.extend(seg.get("words", []))

    word_payload = []
    for i, w in enumerate(words):
        text = (w.get("text") or "").strip()
        prob = w.get("probability")
        word_payload.append({
            "i": i,
            "text": text,
            "start": w["start"],
            "end": w["end"],
            "prob": prob if prob is not None else -1,
            "color": _TAB20_HEX[i % len(_TAB20_HEX)],
        })

    words_js = json.dumps(word_payload, ensure_ascii=False)

    segments_list_html = "".join(
        f'<li>[{s["start"]:.2f}–{s["end"]:.2f}] {html.escape(s.get("text", "").strip())}</li>'
        for s in data["segments"]
    )
    audio_escaped = html.escape(audio_name)
    figure_escaped = html.escape(figure_name)
    model_name = html.escape(data.get("model", "?"))

    out_html.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Word-aligned viewer · {audio_escaped}</title>
<style>
  :root {{
    --plot-left: {_PLOT_LEFT * 100:.3f}%;
    --plot-right: {_PLOT_RIGHT * 100:.3f}%;
  }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
          color: #111; background: #fafafa; padding: 1rem 1.25rem 3rem; }}
  h1 {{ font-size: 1rem; margin: 0 0 0.25rem; }}
  .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 0.6rem; }}
  audio {{ width: 100%; margin-bottom: 0.5rem; }}
  #stage {{ position: relative; width: 100%; max-width: {fig_w}px;
            margin: 0 auto; user-select: none; }}
  #stage img {{ display: block; width: 100%; height: auto;
                border: 1px solid #e5e7eb; border-radius: 4px;
                aspect-ratio: {fig_w} / {fig_h}; }}
  #labels {{ position: relative; height: 46px; margin-bottom: 2px; }}
  #labels .lab {{ position: absolute; transform: translateX(-50%);
                  white-space: nowrap; font-size: 13px; line-height: 1.2;
                  padding: 1px 3px; background: #fff;
                  border: 1px solid #e5e7eb; border-radius: 3px;
                  cursor: pointer; }}
  #labels .lab.low {{ color: #b91c1c; opacity: 0.75; border-style: dashed; }}
  #labels .lab.row0 {{ top: 1px; }}
  #labels .lab.row1 {{ top: 22px; }}
  #labels .lab:hover {{ background: #fef3c7; z-index: 10; }}
  #cursor {{ position: absolute; top: 0; bottom: 0; width: 2px;
             background: #ef4444; pointer-events: none;
             display: none; z-index: 20; }}
  #cursor.visible {{ display: block; }}
  .legend {{ font-size: 0.75rem; color: #666; margin-top: 8px; }}
  ul {{ color: #444; font-size: 0.9rem; line-height: 1.6; }}
  #langswitch {{ position: fixed; top: 10px; right: 14px;
                 font-size: 11px; padding: 3px 7px;
                 background: #fff; border: 1px solid #d1d5db;
                 border-radius: 3px; cursor: pointer;
                 color: #374151; font-family: system-ui, sans-serif; }}
  #langswitch:hover {{ background: #f3f4f6; }}
</style>
</head><body>

<button id="langswitch" type="button" title="Switch language"></button>

<h1>{audio_escaped}</h1>
<div class="meta">
  {duration:.2f}s ·
  <span data-i18n="model">model</span> <code>{model_name}</code> ·
  {len(data['segments'])} <span data-i18n="segments">segments</span> ·
  {len(words)} <span data-i18n="words">words</span> ·
  <a href="transcript.json">transcript.json</a>
</div>
<audio id="audio" controls src="{audio_escaped}"></audio>

<div id="stage">
  <div id="labels"></div>
  <img id="fig" src="{figure_escaped}" alt="waveform + spectrogram">
  <div id="cursor"></div>
</div>

<div class="legend" data-i18n="legend"></div>

<h2 style="font-size:0.95rem;margin-top:1.5rem" data-i18n="segments_heading">Segments</h2>
<ul>{segments_list_html}</ul>

<script>
const WORDS = {words_js};
const DURATION = {duration:.3f};

const I18N = {{
  en: {{
    _label: "EN",
    model: "model",
    segments: "segments",
    words: "words",
    segments_heading: "Segments",
    legend: "Labels with a dashed red border have probability < 0.5 — likely hallucinated. Click any label or band to jump to that time.",
  }},
  zh: {{
    _label: "中",
    model: "模型",
    segments: "段",
    words: "字",
    segments_heading: "段落",
    legend: "红色虚线边框的字 = probability < 0.5，可能是幻觉字。点字或点图都能跳转到对应时间",
  }},
}};

function detectLang() {{
  const saved = (function() {{
    try {{ return localStorage.getItem("viewer_lang"); }} catch (e) {{ return null; }}
  }})();
  if (saved && I18N[saved]) return saved;
  const nav = (navigator.language || "en").toLowerCase();
  return nav.startsWith("zh") ? "zh" : "en";
}}

let currentLang = detectLang();

function applyLang() {{
  const strings = I18N[currentLang] || I18N.en;
  document.documentElement.lang = currentLang === "zh" ? "zh" : "en";
  document.querySelectorAll("[data-i18n]").forEach(el => {{
    const key = el.getAttribute("data-i18n");
    if (strings[key] !== undefined) el.textContent = strings[key];
  }});
  const btn = document.getElementById("langswitch");
  const other = currentLang === "zh" ? "en" : "zh";
  btn.textContent = I18N[other]._label;
  btn.title = "Switch to " + (other === "en" ? "English" : "中文");
}}

document.getElementById("langswitch").addEventListener("click", () => {{
  currentLang = (currentLang === "zh") ? "en" : "zh";
  try {{ localStorage.setItem("viewer_lang", currentLang); }} catch (e) {{}}
  applyLang();
}});
applyLang();
const PLOT_LEFT = {_PLOT_LEFT:.4f};
const PLOT_RIGHT = {_PLOT_RIGHT:.4f};
const PLOT_WIDTH = PLOT_RIGHT - PLOT_LEFT;

const audio = document.getElementById("audio");
const stage = document.getElementById("stage");
const labels = document.getElementById("labels");
const cursor = document.getElementById("cursor");

function timeToPct(t) {{
  return (PLOT_LEFT + (t / DURATION) * PLOT_WIDTH) * 100;
}}

function layout() {{
  labels.innerHTML = "";
  const oldOv = document.getElementById("clickbands");
  if (oldOv) oldOv.remove();
  const imgOverlay = document.createElement("div");
  imgOverlay.id = "clickbands";
  imgOverlay.style.position = "absolute";
  imgOverlay.style.left = "0";
  imgOverlay.style.right = "0";
  imgOverlay.style.top = labels.offsetHeight + "px";
  imgOverlay.style.bottom = "0";
  imgOverlay.style.pointerEvents = "none";
  stage.appendChild(imgOverlay);

  WORDS.forEach((w, idx) => {{
    const lab = document.createElement("div");
    const lowProb = (w.prob >= 0 && w.prob < 0.5);
    lab.className = "lab " + (idx % 2 === 0 ? "row0" : "row1") + (lowProb ? " low" : "");
    lab.textContent = w.text;
    lab.title = `${{w.start.toFixed(3)}}–${{w.end.toFixed(3)}} s${{w.prob >= 0 ? "  p=" + w.prob.toFixed(2) : ""}}`;
    const centerPct = (timeToPct(w.start) + timeToPct(w.end)) / 2;
    lab.style.left = centerPct + "%";
    lab.onclick = (e) => {{ e.preventDefault(); audio.currentTime = w.start; audio.play(); }};
    labels.appendChild(lab);

    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.left = timeToPct(w.start) + "%";
    band.style.width = (timeToPct(w.end) - timeToPct(w.start)) + "%";
    band.style.top = "0";
    band.style.bottom = "0";
    band.style.cursor = "pointer";
    band.style.pointerEvents = "auto";
    band.title = `${{w.text}}  ${{w.start.toFixed(3)}}–${{w.end.toFixed(3)}}s`;
    band.onclick = () => {{ audio.currentTime = w.start; audio.play(); }};
    imgOverlay.appendChild(band);
  }});
}}

function updateCursor() {{
  const t = audio.currentTime;
  if (!isFinite(t) || t < 0) {{ cursor.classList.remove("visible"); return; }}
  cursor.classList.add("visible");
  cursor.style.left = timeToPct(t) + "%";
}}

// 60 Hz cursor while playing so it doesn't step with the browser's
// 4-5 Hz `timeupdate` event.
let rafId = null;
function rafLoop() {{ updateCursor(); rafId = requestAnimationFrame(rafLoop); }}
function startRaf() {{ if (rafId == null) rafLoop(); }}
function stopRaf() {{
  if (rafId != null) {{ cancelAnimationFrame(rafId); rafId = null; }}
  updateCursor();
}}

window.addEventListener("load", layout);
window.addEventListener("resize", layout);
audio.addEventListener("play", startRaf);
audio.addEventListener("pause", stopRaf);
audio.addEventListener("ended", stopRaf);
audio.addEventListener("seeked", updateCursor);
audio.addEventListener("loadedmetadata", updateCursor);
</script>

</body></html>
""", encoding="utf-8")


def default_viewers_root() -> Path:
    """Where the viewer output dir lives by default."""
    import os
    env = os.environ.get("VOICE_MEMO_VIEWERS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".local/share/voicememowhisper/viewers").resolve()


def build_viewer(
    audio_src: Path,
    out_dir: Path,
    model_name: str = "small",
    language: str = "zh",
    *,
    force: bool = False,
) -> Path:
    """Build (or refresh) a viewer directory. Returns path to index HTML.

    ``force`` re-runs transcription even if a cached ``transcript.json``
    with the same audio name is already present.
    """
    audio_src = Path(audio_src).expanduser().resolve()
    if not audio_src.exists():
        raise FileNotFoundError(f"audio not found: {audio_src}")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_copy = out_dir / audio_src.name
    if audio_copy.resolve() != audio_src:
        shutil.copy2(audio_src, audio_copy)

    transcript_path = out_dir / "transcript.json"
    data = None
    if not force and transcript_path.exists():
        try:
            cached = json.loads(transcript_path.read_text(encoding="utf-8"))
            if (
                Path(cached.get("audio", "")).name == audio_src.name
                and cached.get("model") == model_name
            ):
                print(f"[viewer] reusing {transcript_path}", file=sys.stderr)
                data = cached
        except Exception:
            data = None
    if data is None:
        data = _transcribe_with_words(audio_src, model_name, language)
        transcript_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    figure_png = out_dir / "viewer_figure.png"
    figsize = _build_figure(data, audio_copy, figure_png)

    index_html = out_dir / "index.html"
    _build_html(data, audio_copy.name, figure_png.name, figsize, index_html)
    return index_html
