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
    from .._lock import acquire_compute_lock

    acquire_compute_lock(what=f"faster-whisper ({model_name})")
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


_PX_PER_SECOND = 120     # viewer zoom — how many pixels 1 second of audio takes
_TILE_SECONDS = 30       # each matplotlib tile covers this much time
_TILE_HEIGHT_PX = 250    # fixed tile height. ~1/3 of the original 750 —
                         # waveform + spectrogram still readable, a lot
                         # less vertical real estate.
_DPI = 150
_OVERVIEW_HEIGHT_PX = 80


def _render_tile(
    samples,
    sr: int,
    t_start: float,
    t_end: float,
    words_in_window: list[dict],
    all_word_indices: list[int],
    out_png: Path,
    *,
    with_xaxis: bool,
) -> tuple[int, int]:
    """Render one time-window of the waveform + spectrogram as a PNG tile.

    ``all_word_indices`` is the global index of each word in
    ``words_in_window`` so band colors stay consistent across tiles.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import spectrogram

    _configure_mpl_fonts()

    tile_seconds = t_end - t_start
    tile_w_px = int(tile_seconds * _PX_PER_SECOND)
    tile_w_in = tile_w_px / _DPI
    tile_h_in = _TILE_HEIGHT_PX / _DPI

    # Slice samples for this tile window
    i0 = int(t_start * sr)
    i1 = int(t_end * sr)
    seg_samples = samples[i0:i1]
    t_seg = np.arange(len(seg_samples)) / sr + t_start

    fig = plt.figure(figsize=(tile_w_in, tile_h_in), dpi=_DPI)
    ax_wave = fig.add_axes([0.0, _WAVE_BOTTOM, 1.0, _WAVE_TOP - _WAVE_BOTTOM])
    ax_spec = fig.add_axes([0.0, _SPEC_BOTTOM, 1.0, _SPEC_TOP - _SPEC_BOTTOM])

    ax_wave.plot(t_seg, seg_samples, color="#1f2937", linewidth=0.5)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_xlim(t_start, t_end)
    ax_wave.set_xticks([])
    ax_wave.set_yticks([])

    if len(seg_samples) > 0:
        f, t, Sxx = spectrogram(seg_samples, fs=sr, nperseg=512, noverlap=384)
        Sdb = 10 * np.log10(Sxx + 1e-10)
        vmin = np.percentile(Sdb, 5)
        vmax = np.percentile(Sdb, 99)
        ax_spec.pcolormesh(t + t_start, f, Sdb, shading="auto",
                           cmap="magma", vmin=vmin, vmax=vmax)
    ax_spec.set_ylim(0, 4000)
    ax_spec.set_xlim(t_start, t_end)
    # Hide ticks — the HTML layer renders its own time ruler if needed
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])

    cmap = plt.get_cmap("tab20")
    for w, gi in zip(words_in_window, all_word_indices):
        r, g, b, _ = cmap(gi % cmap.N)
        wave_color = (r, g, b, 0.28)
        spec_color = (r, g, b, 0.16)
        # Clamp band to tile edges so a word that straddles two tiles
        # shows on both without overflow.
        ws = max(w["start"], t_start)
        we = min(w["end"], t_end)
        if we <= ws:
            continue
        ax_wave.axvspan(ws, we, facecolor=wave_color,
                        edgecolor="none", zorder=-1)
        ax_spec.axvspan(ws, we, facecolor=spec_color,
                        edgecolor="none", zorder=3)
        for x in (w["start"], w["end"]):
            if t_start <= x <= t_end:
                ax_wave.axvline(x, color="#6b7280", linewidth=0.4, alpha=0.5)
                ax_spec.axvline(x, color="#ffffff", linewidth=0.4, alpha=0.4)

    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return tile_w_px, _TILE_HEIGHT_PX


def _build_tiles(data: dict, audio: Path, out_dir: Path) -> list[dict]:
    """Render the full audio as a series of equal-time PNG tiles.

    Returns a list of {"file": name, "start": float, "end": float,
    "width_px": int, "height_px": int}.
    """
    sr, samples = _load_audio_mono_16k(audio)
    duration = len(samples) / sr

    words: list[dict] = []
    for seg in data["segments"]:
        words.extend(seg.get("words", []))

    # Precompute tile boundaries; absorb any tail shorter than 1s into the
    # previous tile so we don't try to run spectrogram on <512 samples.
    boundaries: list[float] = [0.0]
    t = _TILE_SECONDS
    while t < duration:
        boundaries.append(t)
        t += _TILE_SECONDS
    boundaries.append(duration)
    # Merge a too-short final segment into its predecessor.
    if len(boundaries) >= 3 and (boundaries[-1] - boundaries[-2]) < 1.0:
        boundaries.pop(-2)

    tiles: list[dict] = []
    for tile_idx in range(len(boundaries) - 1):
        t_start = boundaries[tile_idx]
        t_end = boundaries[tile_idx + 1]
        window: list[dict] = []
        window_gi: list[int] = []
        for gi, w in enumerate(words):
            if w["end"] <= t_start or w["start"] >= t_end:
                continue
            window.append(w)
            window_gi.append(gi)
        tile_name = f"tile_{tile_idx:03d}.png"
        tile_path = out_dir / tile_name
        w_px, h_px = _render_tile(
            samples, sr, t_start, t_end, window, window_gi, tile_path,
            with_xaxis=False,
        )
        tiles.append({
            "file": tile_name, "start": t_start, "end": t_end,
            "width_px": w_px, "height_px": h_px,
        })
    return tiles


def _render_overview(
    samples,
    sr: int,
    duration: float,
    words: list[dict],
    out_png: Path,
    width_px: int = 1400,
) -> tuple[int, int]:
    """Compact waveform strip covering the full audio, for navigation."""
    import matplotlib.pyplot as plt
    import numpy as np

    _configure_mpl_fonts()

    w_in = width_px / _DPI
    h_in = _OVERVIEW_HEIGHT_PX / _DPI
    fig = plt.figure(figsize=(w_in, h_in), dpi=_DPI)
    ax = fig.add_axes([0.0, 0.1, 1.0, 0.9])
    t_axis = np.arange(len(samples)) / sr
    ax.plot(t_axis, samples, color="#334155", linewidth=0.3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, duration)
    ax.set_xticks([])
    ax.set_yticks([])

    # Faint color bands for each word so overview echoes main view
    cmap = plt.get_cmap("tab20")
    for i, w in enumerate(words):
        r, g, b, _ = cmap(i % cmap.N)
        ax.axvspan(w["start"], w["end"],
                   facecolor=(r, g, b, 0.25),
                   edgecolor="none", zorder=-1)

    fig.savefig(out_png, dpi=_DPI)
    plt.close(fig)
    return width_px, _OVERVIEW_HEIGHT_PX


def _build_html(
    data: dict,
    audio_name: str,
    tiles: list[dict],
    overview_name: str,
    overview_size_px: tuple[int, int],
    out_html: Path,
) -> None:
    duration = data.get("duration", 1.0) or 1.0
    px_per_second = _PX_PER_SECOND
    timeline_width_px = int(duration * px_per_second)
    tile_height_px = _TILE_HEIGHT_PX

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
        })

    words_js = json.dumps(word_payload, ensure_ascii=False)
    tiles_js = json.dumps(tiles, ensure_ascii=False)

    segments_list_html = "".join(
        f'<li class="seg" data-start="{s["start"]:.3f}">'
        f'<span class="seg-ts">[{s["start"]:.2f}–{s["end"]:.2f}]</span> '
        f'{html.escape(s.get("text", "").strip())}</li>'
        for s in data["segments"]
    )
    audio_escaped = html.escape(audio_name)
    overview_escaped = html.escape(overview_name)
    ov_w, ov_h = overview_size_px
    model_name = html.escape(data.get("model", "?"))

    out_html.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Word-aligned viewer · {audio_escaped}</title>
<style>
  html, body {{ margin: 0; padding: 0; }}
  body {{ font-family: "PingFang SC", "Hiragino Sans GB", system-ui, sans-serif;
          color: #111; background: #fafafa; padding: 1rem 1.25rem 3rem; }}
  h1 {{ font-size: 1rem; margin: 0 0 0.25rem; }}
  .meta {{ color: #555; font-size: 0.85rem; margin-bottom: 0.6rem; }}
  audio {{ width: 100%; margin-bottom: 0.5rem; }}

  /* Overview strip — full audio at a glance. Clicking scrolls the main
     view; the red rectangle shows the current visible window. */
  #overview {{ position: relative; width: 100%; max-width: {ov_w}px;
               margin: 0.3rem auto 0.6rem; cursor: pointer;
               user-select: none; }}
  #overview img {{ display: block; width: 100%; height: auto;
                   border: 1px solid #e5e7eb; border-radius: 3px;
                   aspect-ratio: {ov_w} / {ov_h}; }}
  #overview-window {{ position: absolute; top: 0; bottom: 0;
                      border: 2px solid #ef4444; background: rgba(239,68,68,0.08);
                      pointer-events: none; box-sizing: border-box; }}
  #overview-cursor {{ position: absolute; top: 0; bottom: 0; width: 2px;
                      background: #ef4444; pointer-events: none;
                      display: none; }}
  #overview-cursor.visible {{ display: block; }}

  /* Scrollable main timeline. Its inner content (#track) has a fixed
     pixel width so word labels can be positioned in absolute px. */
  #scroller {{ overflow-x: auto; overflow-y: hidden;
               border: 1px solid #e5e7eb; border-radius: 4px;
               background: #fff; }}
  #track {{ position: relative; height: {tile_height_px + 52}px;
            min-width: 100%; }}
  #labels {{ position: absolute; left: 0; top: 0; right: 0; height: 48px; }}
  #labels .lab {{ position: absolute; transform: translateX(-50%);
                  white-space: nowrap; font-size: 13px; line-height: 1.2;
                  padding: 1px 3px; background: #fff;
                  border: 1px solid #e5e7eb; border-radius: 3px;
                  cursor: pointer; }}
  #labels .lab.low {{ color: #b91c1c; opacity: 0.75; border-style: dashed; }}
  #labels .lab.row0 {{ top: 1px; }}
  #labels .lab.row1 {{ top: 23px; }}
  #labels .lab:hover {{ background: #fef3c7; z-index: 10; }}
  #tiles {{ position: absolute; left: 0; top: 50px;
            height: {tile_height_px}px;
            display: flex; flex-direction: row; }}
  #tiles img {{ display: block; height: 100%; width: auto;
                pointer-events: none; user-select: none; }}
  #cursor {{ position: absolute; top: 0; bottom: 0; width: 2px;
             background: #ef4444; pointer-events: none;
             display: none; z-index: 20; }}
  #cursor.visible {{ display: block; }}
  /* Clickable invisible bands over the tiles area for per-word seek */
  #clickbands {{ position: absolute; left: 0; top: 50px;
                 height: {tile_height_px}px; pointer-events: none; }}

  .legend {{ font-size: 0.75rem; color: #666; margin-top: 8px; }}
  ul {{ color: #444; font-size: 0.9rem; line-height: 1.6;
         padding-left: 1.1rem; }}
  li.seg {{ cursor: pointer; padding: 1px 4px; border-radius: 3px; }}
  li.seg:hover {{ background: #eef2ff; }}
  li.seg.active {{ background: #fef3c7; }}
  .seg-ts {{ color: #888; font-variant-numeric: tabular-nums;
             font-size: 0.82em; margin-right: 2px; }}
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

<div id="overview">
  <img src="{overview_escaped}" alt="full-audio overview">
  <div id="overview-window"></div>
  <div id="overview-cursor"></div>
</div>

<div id="scroller">
  <div id="track" style="width: {timeline_width_px}px">
    <div id="labels"></div>
    <div id="tiles"></div>
    <div id="clickbands"></div>
    <div id="cursor"></div>
  </div>
</div>

<div class="legend" data-i18n="legend"></div>

<h2 style="font-size:0.95rem;margin-top:1.5rem" data-i18n="segments_heading">Segments</h2>
<ul>{segments_list_html}</ul>

<script>
const WORDS = {words_js};
const TILES = {tiles_js};
const DURATION = {duration:.3f};
const PX_PER_SECOND = {px_per_second};
const TIMELINE_WIDTH_PX = {timeline_width_px};

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

const audio = document.getElementById("audio");
const scroller = document.getElementById("scroller");
const track = document.getElementById("track");
const labels = document.getElementById("labels");
const tilesEl = document.getElementById("tiles");
const clickbands = document.getElementById("clickbands");
const cursor = document.getElementById("cursor");
const overview = document.getElementById("overview");
const overviewWindow = document.getElementById("overview-window");
const overviewCursor = document.getElementById("overview-cursor");

function timeToPx(t) {{ return (t / DURATION) * TIMELINE_WIDTH_PX; }}

function layout() {{
  // Tiles — render all PNGs in a flex row at their natural widths
  tilesEl.innerHTML = "";
  TILES.forEach(tile => {{
    const img = document.createElement("img");
    img.src = tile.file;
    img.loading = "lazy";
    img.decoding = "async";
    img.width = tile.width_px;
    img.height = tile.height_px;
    img.alt = `${{tile.start.toFixed(1)}}-${{tile.end.toFixed(1)}}s`;
    tilesEl.appendChild(img);
  }});

  // Word labels + clickable bands
  labels.innerHTML = "";
  clickbands.innerHTML = "";
  WORDS.forEach((w, idx) => {{
    const lab = document.createElement("div");
    const lowProb = (w.prob >= 0 && w.prob < 0.5);
    lab.className = "lab " + (idx % 2 === 0 ? "row0" : "row1") + (lowProb ? " low" : "");
    lab.textContent = w.text;
    lab.title = `${{w.start.toFixed(3)}}–${{w.end.toFixed(3)}} s${{w.prob >= 0 ? "  p=" + w.prob.toFixed(2) : ""}}`;
    const center = (timeToPx(w.start) + timeToPx(w.end)) / 2;
    lab.style.left = center + "px";
    lab.onclick = (e) => {{ e.preventDefault(); audio.currentTime = w.start; audio.play(); }};
    labels.appendChild(lab);

    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.left = timeToPx(w.start) + "px";
    band.style.width = Math.max(1, timeToPx(w.end) - timeToPx(w.start)) + "px";
    band.style.top = "0";
    band.style.bottom = "0";
    band.style.cursor = "pointer";
    band.style.pointerEvents = "auto";
    band.title = `${{w.text}}  ${{w.start.toFixed(3)}}–${{w.end.toFixed(3)}}s`;
    band.onclick = () => {{ audio.currentTime = w.start; audio.play(); }};
    clickbands.appendChild(band);
  }});

  updateOverviewWindow();
}}

function updateCursor() {{
  const t = audio.currentTime;
  if (!isFinite(t) || t < 0) {{
    cursor.classList.remove("visible");
    overviewCursor.classList.remove("visible");
    return;
  }}
  cursor.classList.add("visible");
  cursor.style.left = timeToPx(t) + "px";
  overviewCursor.classList.add("visible");
  overviewCursor.style.left = ((t / DURATION) * overview.clientWidth) + "px";
  // Keep the cursor in view while playing
  if (!audio.paused) {{
    const left = cursor.offsetLeft;
    const vw = scroller.clientWidth;
    const sl = scroller.scrollLeft;
    if (left < sl + 50 || left > sl + vw - 50) {{
      scroller.scrollTo({{ left: Math.max(0, left - vw * 0.3), behavior: "auto" }});
    }}
  }}
}}

function updateOverviewWindow() {{
  // Red rectangle on the overview showing what's currently visible in the scroller.
  const vw = scroller.clientWidth;
  const sl = scroller.scrollLeft;
  const ovW = overview.clientWidth;
  const leftPct = sl / TIMELINE_WIDTH_PX;
  const widthPct = Math.min(1, vw / TIMELINE_WIDTH_PX);
  overviewWindow.style.left = (leftPct * ovW) + "px";
  overviewWindow.style.width = (widthPct * ovW) + "px";
}}

overview.addEventListener("click", (e) => {{
  const rect = overview.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  const t = Math.max(0, Math.min(DURATION, pct * DURATION));
  audio.currentTime = t;
  // Center the scroller on this time
  const targetPx = timeToPx(t);
  scroller.scrollTo({{ left: Math.max(0, targetPx - scroller.clientWidth / 2),
                       behavior: "smooth" }});
}});

scroller.addEventListener("scroll", updateOverviewWindow);

let rafId = null;
function rafLoop() {{ updateCursor(); rafId = requestAnimationFrame(rafLoop); }}
function startRaf() {{ if (rafId == null) rafLoop(); }}
function stopRaf() {{
  if (rafId != null) {{ cancelAnimationFrame(rafId); rafId = null; }}
  updateCursor();
}}

// Segments list — click any row to seek + scroll main view there
document.querySelectorAll("li.seg").forEach(li => {{
  li.addEventListener("click", () => {{
    const t = parseFloat(li.dataset.start);
    if (!isFinite(t)) return;
    audio.currentTime = t;
    audio.play();
    const targetPx = timeToPx(t);
    scroller.scrollTo({{ left: Math.max(0, targetPx - scroller.clientWidth / 2),
                         behavior: "smooth" }});
  }});
}});

// Highlight the segment currently being played
function updateActiveSegment() {{
  const t = audio.currentTime;
  document.querySelectorAll("li.seg").forEach(li => {{
    const s = parseFloat(li.dataset.start);
    li.classList.remove("active");
  }});
  // Find segment containing current time — linear scan; N is small enough
  let activeEl = null;
  document.querySelectorAll("li.seg").forEach(li => {{
    const s = parseFloat(li.dataset.start);
    if (s <= t) activeEl = li;
  }});
  if (activeEl) activeEl.classList.add("active");
}}

window.addEventListener("load", () => {{ layout(); updateOverviewWindow(); }});
window.addEventListener("resize", () => {{ layout(); updateOverviewWindow(); }});
audio.addEventListener("play", startRaf);
audio.addEventListener("pause", stopRaf);
audio.addEventListener("ended", stopRaf);
audio.addEventListener("seeked", () => {{ updateCursor(); updateActiveSegment(); }});
audio.addEventListener("timeupdate", updateActiveSegment);
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

    # Clear old tile PNGs so stale tiles from a previous (different)
    # duration don't leak into the new run.
    for stale in out_dir.glob("tile_*.png"):
        try:
            stale.unlink()
        except OSError:
            pass

    print("[viewer] rendering tiles ...", file=sys.stderr)
    tiles = _build_tiles(data, audio_copy, out_dir)

    print("[viewer] rendering overview ...", file=sys.stderr)
    sr, samples = _load_audio_mono_16k(audio_copy)
    flat_words = []
    for seg in data["segments"]:
        flat_words.extend(seg.get("words", []))
    overview_png = out_dir / "overview.png"
    overview_size = _render_overview(
        samples, sr, data.get("duration", 1.0) or 1.0,
        flat_words, overview_png,
    )

    index_html = out_dir / "index.html"
    _build_html(
        data, audio_copy.name, tiles,
        overview_png.name, overview_size,
        index_html,
    )
    return index_html
