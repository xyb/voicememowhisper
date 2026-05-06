# Voice Memo Whisper

This project watches the local Apple Voice Memos library and feeds new recordings to WhisperKit for transcription. It is designed to run locally on macOS so recordings never leave the machine.

Optionally, the included **speaker-ID pipeline** (`voicememo-whisper si`) replaces the plain transcription path with a multi-speaker flow: transcribe (faster-whisper) → diarize (pyannote) → identify (against a local speaker library) → merge → render. When enabled, transcripts ship with per-turn speaker labels and a ready-to-review `transcript.md`.

## Features

- **Automatic Transcription**: Watches for new Voice Memos and transcribes them using WhisperKit.
- **Audio Archiving**: Optionally copies the original `.m4a` files to a separate directory (`--archive`), allowing you to safely delete them from the Voice Memos app to free up storage space while keeping a backup.
- **Inbox Import**: Optionally process external audio dropped into an Inbox directory (e.g. from iOS) and move it into the archive before transcribing.
- **Listing**: The `--list` command lists all recordings with their transcription and archiving status.
- **Speaker-ID pipeline** (optional): in-process 5-stage pipeline exposed as `voicememo-whisper si`, with stage caching and partial-range re-runs (`si run --from merge --to render`).

## Setup

```bash
brew install whisperkit-cli
```

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The Brew formula installs the WhisperKit CLI and downloads models on demand. The editable install adds this watcher CLI into your virtual environment.

### Optional dependency: watchdog

Continuous watching mode (`--watch`) uses `watchdog`. If you only use one-off runs or `--list`, you do not need it.

```bash
python -m pip install watchdog
```

### Optional extra: speaker-id pipeline

Install once with the `[speaker-id]` extra to pull the ML stack (faster-whisper, pyannote.audio, torch, …):

```bash
python -m pip install -e ".[speaker-id]"
```

Without this extra, `voicememo-whisper` keeps working on the WhisperKit-only path; the speaker-ID subcommands are still available for `--help` / `steps` / `library list` / `inspect`, but any step that actually crunches audio will refuse to run.

## Usage

```bash
# One-off backfill (transcribes existing recordings then exits)
voicememo-whisper

# Continuous mode (keep running and watch for new recordings)
voicememo-whisper --watch

# Disable archiving (archiving is enabled by default)
voicememo-whisper --no-archive

# Inspect processed recordings
voicememo-whisper --list
```

### Options

- `-v`: Increase verbosity (shows startup info like paths/models).
- `-vv`: Debug verbosity (shows extra details like skipped files).
- `--model`: Pick a specific WhisperKit model (default `large-v3-v20240930_turbo`).
- `--language`: Hint the spoken language (`en`, `zh`, etc.).
- `-l/--list`: List recordings and exit.
- `-n/--limit`: For `--list`, number of items to show (default: 10; `0` for all).
- `--archive/--no-archive`: Enable/disable archiving of processed audio files (default: enabled).
- `--archive-dir`: Specify directory for archived audio (defaults to `~/Documents/VoiceMemoWhisper/Audio`).
- `--transcript-dir`: Specify directory for transcripts (defaults to `~/Documents/VoiceMemoWhisper/Transcripts`).

The `--list` command provides a unified view of your recordings:
- Shows transcription (`T`) and archiving (`A`) status.
- Indicates if the source file still exists in Voice Memos (`S`).
- Aggregates all files into a unified list, displaying metadata (Title, Date) including from archived files even if the source is deleted from the App.
- By default, only the most recent 10 items are shown. The header shows `Title (shown/total)`.

Example output:

```
/-- Transcribed
|/-- Archived
||/-- Source Exists
TAS  When                 Duration  Title (3/3)
✓✓✓  2025-12-15 16:46:04  70m11s    Sample Recording 1
✓✓✓  2025-12-14 14:19:53  92m48s    Sample Recording 2
✓✓x  2025-12-13 10:11:16  -         Sample Recording 3 (source deleted)
```

### Speaker-ID pipeline (`si`) usage

Five stages — each can run alone, and `run` plays any contiguous range:

```bash
# Full pipeline (stage cache reused unless --force)
voicememo-whisper si run /path/to/recording.m4a

# Re-run merge + render after tweaking rules (upstream cached)
voicememo-whisper si run /path/to/recording.m4a --from merge --to render --force

# Single stage (always forces re-run)
voicememo-whisper si render /path/to/recording.m4a

# Ordered list of stages (position shown in each subcommand's --help too)
voicememo-whisper si steps

# Which stages have cached output for a recording?
voicememo-whisper si inspect /path/to/recording.m4a

# Speaker library management
voicememo-whisper si library list
voicememo-whisper si library add <speaker> /path/to/clip.wav
voicememo-whisper si library rebuild --speaker <speaker>
```

Default paths (overridable via env):
- Speaker library → `~/Documents/VoiceMemoWhisper/speaker-library/` (durable; co-located with Audio/ and Transcripts/ so a single backup of that folder covers everything hand-curated)
- Stage intermediates (runs) → `~/.local/share/voicememowhisper/speaker-id/runs/` (cache; rebuildable)
- Final outputs → `~/.local/share/voicememowhisper/speaker-id/outputs/` (cache; rebuildable)

## Data Locations

By default, the tool organizes outputs under `~/Documents/VoiceMemoWhisper/`:
- **Transcripts**: `~/Documents/VoiceMemoWhisper/Transcripts/`
- **Archived Audio**: `~/Documents/VoiceMemoWhisper/Audio/` (when `--archive` is enabled)
- **Inbox**: `~/Documents/VoiceMemoWhisper/Inbox/` (when `VOICE_MEMO_INBOX_DIR` is set or using default)

A state database tracks processed files to avoid duplication. It is stored at `~/.local/state/voicememowhisper/state.sqlite`.

## Configuration

Override paths or defaults via environment variables:

- `VOICE_MEMO_RECORDINGS_DIR` – directory containing Voice Memo `.m4a` files.
- `VOICE_MEMO_METADATA_DB` – path to `CloudRecordings.db`.
- `VOICE_MEMO_INBOX_DIR` – optional Inbox directory for importing external audio.
- `VOICE_MEMO_TRANSCRIPT_DIR` – where transcripts are stored.
- `VOICE_MEMO_ARCHIVE_DIR` – where audio files are archived.
- `VOICE_MEMO_STATE_DB` – location of the state database.
- `VOICE_MEMO_WHISPERKIT_CLI` – path to `whisperkit-cli`.
- `VOICE_MEMO_WHISPERKIT_MODEL` – WhisperKit model identifier.
- `VOICE_MEMO_LANGUAGE` – language hint.
- `VOICE_MEMO_SPEAKER_PIPELINE` – set to `0` to disable the speaker-ID pipeline even if installed.
- `VOICE_MEMO_SPEAKER_PIPELINE_MODEL` – faster-whisper model name (default `medium`).
- `VOICE_MEMO_SPEAKER_PIPELINE_THRESHOLD` – cosine match threshold against the speaker library.
- `VOICE_MEMO_SPEAKER_LIBRARY_DIR` – path to the speaker library.
- `VOICE_MEMO_SPEAKER_RUNS_DIR` – stage intermediate directory.
- `VOICE_MEMO_SPEAKER_OUTPUT_DIR` – final outputs directory.

## Development

Run the CLI directly from source:

```bash
python -m voicememowhisper --watch
```

Run tests:

```bash
make test
make test-cov
```

## Voice Memo storage recap

**Note:** macOS Gatekeeper protects the Voice Memos container. You must grant the terminal **Full Disk Access** (System Settings → Privacy & Security → Full Disk Access) so the script can read your recordings.

- Recordings live under `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`.
- Metadata is stored in `CloudRecordings.db` (or `Recents.sqlite` on older macOS).
- For detailed file paths, database schemas, and SQL examples, see [docs/technical_details.md](docs/technical_details.md).
