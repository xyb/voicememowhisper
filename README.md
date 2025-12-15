# Voice Memo Whisper

This project watches the local Apple Voice Memos library and feeds new recordings to WhisperKit for transcription. It is designed to run locally on macOS so recordings never leave the machine.

## Features

- **Automatic Transcription**: Watches for new Voice Memos and transcribes them using WhisperKit.
- **Audio Archiving**: Optionally copies the original `.m4a` files to a separate directory (`--archive`), allowing you to safely delete them from the Voice Memos app to free up storage space while keeping a backup.
- **Listing**: The `--list` command lists all recordings with their transcription and archiving status.

## Setup

```bash
brew install whisperkit-cli
```

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The Brew formula installs the WhisperKit CLI and downloads models on demand. The editable install adds this watcher CLI into your virtual environment.

## Usage

```bash
# One-off backfill (transcribes existing recordings then exits)
voicememo-whisper

# Continuous mode (keep running and watch for new recordings)
voicememo-whisper --watch

# Archive audio files after transcription (copies original .m4a)
voicememo-whisper --archive

# Inspect processed recordings
voicememo-whisper --list
```

### Options

- `--model`: Pick a specific WhisperKit model (default `large-v3-v20240930_turbo`).
- `--language`: Hint the spoken language (`en`, `zh`, etc.).
- `--archive`: Enable archiving of processed audio files.
- `--archive-dir`: Specify directory for archived audio (defaults to `~/Documents/VoiceMemoWhisper/Audio`).
- `--transcript-dir`: Specify directory for transcripts (defaults to `~/Documents/VoiceMemoWhisper/Transcripts`).

The `--list` command provides a unified view of your recordings:
- Shows transcription (`T`) and archiving (`A`) status.
- Indicates if the source file still exists in Voice Memos (`S`).
- Aggregates all files into a unified list, displaying metadata (Title, Date) including from archived files even if the source is deleted from the App.

Example output:

```
/-- Transcribed
|/-- Archived
||/-- Source Exists
TAS  When                 Duration  Title
✓✓✓  2025-12-15 16:46:04  70m11s    Sample Recording 1
✓✓✓  2025-12-14 14:19:53  92m48s    Sample Recording 2
✓✓x  2025-12-13 10:11:16  -         Sample Recording 3 (source deleted)
```

## Data Locations

By default, the tool organizes outputs under `~/Documents/VoiceMemoWhisper/`:
- **Transcripts**: `~/Documents/VoiceMemoWhisper/Transcripts/`
- **Archived Audio**: `~/Documents/VoiceMemoWhisper/Audio/` (when `--archive` is enabled)

A state database tracks processed files to avoid duplication. It is stored at `~/.local/state/voicememowhisper/state.sqlite`.

## Configuration

Override paths or defaults via environment variables:

- `VOICE_MEMO_RECORDINGS_DIR` – directory containing Voice Memo `.m4a` files.
- `VOICE_MEMO_METADATA_DB` – path to `CloudRecordings.db`.
- `VOICE_MEMO_TRANSCRIPT_DIR` – where transcripts are stored.
- `VOICE_MEMO_ARCHIVE_DIR` – where audio files are archived.
- `VOICE_MEMO_STATE_DB` – location of the state database.
- `VOICE_MEMO_WHISPERKIT_CLI` – path to `whisperkit-cli`.
- `VOICE_MEMO_WHISPERKIT_MODEL` – WhisperKit model identifier.
- `VOICE_MEMO_LANGUAGE` – language hint.

## Development

Run the CLI directly from source:

```bash
python -m voicememowhisper --watch
```

## Voice Memo storage recap

**Note:** macOS Gatekeeper protects the Voice Memos container. You must grant the terminal **Full Disk Access** (System Settings → Privacy & Security → Full Disk Access) so the script can read your recordings.

- Recordings live under `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`.
- Metadata is stored in `CloudRecordings.db` (or `Recents.sqlite` on older macOS).
- For detailed file paths, database schemas, and SQL examples, see [docs/technical_details.md](docs/technical_details.md).
