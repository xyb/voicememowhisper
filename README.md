# Voice Memo Whisper

This project watches the local Apple Voice Memos library and feeds new recordings to WhisperKit for transcription. It is designed to run locally on macOS so recordings never leave the machine.

## Voice Memo storage recap

- Recordings live under `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`.
- Metadata (title, creation date, etc.) is stored in `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/CloudRecordings.db`; older macOS builds may instead expose only `Recents.sqlite`.
- macOS Gatekeeper protects this container. Grant the terminal full disk access (System Settings → Privacy & Security → Full Disk Access) so the script can read it.

## Setup

```bash
brew install whisperkit-cli
```

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For a shared environment across contributors, you can also reuse `/Users/xyb/.virtualenvs/voicememowhisper`:

```bash
python -m venv /Users/xyb/.virtualenvs/voicememowhisper
source /Users/xyb/.virtualenvs/voicememowhisper/bin/activate
python -m pip install -e .
```

The Brew formula installs the WhisperKit CLI and downloads models on demand. The editable install adds this watcher CLI into your virtual environment. If no recording path is provided, the tool automatically targets the system Voice Memos library under `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings` and reads metadata from `CloudRecordings.db` in the same container (Full Disk Access required).

## Usage

```bash
# One-off backfill (scans existing recordings then exits)
voicememo-whisper

# Continuous mode (keep running and watch for new recordings)
voicememo-whisper --watch

# Inspect available recordings
voicememo-whisper --list
```

The CLI accepts `--model` to pick a specific WhisperKit model (default `large-v3-v20240930_turbo`) and `--language` to hint the spoken language (`en`, `zh`, etc.). The first transcription run downloads the model to WhisperKit’s cache.

Transcripts are written to `~/Documents/VoiceMemoTranscripts/<timestamp>-<title>.txt`. A state database at `~/.voice-memo-whisper/state.sqlite` keeps track of processed recordings so reruns do not duplicate work.

The `--list` mode reads Voice Memos metadata from `CloudRecordings.db`, so the output matches the titles, timestamps, and durations shown in the macOS/iOS app.

## Metadata schema notes

- Primary table `ZCLOUDRECORDING` lives at `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/CloudRecordings.db`.
  - `ZUNIQUEID`: recording GUID, matches the `.m4a` file at `Recordings/<ZPATH>`.
  - `ZDATE`: recording time (seconds since 2001-01-01 UTC); convert with SQL `datetime(ZDATE + 978307200, 'unixepoch', 'localtime')`.
  - `ZDURATION` / `ZLOCALDURATION`: duration in seconds.
  - `ZENCRYPTEDTITLE`: title shown in the app (newer systems store the plaintext title despite the name).
  - `ZCUSTOMLABEL` / `ZCUSTOMLABELFORSORTING`: user-entered label; falls back to `ZENCRYPTEDTITLE` when empty.
  - `ZPATH`: original filename (usually `timestamp.m4a`).
  - `ZEVICTIONDATE`: non-null means the recording was moved to Recently Deleted (convertible to a deletion time).
- Supporting tables such as `ZFOLDER` are rarely used; when `CloudRecordings` is absent, older systems fall back to the `ZVOICE` table in `Recents.sqlite`.

Example query (Terminal needs Full Disk Access):

```bash
sqlite3 -separator ' | ' "file:$HOME/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings/CloudRecordings.db?mode=ro" "
SELECT
  Z_PK,
  datetime(ZDATE + 978307200, 'unixepoch', 'localtime') AS RecordedAt,
  printf('%.1f', ZDURATION) AS Duration_s,
  ZENCRYPTEDTITLE AS DisplayTitle,
  ZCUSTOMLABEL,
  ZPATH,
  ZUNIQUEID,
  CASE WHEN ZEVICTIONDATE IS NULL THEN 0 ELSE 1 END AS IsDeleted,
  CASE WHEN ZEVICTIONDATE IS NULL THEN '' ELSE datetime(ZEVICTIONDATE + 978307200, 'unixepoch', 'localtime') END AS DeletedAt
FROM ZCLOUDRECORDING
ORDER BY ZDATE DESC;
"
```

## Configuration

Override paths or defaults via environment variables:

- `VOICE_MEMO_RECORDINGS_DIR` – directory containing Voice Memo `.m4a` files.
- `VOICE_MEMO_METADATA_DB` – path to `CloudRecordings.db` (or `Recents.sqlite` on older macOS builds).
- `VOICE_MEMO_LEGACY_METADATA_DB` – optional override for the fallback (`Recents.sqlite`) if you need to point at a different legacy database.
- `VOICE_MEMO_TRANSCRIPT_DIR` – where transcripts are stored.
- `VOICE_MEMO_STATE_DB` – location of the state database.
- `VOICE_MEMO_WHISPERKIT_CLI` – override the path to the `whisperkit-cli` executable.
- `VOICE_MEMO_WHISPERKIT_MODEL` – default WhisperKit model identifier.
- `VOICE_MEMO_WHISPERKIT_ARGS` – extra CLI arguments (e.g. `--without-timestamps`).
- `VOICE_MEMO_LANGUAGE` – hint the spoken language for transcription.

## Development

Run the CLI directly from source without installing:

```bash
python -m voicememowhisper --watch
```

Press `Ctrl+C` to stop the watcher.
