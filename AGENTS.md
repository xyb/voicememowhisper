# Repository Guidelines

## Project Structure & Module Organization
`voicememowhisper/` contains the Python package: `cli.py` exposes the entry point, `service.py` coordinates metadata discovery, transcription, and state tracking, `watcher.py` tails the recordings directory, and `transcribe.py` wraps WhisperKit execution. Supporting modules (`config.py`, `paths.py`, `metadata.py`, `state.py`) centralize environment resolution, filesystem helpers, SQLite access, and processed-record bookkeeping. Temporary assets or scratch fixtures belong under `tmp/`; keep Voice Memo samples out of version control by default.

## Build, Test, and Development Commands
- `brew install whisperkit-cli` downloads the WhisperKit runner and models on demand.
- `/Users/xyb/.virtualenvs/voicememowhisper/bin/python -m pip install -e .` installs the watcher CLI into the shared project virtualenv (activate it via `source /Users/xyb/.virtualenvs/voicememowhisper/bin/activate` when working interactively).
- `voicememo-whisper --list` verifies metadata access; `voicememo-whisper --watch` performs continuous transcription.
- `/Users/xyb/.virtualenvs/voicememowhisper/bin/python -m voicememowhisper --watch` runs the package without relying on the console entry point when iterating on code.

## Coding Style & Naming Conventions
Target Python 3.9+, follow PEP 8 with 4-space indentation, and keep functions small with explicit logging. Use `snake_case` for functions and module-level constants, `PascalCase` for dataclasses, and favour type hints plus dataclasses (see `config.Settings`). Log user-facing events through `logging` instead of prints, and return non-zero exit codes when surfacing failures. Keep CLI help texts concise and sync new flags between `cli.py` and `Settings`.

## Testing Guidelines
Prefer `pytest` for new automated coverage; place tests in `tests/test_<module>.py` mirroring package structure. Add fixture recordings under `tmp/tests/` and point `VOICE_MEMO_RECORDINGS_DIR` there during tests. Invoke tests through `/Users/xyb/.virtualenvs/voicememowhisper/bin/python -m pytest`. Aim to cover metadata parsing edge cases (missing duration, deleted memos), state resumption, and transcription command construction. For manual smoke tests, run `voicememo-whisper --list` followed by `--watch` against a sample library and confirm transcripts land in `~/Documents/VoiceMemoTranscripts/`.

## Commit & Pull Request Guidelines
The history follows Conventional Commits (`type: short summary` such as `chore: trim list output columns`). Keep subject lines ≤72 characters, use the imperative mood, and reference issues with `Fixes #id` when applicable. Pull requests should describe the motivation, outline test evidence (commands, logs, or screenshots), flag required environment tweaks (Full Disk Access, env vars), and call out any migration steps for existing state databases.

## Security & Configuration Tips
Grant the terminal Full Disk Access before running the watcher, and never commit real Voice Memo data or transcripts. Prefer environment variables (`VOICE_MEMO_RECORDINGS_DIR`, `VOICE_MEMO_METADATA_DB`, `VOICE_MEMO_TRANSCRIPT_DIR`, `VOICE_MEMO_STATE_DB`, `VOICE_MEMO_WHISPERKIT_MODEL`) over hard-coded paths, and document overrides inside the PR when they change default behaviour.

## Interaction Retrospective (2025-12-15)

### Critical Errors & Lessons Learned

1.  **Verification Failure (Most Critical)**
    *   **Error**: Committed code multiple times without running the actual shell command (`python -m voicememowhisper --list`) to verify the output. Relied on "mental compilation" which failed to catch `IndentationError`, `NameError`, and logical bugs (sorting, duplication).
    *   **Lesson**: **Always execute the code** in the shell to verify the fix *before* committing. "I think it works" is not enough. Proof is required.

2.  **Git Hygiene**
    *   **Error**: Habitually used `git add .` which staged temporary files like `commit_msg.txt`, requiring multiple `git commit --amend` cleanups.
    *   **Lesson**: Never use `git add .` blindly. Use `git add <file>` for specific files, or ensure the workspace is clean (e.g., delete temp files immediately after use).

3.  **Tool Usage (Text Replacement)**
    *   **Error**: Failed multiple `replace` calls because the `old_string` context didn't match the actual file state (often due to previous edits or indentation).
    *   **Lesson**: When in doubt about the file state, use `read_file` *immediately* before `replace` to get the exact content. Do not guess the indentation.

4.  **Debugging Strategy**
    *   **Error**: Wasted time theorizing about `datetime` comparisons instead of simply printing the data types and values.
    *   **Lesson**: When data processing yields wrong results, **inspect the data** (print types and values) immediately. Don't guess; look.

5.  **Requirement Understanding**
    *   **Error**: Initially misunderstood the requirement for "source deleted" items, thinking DB was the only source of truth, whereas the user wanted the File System to be the ultimate truth for existence, with DB as metadata enrichment.
    *   **Lesson**: Clarify the "Source of Truth" early in data synchronization tasks.