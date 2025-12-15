# Apple Voice Memos Database Schema

## Apple Internal Metadata Schema (CloudRecordings.db)

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
