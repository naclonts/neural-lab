# Historical experiment export

This is a partial snapshot of locally retained experiment outputs, collected on 2026-11-07 for review. `runs/<queue-id>/` preserves the original queue output paths and identifiers. `result.json` contains reported run metrics where available; compact `details.json` files and NL-001 training reports provide additional configurations, per-fit measurements and context. A result file alone does not establish that a run completed or passed its scientific audit. Read the accompanying details and scientist notes before drawing conclusions.

The five `scientist-notes/agent-*-session.md` files are historical handoffs and reviews, not live status. Their references to local artifacts may not resolve in this export. Results and notes reflect their authors' recorded interpretations, including limitations and failed or incomplete runs.

The export includes 49 result files, 25 details files, eight training reports and five notes. It is not a complete archive of every historical experiment: runs without these files are absent, details over 50 KB and other reports were not exported, and many notes remain local. Checkpoints, datasets, model weights, raw logs, large JSON files and queue state are excluded. Selected files were screened for common credential and personal-path patterns; this is not a comprehensive audit of every possible sensitive value.
