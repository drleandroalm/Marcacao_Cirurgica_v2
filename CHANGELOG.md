# Changelog

## v1.1.0 — History, Export, and Parser Fixes

- Added a new “Histórico” tab with persistent sessions and browsing features:
  - Chronological sections, search, entity filters, delete.
  - Session detail with copy/share.
  - “Clear All” with confirmation.
- Bulk export for history (CSV/JSON) with anonymization toggle.
- PHI/PII sanitization in logs (redacted summaries only).
- Unified export pipeline — `FormExporter` delegates to `SurgicalRequestForm.generateFilledTemplate()`.
- Parser improvements:
  - Portuguese date phrase handling (e.g., “vinte e sete de setembro de 2024”).
  - Weekday phrases (e.g., “próxima segunda”).
  - Duration phrases (e.g., “uma hora e meia”).
- New unit tests for time/duration edge cases.
- Build/test configuration updated for command‑line runs.

