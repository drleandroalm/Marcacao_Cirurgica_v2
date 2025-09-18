# History Export Guide

This document describes how history export (CSV/JSON) works, including anonymization and field mapping.

## Where to Export

- Open the app → “Histórico” tab.
- Tap the share icon (top-right) to open the export sheet.
- Choose format (CSV or JSON) and switch “Anonimizar” on/off.
- Tap “Exportar” and share the generated file.

## Anonymization

- When enabled, patient identifiers are excluded from exports.
  - CSV: `patient` column is omitted; flags and metadata remain.
  - JSON: patient‑identifying fields (e.g., `patientName`, templated text) are excluded.
  - Surgeon and procedure remain for operational analytics unless further redaction is required.

## CSV Schema

Anonymized columns:
- `id, createdAt, surgeon, procedure, date, time, needsCTI, needsOPME, needsHem`

Full columns:
- `id, createdAt, patient, surgeon, procedure, date, time, needsCTI, needsOPME, needsHem`

- Booleans are encoded as `1` or `0`.
- `createdAt` is ISO‑8601.

## JSON Schema

Anonymized entry:
```json
{
  "id": "UUID",
  "createdAt": "ISO-8601",
  "surgeonName": "...",
  "procedureName": "...",
  "surgeryDate": "dd/MM/yyyy",
  "surgeryTime": "HH:MM",
  "needsCTI": true,
  "needsOPME": false,
  "needsHemocomponents": false
}
```

Full entry adds:
```json
{
  "patientName": "...",
  "hemocomponentsSpecification": "...",
  "exportedTemplate": "SOLICITAÇÃO..."
}
```

## Storage

- History persists in Documents as `surgery_sessions.json` via `SessionStore`.
- Bulk exports are written to a temporary file and passed to the share sheet.

