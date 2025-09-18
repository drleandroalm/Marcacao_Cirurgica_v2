# ``SwiftTranscriptionSampleApp``

SwiftTranscriptionSampleApp demonstrates an on-device workflow that turns dictated surgical briefings into a validated, shareable request form. The app keeps the entire pipeline local—audio capture, transcription, entity extraction, and compliance review—so sensitive data never leaves the device.

## Capture → Transcription → Extraction → Review

1. ``FormFillerView`` lets the clinician choose between field-by-field and continuous capture while wiring up configuration toggles such as disk persistence or deferred processing.
2. ``Recorder`` streams microphone audio into ``SpokenWordTranscriber`` and optionally keeps a temporary waveform so recordings can be replayed during QA.
3. ``SpokenWordTranscriber`` configures the iOS 26 Speech framework, normalises intermediate text with ``TranscriptionProcessor``, and publishes partial or final transcripts back to the form.
4. ``EntityExtractor`` calls the on-device Foundation Model, enriches results with ``MedicalKnowledgeBase`` datasets loaded from ``Resources/KnowledgeBase`` JSON files, and falls back to deterministic parsing when coverage is low.
5. ``FormPreviewView`` and compliance tooling (``ComplianceValidator`` and ``WhitelistEntityValidator``) highlight confidence levels, let the user refine fields, and drive the CTI / hemocomponent decision popups before export.

## Configuration Hooks

- ``RecordingConfiguration`` toggles disk persistence, input buffer size, and stream backlog for ``Recorder``.
- ``TranscriptionConfiguration`` exposes mode, auto-processing thresholds, and locale overrides for ``SpokenWordTranscriber``.
- ``ExtractionConfiguration`` lets callers tune timeouts, required field coverage, and logging behaviour before handing work to ``EntityExtractor``.

## Maintaining the Knowledge Base

Curated surgeon, procedure, and abbreviation data lives in the JSON files inside *Resources/KnowledgeBase*. Updates only require editing those artifacts; ``MedicalKnowledgeBase`` automatically reloads the decoded structures at launch. This design keeps domain knowledge editable without touching production code and makes it easier to share curation steps with clinical stakeholders.

## Privacy & Export

- Logs avoid PHI/PII by emitting redacted summaries only (for example, value lengths rather than raw content).
- Export functions delegate to ``SurgicalRequestForm/generateFilledTemplate()`` so the final output always mirrors the app’s canonical template and post‑transcription decisions (CTI, hemocomponents, OPME).

## History & Browsing

- Accepted sessions (after pre‑approval) are archived locally and presented in a dedicated “Histórico” tab.
- The history UI supports chronological sections, entity filters (cirurgião, procedimento), full‑text search (paciente, cirurgião, procedimento), and deletion.
- Persistence is implemented by ``SessionStore`` which writes JSON to the user Documents directory. No network transmission is performed.

### History Export

- The history screen supports bulk export of all sessions as CSV or JSON, with an anonymization toggle to remove patient identifiers from exported artifacts.
- CSV includes surgeon, procedure, date/time, and decision flags; JSON mirrors those fields and may include patient data when not anonymized.
