# Recommended Next Steps

- Profile the recording and extraction pipeline with Instruments (Time Profiler + os_signpost) to pinpoint latency around `Recorder.record()`, `TranscriptionProcessor.processText`, and `EntityExtractor.extractEntities`.
- Offload heavy text normalization and knowledge-base enrichment from `SpokenWordTranscriber.processTranscriptionResult` into background tasks and debounce continuous extraction triggers to reduce blocking on the main actor.
- Cache speech assets, analyzer formats, and language-model sessions across runs via lightweight session managers so `setUpTranscriber()` skips repeated setup when locale settings are unchanged.
- Add unit tests for KB‑assisted fallback (surgeon/procedure n‑grams) and phone parsing edge cases (mixed separators; 8–9 digits capture).
- Extract a dedicated AbbreviationExpander helper and pre‑compile regexes to reduce per‑call overhead in `EntityExtractor`.
- Consider parallelizing fallback sub‑parsers (phone/date/time/name) using task groups when running off the main actor.
- Author a DocC overview article describing the capture → transcription → extraction → review flow and add `///` comments to primary entry points (`FormFillerView`, `Recorder`, `SpokenWordTranscriber`, `EntityExtractor`) for richer generated docs.
- Replace PHI-bearing debug logs with sanitized summaries (counts, hashed IDs) and centralize logging through a `RedactedLogger` that defaults to redaction.
- Add XCTest targets covering `TranscriptionProcessor`, `ComplianceValidator`, and matcher helpers using realistic fixtures to lock in formatting and validation behavior before performance tuning.
- Externalize surgeon and procedure datasets into JSON resources with documented curation, and load them through a cached `MedicalKnowledgeBase` adapter to simplify updates.
- Introduce configuration structs (e.g., `RecordingConfig`, `ExtractionConfig`) injected into `Recorder` and `EntityExtractor` to expose toggles like `shouldWriteToDisk`, confidence thresholds, and model timeouts for future modes and testing.
