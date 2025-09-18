# Repository Guidelines

## Project Structure & Module Organization
SwiftTranscriptionSampleApp/ hosts the production code; mirror its folders when adding new files. Use `Helpers/` for non-UI services (EntityExtractor, TranscriptionProcessor), `Models/` for value types like `SurgicalRequestForm`, `Views/` for SwiftUI scenes, and `Recording and Transcription/` for audio capture (`Recorder.swift`) and speech (`Transcription.swift`). Keep demo scripts such as `TestEntityExtraction.swift` in the repository root and document temporary tooling under `Tools/`.

## Build, Test, and Development Commands
Open the project in Xcode with `open SwiftTranscriptionSampleApp.xcodeproj`. Build the simulator target via `xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj -scheme SwiftTranscriptionSampleApp -sdk iphonesimulator build`. Run the automated suite with `xcodebuild test -project SwiftTranscriptionSampleApp.xcodeproj -scheme SwiftTranscriptionSampleApp -destination 'platform=iOS Simulator,name=iPhone 16 Pro'`. Ensure Xcode 26+ and the iOS 26.0 SDK are installed before running these commands.

## Coding Style & Naming Conventions
Write Swift 6 with 4-space indentation and keep lines under ~120 characters. Name types with PascalCase and functions, vars, and constants with camelCase (`let transcriptionProcessor`). Prefer structs and enums for models, avoid force-unwraps, and annotate UI-facing APIs with `@MainActor`. Place new helpers in `Helpers/` rather than views, and keep extensions near their primary type.

## Testing Guidelines
Use XCTest, mirroring the source folder layout under a Tests target. Name files `ThingTests.swift` and methods `test_WhenInputProvided_ShouldProduceExpectedOutput()`. Focus coverage on parsing, formatting, and entity extraction paths (e.g., `MilitaryTimeFormatter.format(_:)`). Run `xcodebuild test ...` before submitting changes and document any gaps.

## Commit & Pull Request Guidelines
Write commits with short imperative subjects and optional bodies describing scope and rationale, such as `feat(extraction): improve time phrase handling in pt-BR`. Link issues using `Fixes #123` where relevant. Pull requests must include a concise summary, screenshots or GIFs for UI updates, test commands with outcomes, and explicit risk/rollback notes.

## Security & Configuration Tips
Never log PHI/PII; strip patient identifiers from debug output. Maintain microphone and speech permission texts in Portuguese within `Info.plist`, and avoid adding network calls without prior discussion.

## Agent-Specific Instructions
Respect the existing folder boundaries; do not move or rename public APIs without consensus. Favor focused, minimal diffs. Propose tooling additions (SwiftLint, SwiftFormat) in separate PRs that include configuration and rationale.
