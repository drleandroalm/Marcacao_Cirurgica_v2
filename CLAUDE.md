# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
WWDC25 Session 277 sample app demonstrating SpeechAnalyzer API for advanced speech-to-text with Swift 6 concurrency support. The app records audio, performs live transcription, and uses FoundationModels for title generation.

## Build and Run Commands
```bash
# Build for iOS Simulator
xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj -scheme SwiftTranscriptionSampleApp -sdk iphonesimulator build

# Build for macOS (if supported)
xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj -scheme SwiftTranscriptionSampleApp build

# Clean build folder
xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj clean
```

## Architecture

### Core Audio Pipeline
1. **Recorder.swift** - AVAudioEngine setup with audio tap on input node
   - Manages audio session configuration (iOS only)
   - Streams PCM buffers via AsyncStream
   - Writes WAV files to temporary directory
   - Critical: Audio tap buffer size is 4096 frames

2. **Transcription.swift** - SpeechAnalyzer integration
   - Uses SpeechTranscriber with volatile/finalized results
   - BufferConverter handles format conversion for analyzer
   - AsyncStream pipeline: Audio → AnalyzerInput → SpeechTranscriber
   - FoundationModels integration for title suggestions

3. **BufferConversion.swift** - Audio format conversion
   - Converts between different PCM formats for SpeechAnalyzer compatibility
   - Critical for audio pipeline integrity

### Swift 6 Concurrency Patterns
- **@Observable** classes: `SpokenWordTranscriber`, `Story`
- **Sendable** conformance on transcriber
- **AsyncStream** for audio buffer streaming
- **Task** for concurrent operations (transcription, title generation)
- Audio tap callback bridges to async context via continuation

### Key Dependencies
- **Speech** framework - SpeechAnalyzer, SpeechTranscriber
- **FoundationModels** - SystemLanguageModel for title generation
- **AVFoundation** - Audio recording and processing
- **SwiftData** - Data persistence (imported but not actively used)

## Critical Implementation Details
- Microphone permission required (`AVAudioSession` authorization)
- Locale: English (US) for transcription model
- Audio format negotiated via `SpeechAnalyzer.bestAvailableAudioFormat`
- Temporary WAV files stored in `FileManager.default.temporaryDirectory`
- Volatile transcripts shown in purple (0.4 opacity) during transcription

## Platform Considerations
- iOS-specific: `AVAudioSession` setup in `setUpAudioSession()`
- macOS: No audio session configuration needed
- Both platforms: Same core transcription pipeline