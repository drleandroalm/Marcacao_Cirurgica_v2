# Technology Stack

## Platform & Requirements
- **iOS**: 26.0+ (requires latest iOS for Foundation Models)
- **Xcode**: 26 Beta or later
- **Swift**: 6.0 with concurrency features
- **Deployment Target**: iOS 26.0+

## Core Frameworks
- **Foundation Models**: iOS 26's SystemLanguageModel for AI entity extraction
- **SpeechAnalyzer**: Apple's speech-to-text transcription API
- **AVFoundation**: Audio recording and buffer management
- **SwiftUI**: Declarative UI framework
- **SwiftData**: Data persistence (if needed)

## Architecture Patterns
- **@Observable**: Swift 6 observation pattern for reactive UI updates
- **AsyncStream**: For real-time audio buffer processing
- **Sendable**: Thread-safe data structures for concurrency
- **Task Groups**: Parallel processing of AI operations

## Key Dependencies
- No external package dependencies - uses only Apple frameworks
- Foundation Models framework (iOS 26 system requirement)
- Portuguese (Brazil) language support required

## Build Commands

### Standard Build
```bash
xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj \
           -scheme SwiftTranscriptionSampleApp \
           -sdk iphonesimulator build
```

### Testing
```bash
xcodebuild test -project SwiftTranscriptionSampleApp.xcodeproj \
                -scheme SwiftTranscriptionSampleApp \
                -destination 'platform=iOS Simulator,name=iPhone 16 Pro'
```

### Quick Development
```bash
# Open project in Xcode
open SwiftTranscriptionSampleApp.xcodeproj

# Build and run (Cmd+R in Xcode)
```

## Performance Targets
- **Transcription Latency**: < 100ms
- **Entity Extraction**: ~500ms per form
- **Memory Usage**: < 150MB peak
- **Confidence Threshold**: 0.92 for entity acceptance