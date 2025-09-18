# Project Structure

## Root Directory
```
SwiftTranscriptionSampleApp/
├── SwiftTranscriptionSampleApp.xcodeproj/    # Xcode project files
├── SwiftTranscriptionSampleApp/              # Main app source code
├── Template.txt                              # Form template reference
├── TestEntityExtraction.swift               # Test utilities
└── README.md                                # Project documentation
```

## App Source Organization

### `/Views` - UI Layer
- **ContentView.swift**: Main app entry point, routes to FormFillerView
- **FormFillerView.swift**: Primary form interface with dual recording modes
- **FieldTranscriptionView.swift**: Individual field transcription UI
- **FormPreviewView.swift**: Review extracted data before confirmation
- **PostTranscriptionDecisionsView.swift**: CTI/precaution decision interface

### `/Models` - Data Layer
- **SurgicalRequestForm.swift**: Main form model with @Observable pattern
- **TemplateModel.swift**: Form field definitions and structure
- **StoryModel.swift**: Legacy model from original WWDC sample

### `/Helpers` - Business Logic & Services
- **EntityExtractor.swift**: AI-powered entity extraction using Foundation Models
- **WhitelistEntityValidator.swift**: 99.9% accuracy validation for known entities
- **IntelligentMatcher.swift**: Fuzzy matching algorithms for medical terms
- **TranscriptionProcessor.swift**: Text processing and formatting
- **MilitaryTimeFormatter.swift**: Portuguese time expression conversion
- **OPMEConfiguration.swift**: Medical equipment requirement rules
- **ComplianceValidator.swift**: Medical compliance checking
- **FormExporter.swift**: Export functionality (JSON/text/clipboard)
- **MedicalKnowledgeBase.swift**: Surgeon and procedure databases
- **SwiftTranscriptionSampleAppApp.swift**: App initialization

### `/Recording and Transcription` - Audio Layer
- **Recorder.swift**: Audio capture and buffer management
- **Transcription.swift**: SpeechAnalyzer integration and real-time processing

### `/Helpers/Assets.xcassets` - Resources
- App icons, accent colors, and visual assets

## Architectural Layers

### Presentation Layer
SwiftUI views handle user interaction and display transcription results with confidence indicators.

### Business Logic Layer  
Entity extraction, validation, and medical knowledge processing. Handles Portuguese language specifics and medical terminology.

### Core Services Layer
Audio recording, speech transcription, and Foundation Models AI processing.

## File Naming Conventions
- Views: Descriptive names ending in "View.swift"
- Models: Domain objects ending in "Model.swift" or descriptive names like "SurgicalRequestForm.swift"
- Helpers: Functional names describing purpose (e.g., "EntityExtractor.swift")
- Use PascalCase for Swift files and classes