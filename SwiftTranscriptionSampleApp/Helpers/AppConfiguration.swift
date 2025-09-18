import Foundation
import AVFoundation

struct RecordingConfiguration: Sendable {
    var shouldWriteToDisk: Bool = false
    var bufferSize: AVAudioFrameCount = 4096
    var streamBacklogDepth: Int = 8
}

struct TranscriptionConfiguration: Sendable {
    var isContinuousMode: Bool = false
    var processAfterRecording: Bool = true
    var autoProcessCharacterCount: Int = 100
    var autoProcessWordCount: Int = 15
    var confidenceThreshold: Double = 0.5
    var localeOverride: Locale? = nil
    var progressiveDebounceInterval: TimeInterval = 0.4
}

struct ExtractionConfiguration: Sendable {
    var responseTimeout: TimeInterval = 10
    var requiredFieldIds: Set<String> = [
        "patientName", "patientAge", "patientPhone",
        "surgeonName", "surgeryDate", "surgeryTime",
        "procedureName", "procedureDuration"
    ]
    var sanitizeLogs: Bool = true
}
