import Foundation
import SwiftUI
import AVFoundation

#if false
// DEPRECATED: Consolidated lifecycle to FormFillerView. Prefer using the view-driven wiring and avoid instantiating this coordinator in new code.

@MainActor
@Observable
final class FormBuildCoordinator {
    // Public, bindable state
    var form: SurgicalRequestForm
    var extractionResult: ExtractionResult?
    var validationResult: ValidationResult?
    var isRecording: Bool = false
    var isPaused: Bool = false
    var isProcessing: Bool = false
    
    // Optional callbacks to notify hosting UI
    var onUpdate: ((SurgicalRequestForm, ExtractionResult?, ValidationResult?) -> Void)?
    var onError: ((Error) -> Void)?
    
    // Internals
    private let transcriber: SpokenWordTranscriber
    private let recorder: Recorder
    
    init(isContinuousMode: Bool = true, processAfterRecording: Bool = true) {
        // Fresh form for each coordinator instance
        let form = SurgicalRequestForm()
        self.form = form
        
        // Create transcriber without the problematic closure first
        let transcriptionConfig = TranscriptionConfiguration(
            isContinuousMode: isContinuousMode,
            processAfterRecording: processAfterRecording
        )
        let newTranscriber = SpokenWordTranscriber(
            form: form,
            onFieldComplete: { _ in
                // Field-by-field mode feedback if needed
            },
            onContinuousComplete: nil,  // Set to nil initially
            configuration: transcriptionConfig
        )
        
        self.transcriber = newTranscriber
        self.recorder = Recorder(transcriber: newTranscriber)
        
        // Now set the closure after initialization is complete
        newTranscriber.onContinuousComplete = { [weak self] result in
            guard let self else { return }
            // The transcriber already merged entities into form on MainActor.
            self.extractionResult = result
            self.validate()
            self.onUpdate?(self.form, self.extractionResult, self.validationResult)
        }
    }
    
    // MARK: - Recording lifecycle
    func start() async {
        guard !isRecording else { return }
        do {
            isRecording = true
            isPaused = false
            try await recorder.record()
        } catch {
            isRecording = false
            onError?(error)
        }
    }
    
    func stop() async {
        guard isRecording else { return }
        do {
            isProcessing = true
            try await recorder.stopRecording()
            isRecording = false
            isPaused = false
            // By this point, SpokenWordTranscriber.finishTranscribing() will have
            // auto-processed the accumulated transcript (per your latest changes),
            // which triggers onContinuousComplete and fills the form.
            // Perform a final validation pass just in case.
            validate()
            onUpdate?(form, extractionResult, validationResult)
        } catch {
            onError?(error)
        }
        isProcessing = false
    }
    
    func pause() {
        guard isRecording, !isPaused else { return }
        recorder.pauseRecording()
        isPaused = true
    }
    
    func resume() {
        guard isRecording, isPaused else { return }
        do {
            try recorder.resumeRecording()
            isPaused = false
        } catch {
            onError?(error)
        }
    }
    
    func reset() {
        transcriber.resetSession()
        transcriber.resetContinuousMode()
        form.reset()
        extractionResult = nil
        validationResult = nil
        isRecording = false
        isPaused = false
        onUpdate?(form, extractionResult, validationResult)
    }
    
    // MARK: - Validation
    func validate() {
        validationResult = ComplianceValidator.validate(form: form)
    }
    
    // MARK: - Preview wiring
    // Use this in SwiftUI to get a ready-made preview with the latest state.
    func makePreviewView(onConfirm: @escaping (SurgicalRequestForm) -> Void,
                         onRetry: @escaping () -> Void) -> some View {
        let result = extractionResult ?? ExtractionResult(entities: [], unprocessedText: "", confidence: 0)
        return FormPreviewView(
            extractionResult: result,
            form: form,
            onConfirm: onConfirm,
            onRetry: onRetry
        )
    }
}
#endif
