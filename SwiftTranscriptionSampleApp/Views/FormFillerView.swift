import SwiftUI
import Speech

/// SwiftUI entry point that coordinates recording, transcription, extraction, and review of the surgical request form.
struct FormFillerView: View {
    @EnvironmentObject private var sessionStore: SessionStore
    @State private var form = SurgicalRequestForm()
    @State private var isRecording = false
    @State private var showingExportOptions = false
    @State private var showCopiedAlert = false
    @State private var recorder: Recorder?
    @State private var speechTranscriber: SpokenWordTranscriber?
    @State private var isContinuousMode = false
    @State private var processAfterRecording = true  // New toggle for processing mode
    @State private var persistRecording = false  // Save audio (WAV) to disk
    @State private var extractionResult: ExtractionResult?
    @State private var showingPreview = false
    @State private var isReconfiguring = false
    
    // Popups after preview confirmation
    @State private var showingCTIPopup = false
    @State private var showingHemPopup = false
    @State private var hemPopupEnabled = true
    @State private var tempHemYes: Bool? = nil
    @State private var tempHemSpec: String = ""
    // Optional step: decisions sheet after popups
    @State private var showingDecisions = false
    
    init() {
        
    }
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                modeToggle
                hemPopupToggle
                progressBar
                
                ScrollView {
                    VStack(spacing: 20) {
                        if isContinuousMode {
                            continuousRecordingCard
                        } else {
                            currentFieldCard
                        }
                        fieldsOverview
                    }
                    .padding()
                }
                
                if form.isComplete {
                    completionToolbar
                }
                
                controlToolbar
            }
            .navigationTitle("Formulário Cirúrgico")
            .navigationBarTitleDisplayMode(.inline)
            .alert("Copiado!", isPresented: $showCopiedAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text("O formulário foi copiado para a área de transferência")
            }
            .onAppear {
                setupRecorder()
            }
            .sheet(isPresented: $showingPreview) {
                if let result = extractionResult {
                    FormPreviewView(
                        extractionResult: result,
                        form: form,
                        onConfirm: { confirmedForm in
                            form = confirmedForm
                            showingPreview = false
                            // Start post-confirmation decisions flow
                            showingCTIPopup = true
                        },
                        onRetry: {
                            showingPreview = false
                            extractionResult = nil
                            form.reset()
                            speechTranscriber?.resetContinuousMode()
                        }
                    )
                }
            }
            // CTI popup
            .sheet(isPresented: $showingCTIPopup) {
                DecisionPopupView(
                    title: "Necessidade de CTI?",
                    negativeTitle: "Não",
                    positiveTitle: "SIM",
                    negativeColor: .blue,
                    positiveColor: .red,
                    onNegative: {
                        form.needsCTI = false
                        showingCTIPopup = false
                        if hemPopupEnabled {
                            showingHemPopup = true
                        } else {
                            showingDecisions = true
                        }
                    },
                    onPositive: {
                        form.needsCTI = true
                        showingCTIPopup = false
                        if hemPopupEnabled {
                            showingHemPopup = true
                        } else {
                            showingDecisions = true
                        }
                    }
                )
                .presentationDetents([.fraction(0.35)])
            }
            // Hemocomponents popup (two-step when SIM)
            .sheet(isPresented: $showingHemPopup) {
                HemocomponentsPopupView(
                    initialYes: form.needsHemocomponents ?? false,
                    initialSpec: form.hemocomponentsSpecification,
                    onNo: {
                        form.needsHemocomponents = false
                        form.hemocomponentsSpecification = ""
                        showingHemPopup = false
                        extractionResult = nil
                        // Optional: open decisions sheet after popups
                        showingDecisions = true
                    },
                    onYesConfirm: { spec in
                        form.needsHemocomponents = true
                        form.hemocomponentsSpecification = spec
                        showingHemPopup = false
                        extractionResult = nil
                        // Optional: open decisions sheet after popups
                        showingDecisions = true
                    },
                    onCancel: {
                        showingHemPopup = false
                        // Optionally let user return to preview to edit again
                        showingPreview = true
                    }
                )
                .presentationDetents([.fraction(0.45)])
            }
            // Optional final decisions sheet
            .sheet(isPresented: $showingDecisions) {
                PostTranscriptionDecisionsView(
                    form: $form,
                    onComplete: {
                        // Archive session to history
                        let session = SurgerySession.from(form: form)
                        sessionStore.add(session)
                        showingDecisions = false
                    },
                    onCancel: {
                        // Return to preview so user can adjust
                        showingDecisions = false
                        showingPreview = true
                    }
                )
            }
            .overlay(alignment: .top) {
                if isReconfiguring {
                    Text("Reconfiguring…")
                        .font(.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                        .padding(.top, 8)
                        .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
        }
    }
    
    private var modeToggle: some View {
        HStack {
            Text("Modo:")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Picker("Modo de Gravação", selection: $isContinuousMode) {
                Text("Campo por Campo").tag(false)
                Text("Contínuo").tag(true)
            }
            .pickerStyle(.segmented)
            .disabled(isRecording)
            
            Spacer()
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(.systemGray6))
        .onChange(of: isContinuousMode) { _, _ in
            // Safely stop, reconfigure, and auto-resume when switching modes
            reconfigureRecorder(autoResume: true)
        }
        .onChange(of: persistRecording) { _, _ in
            // Safely stop, reconfigure, and auto-resume if we were recording
            reconfigureRecorder(autoResume: true)
        }
    }

    private var hemPopupToggle: some View {
        HStack(spacing: 12) {
            Image(systemName: "drop.fill").foregroundColor(.red)
            Toggle("Perguntar reserva de hemocomponentes", isOn: $hemPopupEnabled)
                .toggleStyle(SwitchToggleStyle(tint: .red))
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(.systemGray6))
    }
    
    private var continuousRecordingCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: isRecording ? "waveform" : "mic.slash")
                    .foregroundColor(isRecording ? .red : .gray)
                
                Text("Gravação Contínua")
                    .font(.headline)
                
                Spacer()
                
                // Clear session button
                if !isRecording && (!(speechTranscriber?.continuousTranscript.isEmpty ?? true) ||
                                   !(speechTranscriber?.finalizedTranscript.characters.isEmpty ?? true)) {
                    Button(action: clearSession) {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.red)
                    }
                    .buttonStyle(.plain)
                }
                
                if EntityExtractor.shared.isAvailable {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.blue)
                }
            }
            
            if isRecording {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Fale todas as informações do paciente...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    if let transcriber = speechTranscriber {
                        Text(transcriber.volatileTranscript)
                            .foregroundColor(.purple.opacity(0.4))
                            .font(.body)
                        
                        if !transcriber.continuousTranscript.isEmpty {
                            Text("Já Transcrito:")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.top, 8)
                            
                            Text(transcriber.continuousTranscript)
                                .font(.footnote)
                                .foregroundColor(.primary)
                                .padding()
                                .background(Color(.systemGray6))
                                .cornerRadius(8)
                        }
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    // Processing mode toggle
                    Toggle(isOn: $processAfterRecording) {
                        HStack {
                            Image(systemName: processAfterRecording ? "clock.badge.checkmark" : "bolt.circle")
                                .foregroundColor(processAfterRecording ? .blue : .orange)
                            Text(processAfterRecording ? "Processar após gravação completa" : "Processar durante gravação")
                                .font(.caption)
                        }
                    }
                    .toggleStyle(SwitchToggleStyle(tint: .blue))
                    .padding(.bottom, 8)
                    
                    // Save audio to disk toggle
                    Toggle(isOn: $persistRecording) {
                        HStack {
                            Image(systemName: persistRecording ? "waveform.circle.fill" : "waveform.circle")
                                .foregroundColor(persistRecording ? .green : .gray)
                            Text("Salvar áudio (WAV)")
                                .font(.caption)
                        }
                    }
                    .toggleStyle(SwitchToggleStyle(tint: .green))
                    .padding(.bottom, 8)
                    
                    Text("Toque em 'Gravar' e diga todas as informações de uma só vez:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Text("Exemplo: 'Paciente João Silva, 45 anos, telefone 11987654321, cirurgia amanhã às 14 horas, procedimento apendicectomia, doutor Pedro Santos.'")
                        .font(.footnote)
                        .foregroundColor(.secondary)
                        .italic()
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var progressBar: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Campo \(form.currentFieldIndex + 1) de \(form.fields.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Text("\(Int(form.progressPercentage * 100))% Completo")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            ProgressView(value: form.progressPercentage)
                .tint(.blue)
        }
        .padding()
        .background(Color(.systemGray6))
    }
    
    private var currentFieldCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: form.currentField?.isComplete == true ? "checkmark.circle.fill" : "circle")
                    .foregroundColor(form.currentField?.isComplete == true ? .green : .gray)
                
                Text(form.currentField?.label ?? "")
                    .font(.headline)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Valor Transcrito:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(form.currentField?.value.isEmpty == true ? form.currentField?.placeholder ?? "" : form.currentField?.value ?? "")
                    .font(.body)
                    .foregroundColor(form.currentField?.value.isEmpty == true ? .gray : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                
                if isRecording, let transcriber = speechTranscriber {
                    Text(transcriber.volatileTranscript)
                        .foregroundColor(.purple.opacity(0.4))
                        .font(.body)
                }
            }
            
            if let field = form.currentField, !field.value.isEmpty && !field.validate() {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text(validationMessage(for: field))
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var fieldsOverview: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Todos os Campos")
                .font(.headline)
                .padding(.bottom, 8)
            
            ForEach(Array(form.fields.enumerated()), id: \.element.id) { index, field in
                HStack {
                    Image(systemName: field.isComplete ? "checkmark.circle.fill" : "circle")
                        .foregroundColor(field.isComplete ? .green : .gray)
                        .frame(width: 20)
                    
                    VStack(alignment: .leading) {
                        Text(field.label)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text(field.value.isEmpty ? field.placeholder : field.value)
                            .font(.footnote)
                            .foregroundColor(field.value.isEmpty ? .gray : .primary)
                            .lineLimit(1)
                    }
                    
                    Spacer()
                    
                    if index == form.currentFieldIndex {
                        Image(systemName: "arrowtriangle.right.fill")
                            .foregroundColor(.blue)
                            .font(.caption)
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(index == form.currentFieldIndex ? Color.blue.opacity(0.1) : Color.clear)
                .cornerRadius(8)
                .onTapGesture {
                    if !isRecording {
                        form.moveToField(at: index)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
    
    private var completionToolbar: some View {
        HStack(spacing: 16) {
            Button(action: copyToClipboard) {
                Label("Copiar", systemImage: "doc.on.doc")
                    .font(.footnote)
            }
            .buttonStyle(.borderedProminent)
            
            Button(action: shareForm) {
                Label("Compartilhar", systemImage: "square.and.arrow.up")
                    .font(.footnote)
            }
            .buttonStyle(.bordered)
            
            Button(action: { form.reset() }) {
                Label("Limpar", systemImage: "trash")
                    .font(.footnote)
            }
            .buttonStyle(.bordered)
            .foregroundColor(.red)
        }
        .padding()
        .background(Color(.systemGray6))
    }
    
    private var controlToolbar: some View {
        HStack {
            Button(action: moveToPreviousField) {
                Image(systemName: "chevron.left")
                    .frame(width: 44, height: 44)
            }
            .disabled(form.currentFieldIndex == 0 || isRecording || isContinuousMode)
            
            Spacer()
            
            Button(action: toggleRecording) {
                VStack(spacing: 4) {
                    Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                        .font(.title2)
                        .foregroundColor(isRecording ? .red : .white)
                    Text(isRecording ? "Parar" : "Gravar")
                        .font(.caption)
                        .foregroundColor(isRecording ? .red : .white)
                }
                .frame(width: 80, height: 80)
                .background(isRecording ? Color.red.opacity(0.2) : Color.red)
                .clipShape(Circle())
            }
            
            Spacer()
            
            Button(action: moveToNextField) {
                Image(systemName: "chevron.right")
                    .frame(width: 44, height: 44)
            }
            .disabled(form.currentFieldIndex == form.fields.count - 1 || isRecording || isContinuousMode)
            
            // Play button (when audio was persisted)
            if !isRecording && persistRecording {
                Button(action: { recorder?.playRecording() }) {
                    Image(systemName: "play.circle")
                        .frame(width: 44, height: 44)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
    }
    
    private func setupRecorder() {
        // Clean up previous recorder and transcriber
        Task {
            if isRecording {
                // stopRecording() already manages its own Task; no await needed
                stopRecording()
            }
            
            // Deallocate previous transcriber resources
            if let previousTranscriber = speechTranscriber {
                await previousTranscriber.deallocate()
            }
            
            // Clear previous recorder
            recorder = nil
            speechTranscriber = nil
            
            // Setup new recorder with appropriate mode
            let onFieldComplete: ((String) -> Void)? = isContinuousMode ? nil : { transcribedText in
                DispatchQueue.main.async {
                    if !transcribedText.isEmpty {
                        form.updateCurrentFieldValue(transcribedText)
                    }
                }
            }
            
            let onContinuousComplete: ((ExtractionResult) -> Void)? = isContinuousMode ? { result in
                DispatchQueue.main.async {
                    extractionResult = result
                    showingPreview = true
                }
            } : nil
            
            let transcriptionConfig = TranscriptionConfiguration(
                isContinuousMode: isContinuousMode,
                processAfterRecording: processAfterRecording
            )
            speechTranscriber = SpokenWordTranscriber(
                form: form,
                onFieldComplete: onFieldComplete,
                onContinuousComplete: onContinuousComplete,
                configuration: transcriptionConfig
            )
            
            if let transcriber = speechTranscriber {
                let recordingConfig = RecordingConfiguration(shouldWriteToDisk: persistRecording)
                recorder = Recorder(transcriber: transcriber, configuration: recordingConfig)
            }
        }
    }
    
    private func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }
    
    private func startRecording() {
        guard let recorder = recorder else { return }
        
        isRecording = true
        Task {
            do {
                try await recorder.record()
            } catch {
                print("Recording error: \(error)")
                isRecording = false
            }
        }
    }
    
    private func stopRecording() {
        guard let recorder = recorder else { return }
        
        Task {
            do {
                try await recorder.stopRecording()
                
                isRecording = false
                
                if !isContinuousMode && form.currentField?.isComplete == true && form.currentFieldIndex < form.fields.count - 1 {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        _ = form.moveToNextField()
                    }
                }
            } catch {
                print("Stop recording error: \(error)")
                isRecording = false
            }
        }
    }
    
    // Safely reconfigure the recorder/transcriber pipeline. If currently recording,
    // stop first, re-setup, and optionally auto-resume.
    private func reconfigureRecorder(autoResume: Bool) {
        Task {
            await MainActor.run {
                withAnimation { isReconfiguring = true }
            }
            let wasRecording = isRecording
            if wasRecording {
                await stopRecordingAndWait()
            }
            // Recreate recorder/transcriber with current toggles
            await MainActor.run {
                setupRecorder()
            }
            if autoResume && wasRecording {
                startRecording()
            }
            await MainActor.run {
                withAnimation { isReconfiguring = false }
            }
        }
    }
    
    // Awaitable stop that does not spawn its own Task, so we can sequence operations.
    private func stopRecordingAndWait() async {
        guard let recorder = recorder else { return }
        do {
            try await recorder.stopRecording()
        } catch {
            print("Stop recording (awaitable) error: \(error)")
        }
        isRecording = false
    }
    
    private func moveToNextField() {
        _ = form.moveToNextField()
    }
    
    private func moveToPreviousField() {
        _ = form.moveToPreviousField()
    }
    
    private func copyToClipboard() {
        UIPasteboard.general.string = form.generateFilledTemplate()
        showCopiedAlert = true
    }
    
    private func shareForm() {
        let text = form.generateFilledTemplate()
        let activityVC = UIActivityViewController(activityItems: [text], applicationActivities: nil)
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootVC = window.rootViewController {
            rootVC.present(activityVC, animated: true)
        }
    }
    
    private func validationMessage(for field: TemplateField) -> String {
        switch field.fieldType {
        case .number:
            return "Digite apenas números"
        case .date:
            return "Use o formato DD/MM/AAAA"
        case .time:
            return "Use o formato HH:MM"
        case .phone:
            return "Digite um telefone válido (10 ou 11 dígitos)"
        default:
            return "Campo inválido"
        }
    }
    
    private func clearSession() {
        print("🧹 FormFillerView: Clearing session...")
        
        // Reset transcriber session
        speechTranscriber?.resetSession()
        
        // Reset form
        print("  - Resetting form")
        form.reset()
        
        // Clear extraction result
        print("  - Clearing extraction result")
        extractionResult = nil
        
        // User feedback
        print("✅ Session cleared successfully")
    }
}
