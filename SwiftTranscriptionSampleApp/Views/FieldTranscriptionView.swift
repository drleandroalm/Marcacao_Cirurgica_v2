import SwiftUI
import Speech

@MainActor
struct FieldTranscriptionView: View {
    @Binding var field: TemplateField
    @State private var isRecording = false
    @State private var recorder: Recorder?
    @State private var speechTranscriber: SpokenWordTranscriber?
    @State private var tempValue: String = ""
    
    let form: SurgicalRequestForm
    let onComplete: () -> Void
    
    init(field: Binding<TemplateField>, form: SurgicalRequestForm, onComplete: @escaping () -> Void) {
        self._field = field
        self.form = form
        self.onComplete = onComplete
    }
    
    var body: some View {
        VStack(spacing: 24) {
            VStack(alignment: .leading, spacing: 16) {
                Label(field.label, systemImage: iconForFieldType(field.fieldType))
                    .font(.title2)
                    .fontWeight(.semibold)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Instrução:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(instructionForField(field))
                        .font(.body)
                        .foregroundColor(.primary)
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Valor Transcrito:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text(tempValue.isEmpty ? field.placeholder : tempValue)
                        .font(.title3)
                        .foregroundColor(tempValue.isEmpty ? .gray : .primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    
                    if isRecording, let transcriber = speechTranscriber {
                        HStack {
                            Image(systemName: "mic.fill")
                                .foregroundColor(.red)
                                .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: isRecording)
                            
                            Text(transcriber.volatileTranscript)
                                .foregroundColor(.purple.opacity(0.4))
                                .font(.body)
                        }
                    }
                }
                
                if !tempValue.isEmpty && !validateField() {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text(validationMessage())
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(8)
                }
            }
            
            Spacer()
            
            VStack(spacing: 16) {
                Button(action: toggleRecording) {
                    HStack {
                        Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                            .font(.title2)
                        Text(isRecording ? "Parar Gravação" : "Iniciar Gravação")
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isRecording ? Color.red : Color.blue)
                    .cornerRadius(12)
                }
                
                HStack(spacing: 16) {
                    Button("Limpar") {
                        tempValue = ""
                    }
                    .buttonStyle(.bordered)
                    .disabled(tempValue.isEmpty || isRecording)
                    
                    Button("Confirmar") {
                        field.value = tempValue
                        onComplete()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(tempValue.isEmpty || !validateField() || isRecording)
                }
            }
        }
        .padding()
        .onAppear {
            tempValue = field.value
            setupRecorder()
        }
    }
    
    private func setupRecorder() {
        speechTranscriber = SpokenWordTranscriber(
            form: form,
            onFieldComplete: { transcribedText in
                if !transcribedText.isEmpty {
                    tempValue = transcribedText
                }
            },
            configuration: TranscriptionConfiguration()
        )
        
        if let transcriber = speechTranscriber {
            recorder = Recorder(transcriber: transcriber)
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
        
        tempValue = ""
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
            } catch {
                print("Stop recording error: \(error)")
                isRecording = false
            }
        }
    }
    
    private func iconForFieldType(_ type: FieldType) -> String {
        switch type {
        case .text:
            return "text.alignleft"
        case .age:
            return "number"
        case .number:
            return "number"
        case .date:
            return "calendar"
        case .time:
            return "clock"
        case .duration:
            return "timer"
        case .phone:
            return "phone"
        }
    }
    
    private func instructionForField(_ field: TemplateField) -> String {
        switch field.id {
        case "patientName":
            return "Fale o nome completo do paciente"
        case "patientAge":
            return "Fale a idade do paciente em anos"
        case "patientPhone":
            return "Fale o número de telefone com DDD"
        case "surgeonName":
            return "Fale o nome do cirurgião responsável"
        case "surgeryDate":
            return "Fale a data da cirurgia (dia, mês e ano)"
        case "surgeryTime":
            return "Fale o horário preferencial da cirurgia"
        case "procedureName":
            return "Fale o nome do procedimento cirúrgico"
        case "procedureDuration":
            return "Fale o tempo estimado do procedimento"
        default:
            return "Fale a informação solicitada"
        }
    }
    
    private func validateField() -> Bool {
        var tempField = field
        tempField.value = tempValue
        return tempField.validate()
    }
    
    private func validationMessage() -> String {
        switch field.fieldType {
        case .number:
            return "Digite apenas números (exemplo: 45)"
        case .date:
            return "Use o formato dia/mês/ano (exemplo: 25/12/2024)"
        case .time:
            return "Use o formato hora:minutos (exemplo: 14:30)"
        case .phone:
            return "Digite um telefone válido com DDD (exemplo: 11 98765-4321)"
        default:
            return "Campo inválido"
        }
    }
}
