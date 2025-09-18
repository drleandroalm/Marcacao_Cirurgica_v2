import SwiftUI

struct DecisionPopupView: View {
    let title: String
    let negativeTitle: String
    let positiveTitle: String
    let negativeColor: Color
    let positiveColor: Color
    let onNegative: () -> Void
    let onPositive: () -> Void
    
    var body: some View {
        VStack(spacing: 16) {
            Text(title)
                .font(.title3)
                .fontWeight(.semibold)
                .padding(.top)
            
            HStack(spacing: 16) {
                Button(negativeTitle, action: onNegative)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(negativeColor.opacity(0.15))
                    .foregroundColor(negativeColor)
                    .clipShape(Capsule())
                
                Button(positiveTitle, action: onPositive)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(positiveColor)
                    .foregroundColor(.white)
                    .clipShape(Capsule())
            }
            .padding(.horizontal)
            
            Spacer(minLength: 8)
        }
        .padding()
    }
}

struct HemocomponentsPopupView: View {
    @State private var wantsYes: Bool = false
    @State private var spec: String = ""
    @State private var selectedCH: String? = nil
    @State private var showPlateletsStep: Bool = false
    @State private var plateletsAdded: Bool = false
    @State private var showPlasmaStep: Bool = false
    
    let initialYes: Bool
    let initialSpec: String
    let onNo: () -> Void
    let onYesConfirm: (String) -> Void
    let onCancel: () -> Void
    
    init(initialYes: Bool, initialSpec: String, onNo: @escaping () -> Void, onYesConfirm: @escaping (String) -> Void, onCancel: @escaping () -> Void) {
        self.initialYes = initialYes
        self.initialSpec = initialSpec
        self.onNo = onNo
        self.onYesConfirm = onYesConfirm
        self.onCancel = onCancel
        _wantsYes = State(initialValue: initialYes)
        _spec = State(initialValue: initialSpec)
    }
    
    var body: some View {
        VStack(spacing: 16) {
            Text("Reserva de hemocomponentes?")
                .font(.title3)
                .fontWeight(.semibold)
                .padding(.top)
            
            HStack(spacing: 16) {
                Button("Não") { onNo() }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue.opacity(0.15))
                    .foregroundColor(.blue)
                    .clipShape(Capsule())
                
                Button("SIM") { wantsYes = true }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .clipShape(Capsule())
            }
            .padding(.horizontal)
            
            if wantsYes {
                // Step 1: Choose CH quick options
                VStack(alignment: .leading, spacing: 8) {
                    Text("Concentrado de Hemácias (CH)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    HStack(spacing: 12) {
                        QuickChoiceButton(title: "600mL de CH", selected: selectedCH == "600mL de CH") {
                            withAnimation { selectedCH = "600mL de CH"; updateSpecBase() ; showPlateletsStep = true }
                        }
                        QuickChoiceButton(title: "900mL de CH", selected: selectedCH == "900mL de CH") {
                            withAnimation { selectedCH = "900mL de CH"; updateSpecBase() ; showPlateletsStep = true }
                        }
                    }
                }
                .padding(.horizontal)

                // Step 2: Platelets
                if showPlateletsStep {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Necessidade de Plaquetas?")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        HStack(spacing: 12) {
                            QuickChoiceButton(title: "NÃO", selected: !plateletsAdded) {
                                // finalize immediately with CH only
                                finalize(spec)
                            }
                            QuickChoiceButton(title: "7 UN", selected: plateletsAdded) {
                                withAnimation {
                                    plateletsAdded = true
                                    addPlatelets()
                                    showPlasmaStep = true
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                // Step 3: Plasma
                if showPlasmaStep {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Reserva de Plasma?")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        HStack(spacing: 12) {
                            QuickChoiceButton(title: "NÃO", selected: false) {
                                finalize(spec)
                            }
                            QuickChoiceButton(title: "600mL", selected: false) {
                                addPlasma()
                                finalize(spec)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Especificar")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("Descreva a reserva (ex.: 2 concentrados de hemácias)", text: $spec)
                        .textInputAutocapitalization(.sentences)
                        .autocorrectionDisabled(false)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                }
                .padding(.horizontal)
                
                HStack(spacing: 16) {
                    Button("Cancelar") { onCancel() }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.gray.opacity(0.15))
                        .foregroundColor(.gray)
                        .clipShape(Capsule())
                    
                    Button("Confirmar") { finalize(spec) }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                        .disabled(spec.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .padding(.horizontal)
            }
            
            Spacer(minLength: 8)
        }
        .padding()
    }
    
    private func updateSpecBase() {
        if let base = selectedCH {
            spec = base
        }
    }
    
    private func addPlatelets() {
        if !spec.contains("Plaquetas") {
            if !spec.isEmpty { spec += " + " }
            spec += "7 Unidades de Plaquetas"
        }
    }
    
    private func addPlasma() {
        if !spec.contains("Plasma Fresco Congelado") {
            if !spec.isEmpty { spec += " + " }
            spec += "600mL de Plasma Fresco Congelado"
        }
    }
    
    private func finalize(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        onYesConfirm(trimmed)
    }
}

private struct QuickChoiceButton: View {
    let title: String
    let selected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.footnote)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(selected ? Color.accentColor : Color(.systemGray6))
                .foregroundColor(selected ? .white : .primary)
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}
