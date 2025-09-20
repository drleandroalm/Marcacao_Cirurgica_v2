import SwiftUI

struct DecisionPopupView: View {
    let title: String
    let negativeTitle: String
    let positiveTitle: String
    let negativeColor: Color
    let positiveColor: Color
    let onNegative: () -> Void
    let onPositive: () -> Void

    @State private var appear = false

    var body: some View {
        ZStack {
            // Dark backdrop with blur
            DarkTheme.background
                .opacity(0.95)
                .ignoresSafeArea()
                .onTapGesture { } // Prevent dismissal

            VStack(spacing: 24) {
                // Title with icon
                VStack(spacing: 12) {
                    Image(systemName: "questionmark.circle.fill")
                        .font(.system(size: 48))
                        .foregroundColor(DarkTheme.cyanAccent)
                        .glow(color: DarkTheme.cyanAccent, radius: 12)

                    Text(title)
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(DarkTheme.textPrimary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 8)

                // Buttons
                HStack(spacing: 16) {
                    // Negative button
                    Button(action: {
                        withAnimation(.spring()) {
                            onNegative()
                        }
                    }) {
                        Text(negativeTitle)
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundColor(DarkTheme.textPrimary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                RoundedRectangle(cornerRadius: 16)
                                    .fill(DarkTheme.secondaryBackground)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 16)
                                            .stroke(DarkTheme.borderSecondary, lineWidth: 1)
                                    )
                            )
                    }
                    .buttonStyle(PlainButtonStyle())

                    // Positive button with gradient
                    Button(action: {
                        withAnimation(.spring()) {
                            onPositive()
                        }
                    }) {
                        Text(positiveTitle)
                            .font(.system(size: 16, weight: .bold))
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                DarkTheme.microphoneGradient
                                    .cornerRadius(16)
                                    .glow(color: positiveColor, radius: 8)
                            )
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.horizontal, 8)
            }
            .padding(24)
            .glassMorphism(cornerRadius: 24)
            .padding(.horizontal, 40)
            .scaleEffect(appear ? 1 : 0.8)
            .opacity(appear ? 1 : 0)
            .animation(.spring(response: 0.5, dampingFraction: 0.7), value: appear)
        }
        .onAppear {
            appear = true
        }
    }
}

struct HemocomponentsPopupView: View {
    @State private var wantsYes: Bool = false
    @State private var spec: String = ""
    @State private var selectedCH: String? = nil
    @State private var showPlateletsStep: Bool = false
    @State private var plateletsAdded: Bool = false
    @State private var showPlasmaStep: Bool = false
    @State private var appear = false
    @State private var currentStep = 0

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
        ZStack {
            // Dark backdrop
            DarkTheme.background
                .opacity(0.95)
                .ignoresSafeArea()
                .onTapGesture { }

            VStack(spacing: 20) {
                // Header with icon
                VStack(spacing: 12) {
                    Image(systemName: "drop.fill")
                        .font(.system(size: 48))
                        .foregroundColor(DarkTheme.error)
                        .glow(color: DarkTheme.error, radius: 12)

                    Text("Reserva de hemocomponentes?")
                        .font(.system(size: 20, weight: .semibold))
                        .foregroundColor(DarkTheme.textPrimary)
                }
                .padding(.top)

                if !wantsYes {
                    // Initial question
                    HStack(spacing: 16) {
                        Button(action: {
                            withAnimation(.spring()) { onNo() }
                        }) {
                            Text("Não")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(DarkTheme.textPrimary)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(
                                    RoundedRectangle(cornerRadius: 16)
                                        .fill(DarkTheme.secondaryBackground)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 16)
                                                .stroke(DarkTheme.borderSecondary, lineWidth: 1)
                                        )
                                )
                        }

                        Button(action: {
                            withAnimation(.spring()) {
                                wantsYes = true
                                currentStep = 1
                            }
                        }) {
                            Text("SIM")
                                .font(.system(size: 16, weight: .bold))
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(
                                    LinearGradient(
                                        colors: [DarkTheme.error, DarkTheme.error.opacity(0.8)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                    .cornerRadius(16)
                                    .glow(color: DarkTheme.error, radius: 8)
                                )
                        }
                    }
                    .padding(.horizontal)
                } else {
                    // Multi-step form with animation
                    ScrollView {
                        VStack(spacing: 16) {
                            // Step indicators
                            HStack(spacing: 8) {
                                ForEach(1...3, id: \.self) { step in
                                    Circle()
                                        .fill(currentStep >= step ? DarkTheme.cyanAccent : DarkTheme.borderSecondary)
                                        .frame(width: 8, height: 8)
                                        .animation(.spring(), value: currentStep)
                                }
                            }
                            .padding(.bottom, 8)

                            // Step 1: Choose CH
                            if currentStep >= 1 {
                                VStack(alignment: .leading, spacing: 12) {
                                    Text("Concentrado de Hemácias (CH)")
                                        .font(.system(size: 14, weight: .medium))
                                        .foregroundColor(DarkTheme.textSecondary)

                                    HStack(spacing: 12) {
                                        DarkQuickChoiceButton(
                                            title: "600mL de CH",
                                            selected: selectedCH == "600mL de CH"
                                        ) {
                                            withAnimation(.spring()) {
                                                selectedCH = "600mL de CH"
                                                updateSpecBase()
                                                currentStep = 2
                                            }
                                        }

                                        DarkQuickChoiceButton(
                                            title: "900mL de CH",
                                            selected: selectedCH == "900mL de CH"
                                        ) {
                                            withAnimation(.spring()) {
                                                selectedCH = "900mL de CH"
                                                updateSpecBase()
                                                currentStep = 2
                                            }
                                        }
                                    }
                                }
                                .transition(.asymmetric(
                                    insertion: .scale.combined(with: .opacity),
                                    removal: .opacity
                                ))
                            }

                            // Step 2: Platelets
                            if currentStep >= 2 {
                                VStack(alignment: .leading, spacing: 12) {
                                    Text("Necessidade de Plaquetas?")
                                        .font(.system(size: 14, weight: .medium))
                                        .foregroundColor(DarkTheme.textSecondary)

                                    HStack(spacing: 12) {
                                        DarkQuickChoiceButton(title: "NÃO", selected: !plateletsAdded) {
                                            withAnimation(.spring()) {
                                                plateletsAdded = false
                                                currentStep = 3
                                            }
                                        }

                                        DarkQuickChoiceButton(title: "7 UN", selected: plateletsAdded) {
                                            withAnimation(.spring()) {
                                                plateletsAdded = true
                                                addPlatelets()
                                                currentStep = 3
                                            }
                                        }
                                    }
                                }
                                .transition(.asymmetric(
                                    insertion: .scale.combined(with: .opacity),
                                    removal: .opacity
                                ))
                            }

                            // Step 3: Plasma
                            if currentStep >= 3 {
                                VStack(alignment: .leading, spacing: 12) {
                                    Text("Reserva de Plasma?")
                                        .font(.system(size: 14, weight: .medium))
                                        .foregroundColor(DarkTheme.textSecondary)

                                    HStack(spacing: 12) {
                                        DarkQuickChoiceButton(title: "NÃO", selected: false) {
                                            // Don't add plasma, proceed to confirm
                                        }

                                        DarkQuickChoiceButton(title: "600mL", selected: false) {
                                            addPlasma()
                                        }
                                    }
                                }
                                .transition(.asymmetric(
                                    insertion: .scale.combined(with: .opacity),
                                    removal: .opacity
                                ))
                            }

                            // Specification field
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Especificar")
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundColor(DarkTheme.textSecondary)

                                TextField("Descreva a reserva", text: $spec)
                                    .font(.system(size: 14))
                                    .foregroundColor(DarkTheme.textPrimary)
                                    .padding()
                                    .background(DarkTheme.secondaryBackground)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12)
                                            .stroke(DarkTheme.cyanAccent.opacity(0.3), lineWidth: 1)
                                    )
                                    .cornerRadius(12)
                            }
                            .padding(.top, 8)

                            // Action buttons
                            HStack(spacing: 16) {
                                Button(action: {
                                    withAnimation(.spring()) { onCancel() }
                                }) {
                                    Text("Cancelar")
                                        .font(.system(size: 16, weight: .semibold))
                                        .foregroundColor(DarkTheme.textSecondary)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(
                                            RoundedRectangle(cornerRadius: 16)
                                                .fill(DarkTheme.secondaryBackground)
                                                .overlay(
                                                    RoundedRectangle(cornerRadius: 16)
                                                        .stroke(DarkTheme.borderSecondary, lineWidth: 1)
                                                )
                                        )
                                }

                                Button(action: {
                                    withAnimation(.spring()) { finalize(spec) }
                                }) {
                                    Text("Confirmar")
                                        .font(.system(size: 16, weight: .bold))
                                        .foregroundColor(.white)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 16)
                                        .background(
                                            DarkTheme.cyanGradient
                                                .cornerRadius(16)
                                                .glow(color: DarkTheme.success, radius: 8)
                                        )
                                }
                                .disabled(spec.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                                .opacity(spec.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.5 : 1)
                            }
                            .padding(.top, 16)
                        }
                    }
                    .frame(maxHeight: 400)
                }
            }
            .padding(24)
            .glassMorphism(cornerRadius: 24)
            .padding(.horizontal, 20)
            .scaleEffect(appear ? 1 : 0.8)
            .opacity(appear ? 1 : 0)
            .animation(.spring(response: 0.5, dampingFraction: 0.7), value: appear)
        }
        .onAppear {
            appear = true
        }
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

// MARK: - Dark Quick Choice Button

private struct DarkQuickChoiceButton: View {
    let title: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(selected ? .white : DarkTheme.textPrimary)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(
                    Group {
                        if selected {
                            DarkTheme.cyanGradient
                                .cornerRadius(12)
                                .glow(color: DarkTheme.cyanAccent, radius: 6)
                        } else {
                            RoundedRectangle(cornerRadius: 12)
                                .fill(DarkTheme.secondaryBackground)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(DarkTheme.borderSecondary, lineWidth: 1)
                                )
                        }
                    }
                )
        }
        .buttonStyle(PlainButtonStyle())
        .scaleEffect(selected ? 1.05 : 1.0)
        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: selected)
    }
}

// Original QuickChoiceButton for compatibility
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
