import SwiftUI

// MARK: - Dark Segmented Control

struct DarkSegmentedControl: View {
    @Binding var selection: Bool
    let leftTitle: String
    let rightTitle: String

    var body: some View {
        HStack(spacing: 0) {
            // Left Option
            Button(action: { withAnimation(.spring()) { selection = false } }) {
                Text(leftTitle)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(selection ? DarkTheme.textSecondary : .white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        Group {
                            if !selection {
                                DarkTheme.cyanGradient
                                    .cornerRadius(16)
                                    .glow(radius: 6)
                            }
                        }
                    )
            }
            .buttonStyle(PlainButtonStyle())

            // Right Option
            Button(action: { withAnimation(.spring()) { selection = true } }) {
                Text(rightTitle)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundColor(selection ? .white : DarkTheme.textSecondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        Group {
                            if selection {
                                DarkTheme.cyanGradient
                                    .cornerRadius(16)
                                    .glow(radius: 6)
                            }
                        }
                    )
            }
            .buttonStyle(PlainButtonStyle())
        }
        .background(DarkTheme.secondaryBackground)
        .cornerRadius(18)
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(DarkTheme.borderSecondary, lineWidth: 1)
        )
        .padding(.horizontal, 4)
    }
}

// MARK: - Circular Progress Indicator

struct CircularProgressIndicator: View {
    let current: Int
    let total: Int
    let progress: Double

    var body: some View {
        ZStack {
            // Background circle
            Circle()
                .stroke(DarkTheme.borderSecondary, lineWidth: 3)
                .frame(width: 100, height: 100)

            // Progress arc
            Circle()
                .trim(from: 0, to: progress)
                .stroke(DarkTheme.cyanGradient, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .frame(width: 100, height: 100)
                .rotationEffect(.degrees(-90))
                .animation(.spring(), value: progress)
                .glow(color: DarkTheme.cyanAccent, radius: 4)

            // Text in center
            VStack(spacing: 2) {
                Text("\(current)")
                    .font(.system(size: 32, weight: .bold, design: .rounded))
                    .foregroundColor(DarkTheme.textPrimary)

                Text("de \(total)")
                    .font(.system(size: 14, weight: .medium))
                    .foregroundColor(DarkTheme.textSecondary)
            }
        }
    }
}

// MARK: - Cyan Toggle

struct CyanToggle: View {
    @Binding var isOn: Bool
    let label: String
    let icon: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(isOn ? DarkTheme.cyanAccent : DarkTheme.error)
                .font(.system(size: 18))

            Text(label)
                .font(.system(size: 16, weight: .medium))
                .foregroundColor(DarkTheme.textPrimary)

            Spacer()

            // Custom Toggle
            ZStack {
                Capsule()
                    .fill(isOn ? Color(DarkTheme.cyanAccent) : DarkTheme.secondaryBackground)
                    .frame(width: 50, height: 30)
                    .overlay(
                        Capsule()
                            .stroke(DarkTheme.borderSecondary, lineWidth: 1)
                    )

                Circle()
                    .fill(Color.white)
                    .frame(width: 26, height: 26)
                    .offset(x: isOn ? 10 : -10)
                    .shadow(color: Color.black.opacity(0.2), radius: 2, y: 1)
            }
            .onTapGesture {
                withAnimation(.spring()) {
                    isOn.toggle()
                }
            }
            .glow(color: isOn ? DarkTheme.cyanAccent : Color.clear, radius: 4)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(DarkTheme.cardBackground)
        .cornerRadius(16)
    }
}

// MARK: - Gradient Microphone Button

struct GradientMicrophoneButton: View {
    let isRecording: Bool
    let action: () -> Void

    @State private var isPulsing = false

    var body: some View {
        Button(action: action) {
            ZStack {
                // Outer glow ring when recording
                if isRecording {
                    Circle()
                        .fill(DarkTheme.cyanAccent.opacity(0.2))
                        .frame(width: 100, height: 100)
                        .scaleEffect(isPulsing ? 1.2 : 1.0)
                        .animation(ThemeAnimation.pulse, value: isPulsing)
                        .onAppear {
                            isPulsing = true
                        }
                }

                // Main button
                Circle()
                    .fill(DarkTheme.microphoneGradient)
                    .frame(width: 80, height: 80)
                    .overlay(
                        Circle()
                            .stroke(DarkTheme.cyanAccent.opacity(0.5), lineWidth: 2)
                    )
                    .glow(color: isRecording ? DarkTheme.cyanAccent : DarkTheme.blueAccent, radius: 8)

                // Icon
                Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                    .font(.system(size: 30, weight: .semibold))
                    .foregroundColor(.white)
                    .scaleEffect(isRecording ? 0.8 : 1.0)
            }
        }
        .buttonStyle(PlainButtonStyle())
        .scaleEffect(isRecording ? 1.05 : 1.0)
        .animation(.spring(), value: isRecording)
    }
}

// MARK: - Form Field Row

struct FormFieldRow: View {
    let field: TemplateField
    let isActive: Bool
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            HStack(spacing: 16) {
                // Circle indicator
                ZStack {
                    Circle()
                        .stroke(field.isComplete ? DarkTheme.cyanAccent : DarkTheme.borderSecondary, lineWidth: 2)
                        .frame(width: 24, height: 24)
                        .background(
                            Circle()
                                .fill(field.isComplete ? DarkTheme.cyanAccent : Color.clear)
                        )

                    if field.isComplete {
                        Image(systemName: "checkmark")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundColor(DarkTheme.background)
                    }
                }
                .glow(color: field.isComplete ? DarkTheme.cyanAccent : Color.clear, radius: 4)

                // Field content
                VStack(alignment: .leading, spacing: 4) {
                    Text(field.label)
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(DarkTheme.textPrimary)

                    Text(field.value.isEmpty ? field.placeholder : field.value)
                        .font(.system(size: 14))
                        .foregroundColor(field.value.isEmpty ? DarkTheme.textPlaceholder : DarkTheme.cyanLight)
                        .lineLimit(1)
                }

                Spacer()

                // Arrow for active field
                if isActive {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(DarkTheme.cyanAccent)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(isActive ? DarkTheme.cyanAccent.opacity(0.1) : DarkTheme.cardBackground)
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(isActive ? DarkTheme.cyanAccent : DarkTheme.borderSecondary, lineWidth: 1)
                    )
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

// MARK: - Glass Card

struct GlassCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(20)
            .background(
                ZStack {
                    DarkTheme.cardBackground
                        .blur(radius: 20)

                    DarkTheme.glassBackground
                }
            )
            .cornerRadius(20)
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(DarkTheme.borderPrimary, lineWidth: 1)
            )
            .shadow(color: DarkTheme.cyanAccent.opacity(0.05), radius: 10, y: 5)
    }
}

// MARK: - Example Card

struct ExampleCard: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 14, weight: .regular))
            .foregroundColor(DarkTheme.cyanLight)
            .italic()
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(DarkTheme.cyanAccent.opacity(0.1))
            .cornerRadius(16)
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(DarkTheme.cyanAccent.opacity(0.3), lineWidth: 1)
            )
    }
}

// MARK: - Navigation Button

struct NavigationArrowButton: View {
    let direction: Direction
    let isDisabled: Bool
    let action: () -> Void

    enum Direction {
        case left, right

        var icon: String {
            switch self {
            case .left: return "chevron.left"
            case .right: return "chevron.right"
            }
        }
    }

    var body: some View {
        Button(action: action) {
            Image(systemName: direction.icon)
                .font(.system(size: 20, weight: .semibold))
                .foregroundColor(isDisabled ? DarkTheme.textTertiary : DarkTheme.cyanAccent)
                .frame(width: 44, height: 44)
                .background(
                    Circle()
                        .fill(DarkTheme.cardBackground)
                        .overlay(
                            Circle()
                                .stroke(isDisabled ? DarkTheme.borderSecondary : DarkTheme.cyanAccent.opacity(0.5), lineWidth: 1)
                        )
                )
        }
        .disabled(isDisabled)
        .opacity(isDisabled ? 0.5 : 1.0)
    }
}

// MARK: - Dark Tab Bar

struct DarkTabBar: View {
    @Binding var selectedTab: Int
    let tabs: [(String, String)] // (title, icon)

    var body: some View {
        HStack(spacing: 0) {
            ForEach(0..<tabs.count, id: \.self) { index in
                Button(action: { selectedTab = index }) {
                    VStack(spacing: 4) {
                        Image(systemName: tabs[index].1)
                            .font(.system(size: 22))

                        Text(tabs[index].0)
                            .font(.system(size: 10))
                    }
                    .foregroundColor(selectedTab == index ? DarkTheme.cyanAccent : DarkTheme.textTertiary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
            }
        }
        .padding(.horizontal)
        .background(DarkTheme.secondaryBackground)
        .overlay(
            Rectangle()
                .fill(DarkTheme.borderSecondary)
                .frame(height: 1),
            alignment: .top
        )
    }
}