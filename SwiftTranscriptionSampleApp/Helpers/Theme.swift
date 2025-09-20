import SwiftUI

// MARK: - Dark Medical Theme System

enum DarkTheme {
    // MARK: - Background Colors
    static let background = Color(red: 0.04, green: 0.08, blue: 0.12)
    static let secondaryBackground = Color(red: 0.06, green: 0.11, blue: 0.16)
    static let cardBackground = Color(red: 0.08, green: 0.14, blue: 0.20).opacity(0.8)
    static let glassBackground = Color(red: 0.12, green: 0.18, blue: 0.24).opacity(0.3)

    // MARK: - Accent Colors
    static let cyanAccent = Color(red: 0, green: 0.78, blue: 0.98)
    static let cyanLight = Color(red: 0.2, green: 0.85, blue: 1.0)
    static let blueAccent = Color(red: 0, green: 0.47, blue: 1.0)
    static let purpleAccent = Color(red: 0.58, green: 0.42, blue: 0.98)

    // MARK: - Text Colors
    static let textPrimary = Color.white.opacity(0.95)
    static let textSecondary = Color.white.opacity(0.6)
    static let textTertiary = Color.white.opacity(0.4)
    static let textPlaceholder = Color(red: 0.4, green: 0.5, blue: 0.6)

    // MARK: - Status Colors
    static let success = Color(red: 0.2, green: 0.78, blue: 0.4)
    static let warning = Color(red: 0.98, green: 0.65, blue: 0.2)
    static let error = Color(red: 0.98, green: 0.3, blue: 0.3)
    static let info = cyanAccent

    // MARK: - Gradients
    static let cyanGradient = LinearGradient(
        colors: [cyanAccent, cyanLight],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let microphoneGradient = LinearGradient(
        colors: [cyanAccent, blueAccent, purpleAccent],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let glowGradient = RadialGradient(
        colors: [cyanAccent.opacity(0.3), cyanAccent.opacity(0)],
        center: .center,
        startRadius: 0,
        endRadius: 100
    )

    // MARK: - Border Colors
    static let borderPrimary = cyanAccent.opacity(0.3)
    static let borderSecondary = Color.white.opacity(0.1)
    static let borderActive = cyanAccent
}

// MARK: - Glass Morphism View Modifier

struct GlassMorphism: ViewModifier {
    let cornerRadius: CGFloat
    let borderWidth: CGFloat

    init(cornerRadius: CGFloat = 20, borderWidth: CGFloat = 1) {
        self.cornerRadius = cornerRadius
        self.borderWidth = borderWidth
    }

    func body(content: Content) -> some View {
        content
            .background(
                ZStack {
                    // Blur background
                    DarkTheme.glassBackground
                        .blur(radius: 10)

                    // Glass effect
                    DarkTheme.cardBackground
                        .opacity(0.6)
                }
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(DarkTheme.borderPrimary, lineWidth: borderWidth)
            )
            .cornerRadius(cornerRadius)
            .shadow(color: DarkTheme.cyanAccent.opacity(0.1), radius: 10, x: 0, y: 4)
    }
}

extension View {
    func glassMorphism(cornerRadius: CGFloat = 20, borderWidth: CGFloat = 1) -> some View {
        modifier(GlassMorphism(cornerRadius: cornerRadius, borderWidth: borderWidth))
    }
}

// MARK: - Glow Effect Modifier

struct GlowEffect: ViewModifier {
    let color: Color
    let radius: CGFloat

    func body(content: Content) -> some View {
        content
            .shadow(color: color.opacity(0.6), radius: radius, x: 0, y: 0)
            .shadow(color: color.opacity(0.3), radius: radius * 2, x: 0, y: 0)
    }
}

extension View {
    func glow(color: Color = DarkTheme.cyanAccent, radius: CGFloat = 8) -> some View {
        modifier(GlowEffect(color: color, radius: radius))
    }
}

// MARK: - Animation Constants

enum ThemeAnimation {
    static let standard = Animation.easeInOut(duration: 0.3)
    static let spring = Animation.spring(response: 0.5, dampingFraction: 0.7, blendDuration: 0)
    static let pulse = Animation.easeInOut(duration: 1.5).repeatForever(autoreverses: true)
}

// MARK: - Layout Constants

enum ThemeLayout {
    static let cornerRadius: CGFloat = 20
    static let smallCornerRadius: CGFloat = 12
    static let padding: CGFloat = 20
    static let smallPadding: CGFloat = 12
    static let borderWidth: CGFloat = 1
}