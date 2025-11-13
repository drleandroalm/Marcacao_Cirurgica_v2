import Foundation

/// Multi-factor confidence scoring system for entity extraction
struct EnhancedConfidence: Sendable {
    let overallScore: Double

    // Breakdown components
    let transcriptionQuality: Double  // ASR confidence (volatile vs finalized)
    let entityMatch: Double          // KB match score or pattern match quality
    let contextConsistency: Double   // Temporal/logical coherence
    let historicalAccuracy: Double   // Based on past corrections (future feature)

    init(
        transcriptionQuality: Double = 1.0,
        entityMatch: Double,
        contextConsistency: Double = 1.0,
        historicalAccuracy: Double = 1.0
    ) {
        self.transcriptionQuality = transcriptionQuality
        self.entityMatch = entityMatch
        self.contextConsistency = contextConsistency
        self.historicalAccuracy = historicalAccuracy

        // Weighted average: entityMatch is most important
        self.overallScore = (
            entityMatch * 0.50 +
            transcriptionQuality * 0.25 +
            contextConsistency * 0.15 +
            historicalAccuracy * 0.10
        )
    }

    /// Auto-accept threshold: very high confidence across all factors
    func shouldAutoAccept() -> Bool {
        return overallScore > 0.95 &&
               entityMatch > 0.90 &&
               contextConsistency > 0.85
    }

    /// Requires user confirmation: medium confidence
    func requiresConfirmation() -> Bool {
        return overallScore >= 0.70 && overallScore <= 0.95
    }

    /// Should reject or require manual entry: low confidence
    func shouldReject() -> Bool {
        return overallScore < 0.70
    }

    /// UI indicator color based on confidence level
    var uiColor: ConfidenceColor {
        if shouldAutoAccept() {
            return .green
        } else if requiresConfirmation() {
            return .orange
        } else {
            return .red
        }
    }

    enum ConfidenceColor: String, Sendable {
        case green = "High"    // Auto-accept
        case orange = "Medium" // Requires confirmation
        case red = "Low"       // Manual entry recommended
    }
}

/// Enhanced extracted entity with multi-factor confidence
struct EnhancedExtractedEntity: Sendable {
    let fieldId: String
    let value: String
    let confidence: EnhancedConfidence
    let alternatives: [String]
    let originalText: String
    let extractionMethod: ExtractionMethod

    enum ExtractionMethod: String, Sendable {
        case llm = "Foundation Models"
        case ruleBased = "Rule-Based Pattern"
        case knowledgeBase = "Knowledge Base Match"
        case hybrid = "Hybrid (Multiple Sources)"
    }
}

/// Confidence scorer for different entity types
@MainActor
final class ConfidenceScorer {

    /// Score a date entity based on format and context
    static func scoreDateEntity(
        extractedDate: String,
        isRelativeKeyword: Bool,
        hasExplicitContext: Bool
    ) -> EnhancedConfidence {
        let entityMatch: Double
        if isRelativeKeyword {
            entityMatch = 0.95  // "amanhã", "hoje" are very reliable
        } else if hasExplicitContext {
            entityMatch = 0.85  // "dia 15 de março"
        } else {
            entityMatch = 0.75  // Bare numeric date
        }

        let contextConsistency = validateDateConsistency(extractedDate)

        return EnhancedConfidence(
            entityMatch: entityMatch,
            contextConsistency: contextConsistency
        )
    }

    /// Score a time entity based on format
    static func scoreTimeEntity(
        extractedTime: String,
        hasExplicitContext: Bool
    ) -> EnhancedConfidence {
        let entityMatch: Double
        if extractedTime.contains(":") {
            entityMatch = 0.90  // "14:30" format
        } else if hasExplicitContext {
            entityMatch = 0.85  // "às 14 horas"
        } else {
            entityMatch = 0.75  // "14h" or spoken "quatorze horas"
        }

        let contextConsistency = validateTimeConsistency(extractedTime)

        return EnhancedConfidence(
            entityMatch: entityMatch,
            contextConsistency: contextConsistency
        )
    }

    /// Score a phone number entity
    static func scorePhoneEntity(
        extractedPhone: String,
        digitCount: Int,
        hasContext: Bool
    ) -> EnhancedConfidence {
        let entityMatch: Double
        if digitCount == 11 && hasContext {
            entityMatch = 0.85  // "telefone é (11) 98765-4321"
        } else if digitCount == 11 {
            entityMatch = 0.75  // Bare 11 digits
        } else if digitCount == 10 && hasContext {
            entityMatch = 0.80  // "telefone é (11) 8765-4321"
        } else {
            entityMatch = 0.65  // 8-9 digits or no context
        }

        return EnhancedConfidence(entityMatch: entityMatch)
    }

    /// Score a knowledge base matched entity
    static func scoreKnowledgeBaseEntity(
        matchType: KBMatchType,
        fuzzyMatchScore: Double = 1.0
    ) -> EnhancedConfidence {
        let entityMatch: Double
        switch matchType {
        case .exactCanonical:
            entityMatch = 0.95
        case .exactVariation:
            entityMatch = 0.92
        case .fuzzyMatch:
            entityMatch = 0.80 * fuzzyMatchScore
        case .componentMatch:
            entityMatch = 0.75
        }

        return EnhancedConfidence(entityMatch: entityMatch)
    }

    /// Score a pattern-matched entity
    static func scorePatternEntity(
        patternConfidence: Double,
        validationPassed: Bool
    ) -> EnhancedConfidence {
        let entityMatch = validationPassed ? patternConfidence : patternConfidence * 0.85
        return EnhancedConfidence(entityMatch: entityMatch)
    }

    /// Score an LLM-extracted entity
    static func scoreLLMEntity(
        hasKBValidation: Bool,
        llmConfidence: Double = 0.85
    ) -> EnhancedConfidence {
        let entityMatch = hasKBValidation ? llmConfidence * 1.05 : llmConfidence
        let contextConsistency = hasKBValidation ? 0.95 : 0.80

        return EnhancedConfidence(
            entityMatch: min(entityMatch, 1.0),
            contextConsistency: contextConsistency
        )
    }

    // MARK: - Private Validation Helpers

    private static func validateDateConsistency(_ dateString: String) -> Double {
        // Check if date is not in the past (for surgery scheduling)
        // For now, return high consistency if basic format looks valid
        if dateString.contains("/") || dateString.contains("-") {
            return 0.90
        }
        return 0.85
    }

    private static func validateTimeConsistency(_ timeString: String) -> Double {
        // Check if time is within business hours (7:00 - 20:00 for surgeries)
        // Extract hour if possible
        let components = timeString.components(separatedBy: CharacterSet(charactersIn: ":hH "))
        if let hourStr = components.first, let hour = Int(hourStr) {
            if (7...20).contains(hour) {
                return 0.95  // Within typical surgery hours
            } else {
                return 0.70  // Outside typical hours (might be correct but unusual)
            }
        }
        return 0.85  // Can't parse, assume reasonable
    }

    enum KBMatchType {
        case exactCanonical     // Exact match to canonical name
        case exactVariation     // Exact match to known variation
        case fuzzyMatch         // Fuzzy match with score
        case componentMatch     // Component-based match
    }
}
