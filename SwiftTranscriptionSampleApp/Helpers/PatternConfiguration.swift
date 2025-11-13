import Foundation

/// Pattern configuration loader for externalized regex patterns
struct PatternConfiguration: Sendable {
    let version: String
    let locale: String
    let patterns: [String: EntityPatternConfig]

    static let shared: PatternConfiguration = loadConfiguration()

    private static func loadConfiguration() -> PatternConfiguration {
        guard let url = Bundle.main.url(
            forResource: "extraction_patterns",
            withExtension: "json",
            subdirectory: "Resources/Patterns/pt-BR"
        ) ?? Bundle.main.url(
            forResource: "extraction_patterns",
            withExtension: "json",
            subdirectory: "Patterns/pt-BR"
        ) else {
            print("⚠️ PatternConfiguration: extraction_patterns.json not found, using defaults")
            return defaultConfiguration()
        }

        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            let config = try decoder.decode(PatternConfigurationJSON.self, from: data)
            return config.toPatternConfiguration()
        } catch {
            print("⚠️ PatternConfiguration: failed to load patterns: \(error)")
            return defaultConfiguration()
        }
    }

    private static func defaultConfiguration() -> PatternConfiguration {
        // Minimal fallback configuration
        return PatternConfiguration(
            version: "1.0.0",
            locale: "pt-BR",
            patterns: [:]
        )
    }
}

struct EntityPatternConfig: Sendable {
    let regexPatterns: [RegexPattern]
    let contextKeywords: [ContextKeyword]
    let relativeKeywords: [RelativeKeyword]
    let advancedPatterns: [AdvancedPattern]
    let validation: ValidationRules?
    let useKnowledgeBase: Bool
    let minimumMatchScore: Double?
    let weekdaySupport: Bool

    init(
        regexPatterns: [RegexPattern] = [],
        contextKeywords: [ContextKeyword] = [],
        relativeKeywords: [RelativeKeyword] = [],
        advancedPatterns: [AdvancedPattern] = [],
        validation: ValidationRules? = nil,
        useKnowledgeBase: Bool = false,
        minimumMatchScore: Double? = nil,
        weekdaySupport: Bool = false
    ) {
        self.regexPatterns = regexPatterns
        self.contextKeywords = contextKeywords
        self.relativeKeywords = relativeKeywords
        self.advancedPatterns = advancedPatterns
        self.validation = validation
        self.useKnowledgeBase = useKnowledgeBase
        self.minimumMatchScore = minimumMatchScore
        self.weekdaySupport = weekdaySupport
    }
}

struct RegexPattern: Sendable {
    let pattern: String
    let confidence: Double
}

struct ContextKeyword: Sendable {
    let keyword: String
    let wordOffset: [Int]  // [start, end] word offsets
    let confidence: Double
}

struct RelativeKeyword: Sendable {
    let keyword: String
    let daysOffset: Int
    let confidence: Double
}

struct AdvancedPattern: Sendable {
    let pattern: String
    let type: String
    let confidence: Double
}

struct ValidationRules: Sendable {
    let minLength: Int?
    let maxLength: Int?
    let minValue: Int?
    let maxValue: Int?
    let minDigits: Int?
    let maxDigits: Int?
    let format: String?
    let capitalizedRequired: Bool?
    let hourMin: Int?
    let hourMax: Int?
    let minuteMin: Int?
    let minuteMax: Int?
    let minMinutes: Int?
    let maxMinutes: Int?
}

// MARK: - JSON Decodable Types

private struct PatternConfigurationJSON: Decodable {
    let version: String
    let locale: String
    let description: String?
    let patterns: [String: EntityPatternConfigJSON]

    func toPatternConfiguration() -> PatternConfiguration {
        let convertedPatterns = patterns.mapValues { $0.toEntityPatternConfig() }
        return PatternConfiguration(
            version: version,
            locale: locale,
            patterns: convertedPatterns
        )
    }
}

private struct EntityPatternConfigJSON: Decodable {
    let regexPatterns: [RegexPatternJSON]?
    let contextKeywords: [ContextKeywordJSON]?
    let relativeKeywords: [RelativeKeywordJSON]?
    let advancedPatterns: [AdvancedPatternJSON]?
    let validation: ValidationRulesJSON?
    let useKnowledgeBase: Bool?
    let minimumMatchScore: Double?
    let weekdaySupport: Bool?

    func toEntityPatternConfig() -> EntityPatternConfig {
        return EntityPatternConfig(
            regexPatterns: regexPatterns?.map { $0.toRegexPattern() } ?? [],
            contextKeywords: contextKeywords?.map { $0.toContextKeyword() } ?? [],
            relativeKeywords: relativeKeywords?.map { $0.toRelativeKeyword() } ?? [],
            advancedPatterns: advancedPatterns?.map { $0.toAdvancedPattern() } ?? [],
            validation: validation?.toValidationRules(),
            useKnowledgeBase: useKnowledgeBase ?? false,
            minimumMatchScore: minimumMatchScore,
            weekdaySupport: weekdaySupport ?? false
        )
    }
}

private struct RegexPatternJSON: Decodable {
    let pattern: String
    let confidence: Double

    func toRegexPattern() -> RegexPattern {
        return RegexPattern(pattern: pattern, confidence: confidence)
    }
}

private struct ContextKeywordJSON: Decodable {
    let keyword: String
    let wordOffset: [Int]
    let confidence: Double

    func toContextKeyword() -> ContextKeyword {
        return ContextKeyword(keyword: keyword, wordOffset: wordOffset, confidence: confidence)
    }
}

private struct RelativeKeywordJSON: Decodable {
    let keyword: String
    let daysOffset: Int
    let confidence: Double

    func toRelativeKeyword() -> RelativeKeyword {
        return RelativeKeyword(keyword: keyword, daysOffset: daysOffset, confidence: confidence)
    }
}

private struct AdvancedPatternJSON: Decodable {
    let pattern: String
    let type: String
    let confidence: Double

    func toAdvancedPattern() -> AdvancedPattern {
        return AdvancedPattern(pattern: pattern, type: type, confidence: confidence)
    }
}

private struct ValidationRulesJSON: Decodable {
    let minLength: Int?
    let maxLength: Int?
    let minValue: Int?
    let maxValue: Int?
    let minDigits: Int?
    let maxDigits: Int?
    let format: String?
    let capitalizedRequired: Bool?
    let hourMin: Int?
    let hourMax: Int?
    let minuteMin: Int?
    let minuteMax: Int?
    let minMinutes: Int?
    let maxMinutes: Int?

    func toValidationRules() -> ValidationRules {
        return ValidationRules(
            minLength: minLength,
            maxLength: maxLength,
            minValue: minValue,
            maxValue: maxValue,
            minDigits: minDigits,
            maxDigits: maxDigits,
            format: format,
            capitalizedRequired: capitalizedRequired,
            hourMin: hourMin,
            hourMax: hourMax,
            minuteMin: minuteMin,
            minuteMax: minuteMax,
            minMinutes: minMinutes,
            maxMinutes: maxMinutes
        )
    }
}
