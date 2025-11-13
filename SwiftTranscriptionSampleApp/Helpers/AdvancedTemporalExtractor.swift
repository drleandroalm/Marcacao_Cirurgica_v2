import Foundation

/// Advanced temporal expression parser for Brazilian Portuguese
struct AdvancedTemporalExtractor: Sendable {

    /// Extract date from advanced temporal expressions
    /// - Parameter text: Input text in Brazilian Portuguese
    /// - Returns: Tuple of (Date, confidence, matched expression) if found
    static func extractAdvancedDate(from text: String) -> (date: Date, confidence: Double, expression: String)? {
        let lowercased = text.lowercased()

        // Try relative days patterns
        if let result = extractRelativeDays(from: lowercased) {
            return result
        }

        // Try relative weeks patterns
        if let result = extractRelativeWeeks(from: lowercased) {
            return result
        }

        // Try next day of month patterns
        if let result = extractNextDayOfMonth(from: lowercased) {
            return result
        }

        return nil
    }

    // MARK: - Private Pattern Extractors

    /// Extract "daqui a X dias" or "em X dias"
    private static func extractRelativeDays(from text: String) -> (Date, Double, String)? {
        let patterns = [
            (#"daqui\s+a\s+(\d+)\s+dias?"#, 0.88),
            (#"em\s+(\d+)\s+dias?"#, 0.85),
            (#"dentro\s+de\s+(\d+)\s+dias?"#, 0.85)
        ]

        for (pattern, confidence) in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)),
               match.numberOfRanges > 1 {

                let daysRange = match.range(at: 1)
                if let range = Range(daysRange, in: text),
                   let days = Int(text[range]) {

                    if let targetDate = Calendar.current.date(byAdding: .day, value: days, to: Date()) {
                        let matchedText = String(text[Range(match.range, in: text)!])
                        return (targetDate, confidence, matchedText)
                    }
                }
            }
        }

        return nil
    }

    /// Extract "daqui a X semanas" or "em X semanas"
    private static func extractRelativeWeeks(from text: String) -> (Date, Double, String)? {
        let patterns = [
            (#"daqui\s+a\s+(\d+)\s+semanas?"#, 0.88),
            (#"em\s+(\d+)\s+semanas?"#, 0.85),
            (#"dentro\s+de\s+(\d+)\s+semanas?"#, 0.85)
        ]

        for (pattern, confidence) in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)),
               match.numberOfRanges > 1 {

                let weeksRange = match.range(at: 1)
                if let range = Range(weeksRange, in: text),
                   let weeks = Int(text[range]) {

                    let days = weeks * 7
                    if let targetDate = Calendar.current.date(byAdding: .day, value: days, to: Date()) {
                        let matchedText = String(text[Range(match.range, in: text)!])
                        return (targetDate, confidence, matchedText)
                    }
                }
            }
        }

        return nil
    }

    /// Extract "próximo dia 15" or "dia 15 do próximo mês"
    private static func extractNextDayOfMonth(from text: String) -> (Date, Double, String)? {
        let patterns = [
            (#"próxim[ao]\s+dia\s+(\d{1,2})"#, 0.83),
            (#"dia\s+(\d{1,2})\s+do\s+próximo\s+mês"#, 0.85),
            (#"no\s+dia\s+(\d{1,2})\s+do\s+próximo"#, 0.82)
        ]

        for (pattern, confidence) in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)),
               match.numberOfRanges > 1 {

                let dayRange = match.range(at: 1)
                if let range = Range(dayRange, in: text),
                   let dayOfMonth = Int(text[range]),
                   (1...31).contains(dayOfMonth) {

                    if let targetDate = nextOccurrence(ofDay: dayOfMonth) {
                        let matchedText = String(text[Range(match.range, in: text)!])
                        return (targetDate, confidence, matchedText)
                    }
                }
            }
        }

        return nil
    }

    /// Calculate next occurrence of specific day of month
    private static func nextOccurrence(ofDay day: Int) -> Date? {
        let calendar = Calendar.current
        let today = Date()

        guard let currentDay = calendar.dateComponents([.day], from: today).day else {
            return nil
        }

        // If the target day hasn't occurred this month yet, use this month
        if day > currentDay {
            var components = calendar.dateComponents([.year, .month], from: today)
            components.day = day
            return calendar.date(from: components)
        }

        // Otherwise, use next month
        guard let nextMonth = calendar.date(byAdding: .month, value: 1, to: today) else {
            return nil
        }

        var components = calendar.dateComponents([.year, .month], from: nextMonth)
        components.day = day

        // Validate the day exists in that month (e.g., February 30 doesn't exist)
        if let targetDate = calendar.date(from: components),
           calendar.component(.day, from: targetDate) == day {
            return targetDate
        }

        return nil
    }

    // MARK: - Formatting Helpers

    /// Format extracted date to standard Brazilian format (dd/MM/yyyy)
    static func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd/MM/yyyy"
        formatter.locale = Locale(identifier: "pt_BR")
        return formatter.string(from: date)
    }

    /// Extract all temporal expressions from text with their dates and confidences
    static func extractAllTemporalExpressions(from text: String) -> [(date: Date, confidence: Double, expression: String)] {
        var results: [(Date, Double, String)] = []

        // Check for advanced patterns
        if let advanced = extractAdvancedDate(from: text) {
            results.append(advanced)
        }

        // Could be extended to include basic patterns (hoje, amanhã, etc.)
        // Those are already handled in EntityExtractor.swift

        return results
    }
}

// MARK: - Testing Helper

#if DEBUG
extension AdvancedTemporalExtractor {
    /// Test the temporal extractor with sample inputs
    static func runTests() {
        let testCases = [
            "A cirurgia será daqui a 3 dias",
            "Agendar em 5 dias",
            "Daqui a 2 semanas",
            "Próximo dia 15",
            "Dia 20 do próximo mês",
            "Dentro de 10 dias"
        ]

        print("🧪 Testing AdvancedTemporalExtractor:")
        for testCase in testCases {
            if let result = extractAdvancedDate(from: testCase) {
                let formatted = formatDate(result.date)
                print("  ✅ \"\(testCase)\" → \(formatted) (confidence: \(result.confidence), matched: \"\(result.expression)\")")
            } else {
                print("  ❌ \"\(testCase)\" → No match")
            }
        }
    }
}
#endif
