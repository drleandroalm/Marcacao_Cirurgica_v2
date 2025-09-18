import Foundation

class DurationFormatter {
    // Converts Portuguese duration phrases to HH:MM (e.g., "uma hora e meia" -> "01:30")
    static func format(_ input: String) -> String {
        let text = input.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return input }

        // Quick path: already HH:MM
        if let hhmm = matchHHMM(text) { return hhmm }

        var hours = 0
        var minutes = 0

        // Handle common expressions
        if text.contains("meia") { minutes += 30 }
        if text.contains("quarto") { minutes += 15 }
        if text.contains("quinze") { minutes += 15 }
        if text.contains("trinta") { minutes = max(minutes, 30) }

        // Extract numbers that could be hours/minutes
        let numbers = text.components(separatedBy: CharacterSet.decimalDigits.inverted).compactMap { Int($0) }
        if numbers.count >= 1 {
            // Heuristic: first number is hours if words imply hours
            if text.contains("hora") {
                hours = numbers[0]
                if numbers.count >= 2 { minutes = numbers[1] }
            } else if numbers.count >= 2 {
                hours = numbers[0]; minutes = numbers[1]
            } else {
                // Single number with no unit → treat as minutes if <= 59, else hours
                if numbers[0] <= 59 { minutes = numbers[0] } else { hours = numbers[0] }
            }
        } else {
            // Words-only numbers for hours
            if let h = wordToNumber(text) { hours = h }
            // Words-only minutes when explicit
            if text.contains("quinze") { minutes = max(minutes, 15) }
            if text.contains("trinta") { minutes = max(minutes, 30) }
            if text.contains("quarenta e cinco") { minutes = max(minutes, 45) }
        }

        // Clamp
        minutes = minutes % 60
        hours = max(0, min(23, hours))

        return String(format: "%02d:%02d", hours, minutes)
    }

    private static func matchHHMM(_ s: String) -> String? {
        let pattern = #"\b(\d{1,2})[:hH](\d{2})\b"#
        if let re = try? NSRegularExpression(pattern: pattern),
           let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
           let r1 = Range(m.range(at: 1), in: s), let r2 = Range(m.range(at: 2), in: s) {
            let h = Int(s[r1]) ?? 0
            let m = Int(s[r2]) ?? 0
            return String(format: "%02d:%02d", h, m)
        }
        return nil
    }

    private static func wordToNumber(_ s: String) -> Int? {
        let mapping: [String: Int] = [
            "uma": 1, "um": 1, "duas": 2, "dois": 2, "três": 3, "quatro": 4, "cinco": 5, "seis": 6,
            "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11, "doze": 12
        ]
        for (w, n) in mapping where s.contains(w) { return n }
        return nil
    }
}
