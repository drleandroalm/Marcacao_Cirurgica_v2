import Foundation

class TranscriptionProcessor {
    
    // Portuguese number words to digits mapping
    private static let numberWords: [String: String] = [
        // Units
        "zero": "0", "um": "1", "uma": "1", "dois": "2", "duas": "2",
        "três": "3", "quatro": "4", "cinco": "5", "seis": "6",
        "sete": "7", "oito": "8", "nove": "9",
        
        // 10-19
        "dez": "10", "onze": "11", "doze": "12", "treze": "13",
        "catorze": "14", "quatorze": "14", "quinze": "15", "dezesseis": "16",
        "dezessete": "17", "dezoito": "18", "dezenove": "19",
        
        // Tens
        "vinte": "20", "trinta": "30", "quarenta": "40",
        "cinquenta": "50", "sessenta": "60", "setenta": "70",
        "oitenta": "80", "noventa": "90",
        
        // Hundred
        "cem": "100"
    ]
    
    // Month names to numbers
    private static let monthNames: [String: String] = [
        "janeiro": "01", "fevereiro": "02", "março": "03",
        "abril": "04", "maio": "05", "junho": "06",
        "julho": "07", "agosto": "08", "setembro": "09",
        "outubro": "10", "novembro": "11", "dezembro": "12"
    ]
    
    // Main processing function
    static func processText(_ text: String, fieldType: FieldType) -> String {
        var processedText = text.lowercased()
        
        // Remove punctuation for all field types
        processedText = removePunctuation(processedText)
        
        // Apply field-specific processing
        switch fieldType {
        case .age:
            processedText = convertPortugueseNumberToDigits(processedText)
        case .number:
            processedText = convertPortugueseNumberToDigits(processedText)
        case .date:
            processedText = convertPortugueseDateToFormat(processedText)
        case .phone:
            processedText = cleanPhoneNumber(processedText)
        case .time:
            processedText = convertTimeFormat(processedText)
        case .duration:
            processedText = DurationFormatter.format(processedText)
        default:
            break
        }
        
        return processedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    // Remove punctuation marks
    private static func removePunctuation(_ text: String) -> String {
        let punctuationCharacters = CharacterSet(charactersIn: ".,;:!?\"'")
        let components = text.components(separatedBy: punctuationCharacters)
        return components.joined(separator: " ").replacingOccurrences(of: "  ", with: " ")
    }
    
    // Convert Portuguese number words to digits
    private static func convertPortugueseNumberToDigits(_ text: String) -> String {
        var result = text
        let words = text.split(separator: " ").map { String($0) }
        
        // Handle composite numbers (e.g., "vinte e três" -> "23")
        if words.count >= 3 {
            for i in 0..<words.count-2 {
                if words[i+1] == "e" {
                    if let tens = numberWords[words[i]], let units = numberWords[words[i+2]] {
                        if let tensNum = Int(tens), let unitsNum = Int(units) {
                            let composite = tensNum + unitsNum
                            result = result.replacingOccurrences(
                                of: "\(words[i]) e \(words[i+2])",
                                with: "\(composite)"
                            )
                        }
                    }
                }
            }
        }
        
        // Replace single number words
        for (word, digit) in numberWords {
            result = result.replacingOccurrences(of: word, with: digit)
        }
        
        // Remove the word "anos" for age fields
        result = result.replacingOccurrences(of: "anos", with: "")
        
        // Keep only numbers and spaces
        let allowedCharacters = CharacterSet(charactersIn: "0123456789 ")
        result = String(result.unicodeScalars.filter { allowedCharacters.contains($0) })
        
        return result.trimmingCharacters(in: .whitespaces)
    }
    
    // Convert Portuguese date speech to DD/MM/YYYY format
    private static func convertPortugueseDateToFormat(_ text: String) -> String {
        var result = text
        let words = text.split(separator: " ").map { String($0) }
        
        var day = ""
        var month = ""
        var year = ""
        
        // Try to infer day using patterns like "vinte e sete de setembro"
        for i in 0..<words.count {
            // Month detection
            if let monthNum = monthNames[words[i]] {
                month = monthNum
                // Look backwards for day tokens in pattern: "X e Y de <month>"
                if i >= 4 && words[i-1] == "de",
                   let d1s = numberWords[words[i-4]], let d1 = Int(d1s),
                   let d2s = numberWords[words[i-2]], let d2 = Int(d2s) {
                    let d = d1 + d2
                    if d >= 1 && d <= 31 { day = String(d) }
                } else if i >= 2 {
                    // Try single token before month or before 'de'
                    let candidate = (words[i-1] == "de" ? words[i-2] : words[i-1])
                    if let d = Int(candidate), d >= 1 && d <= 31 {
                        day = String(d)
                    } else if let dWord = numberWords[candidate], let d = Int(dWord), d >= 1 && d <= 31 {
                        day = String(d)
                    }
                } else if i >= 1 {
                    if let d = Int(words[i-1]), d >= 1 && d <= 31 {
                        day = String(d)
                    } else if let dWord = numberWords[words[i-1]], let d = Int(dWord), d >= 1 && d <= 31 {
                        day = String(d)
                    }
                }
            }
            // Look for year (4-digit number)
            if words[i].count == 4, let _ = Int(words[i]) {
                year = words[i]
            }
        }
        
        // Handle year in words (e.g., "dois mil e vinte e cinco")
        if year.isEmpty {
            if let doisMilIndex = words.firstIndex(of: "dois"),
               doisMilIndex + 1 < words.count && words[doisMilIndex + 1] == "mil" {
                var yearValue = 2000
                
                // Look for additional year components after "dois mil e"
                if doisMilIndex + 3 < words.count && words[doisMilIndex + 2] == "e" {
                    // Handle "dois mil e vinte"
                    if let twenties = numberWords[words[doisMilIndex + 3]] {
                        yearValue += Int(twenties) ?? 0
                        
                        // Handle "dois mil e vinte e cinco"
                        if doisMilIndex + 5 < words.count && words[doisMilIndex + 4] == "e" {
                            if let units = numberWords[words[doisMilIndex + 5]] {
                                yearValue += Int(units) ?? 0
                            }
                        }
                    }
                }
                year = String(yearValue)
            }
        }
        
        // Weekday handling (e.g., "próxima segunda")
        if day.isEmpty && month.isEmpty {
            if let computed = computeWeekdayDate(from: text) {
                let df = DateFormatter(); df.dateFormat = "dd/MM/yyyy"
                return df.string(from: computed)
            }
        }

        // Format as DD/MM/YYYY if all components found
        if !day.isEmpty && !month.isEmpty && !year.isEmpty {
            // Pad day and month with zeros if needed
            if day.count == 1 {
                day = "0" + day
            }
            result = "\(day)/\(month)/\(year)"
        } else {
            // Try to extract date pattern from numbers already in text
            let pattern = #"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})"#
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: result, range: NSRange(result.startIndex..., in: result)) {
                let dayRange = Range(match.range(at: 1), in: result)
                let monthRange = Range(match.range(at: 2), in: result)
                let yearRange = Range(match.range(at: 3), in: result)
                
                if let dayRange = dayRange, let monthRange = monthRange, let yearRange = yearRange {
                    day = String(result[dayRange])
                    month = String(result[monthRange])
                    year = String(result[yearRange])
                    
                    // Pad with zeros if needed
                    if day.count == 1 { day = "0" + day }
                    if month.count == 1 { month = "0" + month }
                    if year.count == 2 { year = "20" + year }
                    
                    result = "\(day)/\(month)/\(year)"
                }
            }
        }
        
        return result
    }

    static func computeWeekdayDate(from text: String) -> Date? {
        let lower = text.lowercased()
        let weekdays: [String: Int] = [
            "domingo": 1, "segunda": 2, "segunda-feira": 2, "terça": 3, "terca": 3, "terça-feira": 3,
            "quarta": 4, "quarta-feira": 4, "quinta": 5, "quinta-feira": 5, "sexta": 6, "sexta-feira": 6,
            "sábado": 7, "sabado": 7
        ]
        let isNext = lower.contains("próxima") || lower.contains("proxima")
        guard let match = weekdays.first(where: { lower.contains($0.key) }) else { return nil }
        let idx = match.value
        let cal = Calendar.current
        let today = Date()
        var next = today
        let target = idx
        let todayIdx = cal.component(.weekday, from: today)
        var add = (target - todayIdx + 7) % 7
        if add == 0 { add = isNext ? 7 : 0 }
        else if !isNext && add < 0 { add += 7 }
        if add > 0 { next = cal.date(byAdding: .day, value: add, to: today)! }
        return next
    }
    
    // Clean phone number (remove non-digits, format if needed)
    private static func cleanPhoneNumber(_ text: String) -> String {
        // Keep only digits and spaces
        let digitsOnly = text.filter { $0.isNumber || $0 == " " }
        
        // Remove spaces and return just the digits
        return digitsOnly.replacingOccurrences(of: " ", with: "")
    }
    
    // Convert time format (e.g., "três horas e trinta minutos" -> "03:30")
    private static func convertTimeFormat(_ text: String) -> String {
        var result = text
        let words = text.split(separator: " ").map { String($0) }
        
        var hours = ""
        var minutes = ""
        
        for i in 0..<words.count {
            // Look for hours
            if i + 1 < words.count && (words[i + 1] == "horas" || words[i + 1] == "hora") {
                if let hourNum = numberWords[words[i]] {
                    hours = hourNum
                } else if let _ = Int(words[i]) {
                    hours = words[i]
                }
            }
            
            // Look for minutes
            if i + 1 < words.count && (words[i + 1] == "minutos" || words[i + 1] == "minuto") {
                if let minNum = numberWords[words[i]] {
                    minutes = minNum
                } else if let _ = Int(words[i]) {
                    minutes = words[i]
                }
            }
        }
        
        // If we found hours and/or minutes, format them
        if !hours.isEmpty {
            if hours.count == 1 {
                hours = "0" + hours
            }
            if minutes.isEmpty {
                minutes = "00"
            } else if minutes.count == 1 {
                minutes = "0" + minutes
            }
            result = "\(hours):\(minutes)"
        } else {
            // Try to extract time pattern already in text
            let pattern = #"(\d{1,2}):(\d{2})"#
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: result, range: NSRange(result.startIndex..., in: result)) {
                let range = Range(match.range, in: result)
                if let range = range {
                    result = String(result[range])
                }
            }
        }
        
        return result
    }
}
