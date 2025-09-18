import Foundation

enum FieldType: Sendable {
    case text
    case number
    case date
    case time
    case phone
    case age
    case duration
}

struct TemplateField: Identifiable, Sendable {
    let id: String
    let label: String
    let placeholder: String
    var value: String
    let fieldType: FieldType
    let isRequired: Bool
    
    init(id: String, label: String, placeholder: String, value: String = "", fieldType: FieldType = .text, isRequired: Bool = true) {
        self.id = id
        self.label = label
        self.placeholder = placeholder
        self.value = value
        self.fieldType = fieldType
        self.isRequired = isRequired
    }
    
    var isComplete: Bool {
        return !value.isEmpty
    }
    
    func validate() -> Bool {
        guard isRequired && !value.isEmpty else {
            return !isRequired
        }
        
        switch fieldType {
        case .age:
            let digits = value.filter { $0.isNumber }
            if let v = Int(digits), v >= 0, v <= 120 { return true }
            return false
        case .number:
            return Int(value) != nil
        case .date:
            let components = value.split(separator: "/")
            guard components.count == 3,
                  let day = Int(components[0]), day >= 1, day <= 31,
                  let month = Int(components[1]), month >= 1, month <= 12,
                  let year = Int(components[2]), year >= 2024, year <= 2100 else {
                return false
            }
            return true
        case .time:
            let components = value.split(separator: ":")
            guard components.count == 2,
                  let hour = Int(components[0]), hour >= 0, hour <= 23,
                  let minute = Int(components[1]), minute >= 0, minute <= 59 else {
                return false
            }
            return true
        case .phone:
            let cleaned = value.filter { $0.isNumber }
            return cleaned.count >= 10 && cleaned.count <= 11
        case .duration:
            let components = value.split(separator: ":")
            guard components.count == 2,
                  let hour = Int(components[0]), hour >= 0, hour <= 23,
                  let minute = Int(components[1]), minute >= 0, minute <= 59 else {
                return false
            }
            return true
        case .text:
            return !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }
    
    func formattedValue() -> String {
        switch fieldType {
        case .age:
            let digits = value.filter { $0.isNumber }
            if let v = Int(digits) { return "\(v) anos" }
            return value
        case .date:
            // Normalize various spoken formats into dd/MM/yyyy
            let normalized = TranscriptionProcessor.processText(value, fieldType: .date)
            // Ensure final shape dd/MM/yyyy
            let comps = normalized.replacingOccurrences(of: "-", with: "/").split(separator: "/")
            if comps.count == 3 {
                let d = String(comps[0]).padding(toLength: 2, withPad: "0", startingAt: 0)
                let m = String(comps[1]).padding(toLength: 2, withPad: "0", startingAt: 0)
                var y = String(comps[2])
                if y.count == 2 { y = "20" + y }
                return "\(d)/\(m)/\(y)"
            }
            return normalized
        case .time:
            return MilitaryTimeFormatter.format(value)
        case .phone:
            var cleaned = value.filter { $0.isNumber }
            // Remove trunk prefix 0 (e.g., 031 -> 31)
            if cleaned.first == "0" { cleaned.removeFirst() }
            if cleaned.count == 11 {
                let formatted = "(\(cleaned.prefix(2))) \(cleaned.dropFirst(2).prefix(5))-\(cleaned.suffix(4))"
                return formatted
            } else if cleaned.count == 10 {
                let formatted = "(\(cleaned.prefix(2))) \(cleaned.dropFirst(2).prefix(4))-\(cleaned.suffix(4))"
                return formatted
            } else if cleaned.count == 9 { // mobile without DDD
                // Format as placeholder DDD + 9-digit number pattern
                let first = cleaned.prefix(5)
                let last = cleaned.suffix(4)
                return "(DDD) \(first)-\(last)"
            } else if cleaned.count == 8 { // landline without DDD
                let first = cleaned.prefix(4)
                let last = cleaned.suffix(4)
                return "(DDD) \(first)-\(last)"
            }
            return value
        case .duration:
            return DurationFormatter.format(value)
        default:
            return value
        }
    }
}
