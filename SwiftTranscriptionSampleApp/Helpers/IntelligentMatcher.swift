import Foundation

enum MatchType: Sendable {
    case exact(score: Double)
    case fuzzyClose(score: Double)
    case phonetic(score: Double)
    case partial(score: Double)
    case contextual(score: Double)
    case unknown(score: Double)
    
    var score: Double {
        switch self {
        case .exact(let score), .fuzzyClose(let score), .phonetic(let score),
             .partial(let score), .contextual(let score), .unknown(let score):
            return score
        }
    }
    
    var description: String {
        switch self {
        case .exact: return "Correspondência Exata"
        case .fuzzyClose: return "Correspondência Próxima"
        case .phonetic: return "Correspondência Fonética"
        case .partial: return "Correspondência Parcial"
        case .contextual: return "Correspondência Contextual"
        case .unknown: return "Entidade Desconhecida"
        }
    }
}

struct MatchResult: Sendable {
    let value: String
    let matchType: MatchType
    let confidence: Double
    let alternatives: [String]
    let isKnownEntity: Bool
}

class IntelligentMatcher {
    private static func redacted(_ text: String) -> String {
        "<len=\(text.count)>"
    }
    
    // MARK: - Main Matching Method
    static func matchSurgeon(_ input: String, context: String? = nil) -> MatchResult {
        print("🔍 IntelligentMatcher: Matching surgeon for \(redacted(input))")
        let normalizedInput = normalize(input)
        var bestMatch: (surgeon: SurgeonEntity?, score: Double, type: MatchType)?
        
        for surgeon in MedicalKnowledgeBase.surgeons {
            // Exact match
            if matchesExactly(normalizedInput, surgeon: surgeon) {
                return MatchResult(
                    value: surgeon.canonical,
                    matchType: .exact(score: 100),
                    confidence: 1.0,
                    alternatives: [],
                    isKnownEntity: true
                )
            }
            
            // Fuzzy match
            let fuzzyScore = calculateFuzzyScore(normalizedInput, surgeon: surgeon)
            if fuzzyScore > 0.92 {
                let matchScore = fuzzyScore * 100
                let currentMatch = (surgeon, matchScore, MatchType.fuzzyClose(score: matchScore))
                if bestMatch == nil || matchScore > bestMatch!.score {
                    bestMatch = currentMatch
                }
            }
            
            // Phonetic match
            let phoneticScore = calculatePhoneticScore(normalizedInput, surgeon: surgeon)
            if phoneticScore > 0.88 {
                let matchScore = phoneticScore * 100
                let currentMatch = (surgeon, matchScore, MatchType.phonetic(score: matchScore))
                if bestMatch == nil || matchScore > bestMatch!.score {
                    bestMatch = currentMatch
                }
            }
            
            // Partial match
            if containsPartialMatch(normalizedInput, surgeon: surgeon) {
                let matchScore = 75.0
                let currentMatch = (surgeon, matchScore, MatchType.partial(score: matchScore))
                if bestMatch == nil || matchScore > bestMatch!.score {
                    bestMatch = currentMatch
                }
            }
        }
        
        // Context boost
        if let match = bestMatch, let context = context {
            if context.lowercased().contains(match.surgeon!.canonical.lowercased()) {
                let boostedScore = min(match.score + 10, 100)
                bestMatch = (match.surgeon, boostedScore, MatchType.contextual(score: boostedScore))
            }
        }
        
        if let match = bestMatch {
            let alternatives = findAlternativeSurgeons(for: normalizedInput, excluding: match.surgeon!)
            return MatchResult(
                value: match.surgeon!.canonical,
                matchType: match.type,
                confidence: match.score / 100.0,
                alternatives: alternatives,
                isKnownEntity: true
            )
        }
        
        // Unknown entity
        print("⚠️ IntelligentMatcher: No match found for surgeon \(redacted(input)) - treating as unknown")
        return MatchResult(
            value: capitalizeProperName(input),
            matchType: .unknown(score: 60),
            confidence: 0.6,
            alternatives: [],
            isKnownEntity: false
        )
    }
    
    static func matchProcedure(_ input: String, context: String? = nil) -> MatchResult {
        print("🔍 IntelligentMatcher: Matching procedure for \(redacted(input))")
        let normalizedInput = normalize(input)
        let expandedInput = expandAbbreviations(normalizedInput)
        print("📝 Expanded input: \(redacted(expandedInput))")
        
        var bestMatch: (procedure: ProcedureEntity?, score: Double, type: MatchType)?
        
        for procedure in MedicalKnowledgeBase.procedures {
            // Exact match (including abbreviations)
            if matchesExactly(expandedInput, procedure: procedure) {
                return MatchResult(
                    value: procedure.canonical,
                    matchType: .exact(score: 100),
                    confidence: 1.0,
                    alternatives: [],
                    isKnownEntity: true
                )
            }
            
            // Fuzzy match
            let fuzzyScore = calculateFuzzyScore(expandedInput, procedure: procedure)
            if fuzzyScore > 0.92 {
                let matchScore = fuzzyScore * 100
                let currentMatch = (procedure, matchScore, MatchType.fuzzyClose(score: matchScore))
                if bestMatch == nil || matchScore > bestMatch!.score {
                    bestMatch = currentMatch
                }
            }
            
            // Component match
            let componentScore = calculateComponentScore(expandedInput, procedure: procedure)
            if componentScore > 0.85 {
                let matchScore = componentScore * 100
                let currentMatch = (procedure, matchScore, MatchType.partial(score: matchScore))
                if bestMatch == nil || matchScore > bestMatch!.score {
                    bestMatch = currentMatch
                }
            }
        }
        
        if let match = bestMatch {
            let alternatives = findAlternativeProcedures(for: expandedInput, excluding: match.procedure!)
            return MatchResult(
                value: match.procedure!.canonical,
                matchType: match.type,
                confidence: match.score / 100.0,
                alternatives: alternatives,
                isKnownEntity: true
            )
        }
        
        // Check for compound procedures
        if MedicalKnowledgeBase.isCompoundProcedure(input) {
            let components = parseCompoundProcedure(input)
            if !components.isEmpty {
                let compoundName = components.joined(separator: " + ")
                return MatchResult(
                    value: compoundName,
                    matchType: .contextual(score: 92),
                    confidence: 0.92,
                    alternatives: [],
                    isKnownEntity: true
                )
            }
        }
        
        // Unknown procedure
        return MatchResult(
            value: capitalizeProperName(input),
            matchType: .unknown(score: 60),
            confidence: 0.6,
            alternatives: [],
            isKnownEntity: false
        )
    }
    
    // MARK: - Matching Helpers
    private static func matchesExactly(_ input: String, surgeon: SurgeonEntity) -> Bool {
        let lowercased = input.lowercased()
        
        if surgeon.canonical.lowercased() == lowercased {
            return true
        }
        
        for variation in surgeon.variations {
            if variation.lowercased() == lowercased {
                return true
            }
        }
        
        return false
    }
    
    private static func matchesExactly(_ input: String, procedure: ProcedureEntity) -> Bool {
        let lowercased = input.lowercased()
        
        if procedure.canonical.lowercased() == lowercased {
            return true
        }
        
        for abbreviation in procedure.abbreviations {
            if abbreviation.lowercased() == lowercased {
                return true
            }
        }
        
        for variation in procedure.spokenVariations {
            if variation.lowercased() == lowercased {
                return true
            }
        }
        
        return false
    }
    
    // MARK: - Levenshtein Distance
    private static func levenshteinDistance(_ s1: String, _ s2: String) -> Int {
        let s1Array = Array(s1)
        let s2Array = Array(s2)
        
        var matrix = [[Int]](repeating: [Int](repeating: 0, count: s2Array.count + 1), count: s1Array.count + 1)
        
        for i in 0...s1Array.count {
            matrix[i][0] = i
        }
        
        for j in 0...s2Array.count {
            matrix[0][j] = j
        }
        
        for i in 1...s1Array.count {
            for j in 1...s2Array.count {
                if s1Array[i-1] == s2Array[j-1] {
                    matrix[i][j] = matrix[i-1][j-1]
                } else {
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,
                        matrix[i][j-1] + 1,
                        matrix[i-1][j-1] + 1
                    )
                }
            }
        }
        
        return matrix[s1Array.count][s2Array.count]
    }
    
    private static func calculateFuzzyScore(_ input: String, surgeon: SurgeonEntity) -> Double {
        var bestScore = 0.0
        
        let checkStrings = [surgeon.canonical] + surgeon.variations
        
        for checkString in checkStrings {
            let distance = levenshteinDistance(input.lowercased(), checkString.lowercased())
            let maxLength = max(input.count, checkString.count)
            let score = 1.0 - (Double(distance) / Double(maxLength))
            bestScore = max(bestScore, score)
        }
        
        return bestScore
    }
    
    private static func calculateFuzzyScore(_ input: String, procedure: ProcedureEntity) -> Double {
        var bestScore = 0.0
        
        let checkStrings = [procedure.canonical] + procedure.abbreviations + procedure.spokenVariations
        
        for checkString in checkStrings {
            let distance = levenshteinDistance(input.lowercased(), checkString.lowercased())
            let maxLength = max(input.count, checkString.count)
            let score = 1.0 - (Double(distance) / Double(maxLength))
            bestScore = max(bestScore, score)
        }
        
        return bestScore
    }
    
    // MARK: - Portuguese Phonetic Matching
    private static func portuguesePhonetic(_ text: String) -> String {
        var result = text.lowercased()
        
        // Common Portuguese phonetic transformations
        let transformations = [
            ("ph", "f"),
            ("ch", "x"),
            ("ç", "s"),
            ("ss", "s"),
            ("sc", "s"),
            ("z", "s"),
            ("x", "s"),
            ("w", "v"),
            ("v", "w"),
            ("y", "i"),
            ("h", ""),
            ("ck", "k"),
            ("c", "k"),
            ("q", "k"),
            ("ã", "a"),
            ("õ", "o"),
            ("á", "a"),
            ("é", "e"),
            ("í", "i"),
            ("ó", "o"),
            ("ú", "u"),
            ("â", "a"),
            ("ê", "e"),
            ("ô", "o")
        ]
        
        for (from, to) in transformations {
            result = result.replacingOccurrences(of: from, with: to)
        }
        
        // Remove duplicate consonants
        var chars = Array(result)
        var i = 1
        while i < chars.count {
            if i > 0 && chars[i] == chars[i-1] && !["a", "e", "i", "o", "u"].contains(String(chars[i])) {
                chars.remove(at: i)
            } else {
                i += 1
            }
        }
        
        return String(chars)
    }
    
    private static func calculatePhoneticScore(_ input: String, surgeon: SurgeonEntity) -> Double {
        let inputPhonetic = portuguesePhonetic(input)
        var bestScore = 0.0
        
        let checkStrings = [surgeon.canonical] + surgeon.variations + surgeon.phonetic
        
        for checkString in checkStrings {
            let checkPhonetic = portuguesePhonetic(checkString)
            if inputPhonetic == checkPhonetic {
                return 1.0
            }
            
            let distance = levenshteinDistance(inputPhonetic, checkPhonetic)
            let maxLength = max(inputPhonetic.count, checkPhonetic.count)
            let score = 1.0 - (Double(distance) / Double(maxLength))
            bestScore = max(bestScore, score)
        }
        
        return bestScore
    }
    
    // MARK: - Component Matching for Procedures
    private static func calculateComponentScore(_ input: String, procedure: ProcedureEntity) -> Double {
        let inputWords = Set(input.lowercased().split(separator: " ").map(String.init))
        let procedureComponents = Set(procedure.components.map { $0.lowercased() })
        
        if procedureComponents.isEmpty {
            return 0
        }
        
        let intersection = inputWords.intersection(procedureComponents)
        let score = Double(intersection.count) / Double(procedureComponents.count)
        
        return score
    }
    
    // MARK: - Partial Matching
    private static func containsPartialMatch(_ input: String, surgeon: SurgeonEntity) -> Bool {
        let words = input.lowercased().split(separator: " ").map(String.init)
        let surgeonWords = surgeon.canonical.lowercased().split(separator: " ").map(String.init)
        
        for word in words {
            if word.count >= 4 {
                for surgeonWord in surgeonWords {
                    if surgeonWord.contains(word) || word.contains(surgeonWord) {
                        return true
                    }
                }
            }
        }
        
        return false
    }
    
    // MARK: - Helper Methods
    private static func normalize(_ text: String) -> String {
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "  ", with: " ")
            .replacingOccurrences(of: "doutor ", with: "")
            .replacingOccurrences(of: "dr. ", with: "")
            .replacingOccurrences(of: "dr ", with: "")
    }
    
    private static func expandAbbreviations(_ text: String) -> String {
        var result = text
        
        for (abbr, expansion) in MedicalKnowledgeBase.abbreviationExpansions {
            let pattern = "\\b\(abbr)\\b"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
                let range = NSRange(location: 0, length: result.utf16.count)
                result = regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: expansion)
            }
        }
        
        return result
    }
    
    private static func capitalizeProperName(_ text: String) -> String {
        let words = text.split(separator: " ")
        let dontCapitalize = ["de", "da", "do", "das", "dos", "e", "com"]
        
        return words.enumerated().map { index, word in
            let lowercased = word.lowercased()
            if index == 0 || !dontCapitalize.contains(lowercased) {
                return word.prefix(1).uppercased() + word.dropFirst().lowercased()
            } else {
                return String(lowercased)
            }
        }.joined(separator: " ")
    }
    
    private static func findAlternativeSurgeons(for input: String, excluding: SurgeonEntity) -> [String] {
        var alternatives: [(surgeon: SurgeonEntity, score: Double)] = []
        
        for surgeon in MedicalKnowledgeBase.surgeons where surgeon.canonical != excluding.canonical {
            let fuzzyScore = calculateFuzzyScore(input, surgeon: surgeon)
            if fuzzyScore > 0.6 {
                alternatives.append((surgeon, fuzzyScore))
            }
        }
        
        return alternatives
            .sorted { $0.score > $1.score }
            .prefix(3)
            .map { $0.surgeon.canonical }
    }
    
    private static func findAlternativeProcedures(for input: String, excluding: ProcedureEntity) -> [String] {
        var alternatives: [(procedure: ProcedureEntity, score: Double)] = []
        
        for procedure in MedicalKnowledgeBase.procedures where procedure.canonical != excluding.canonical {
            let fuzzyScore = calculateFuzzyScore(input, procedure: procedure)
            if fuzzyScore > 0.6 {
                alternatives.append((procedure, fuzzyScore))
            }
        }
        
        return alternatives
            .sorted { $0.score > $1.score }
            .prefix(3)
            .map { $0.procedure.canonical }
    }
    
    private static func parseCompoundProcedure(_ text: String) -> [String] {
        var components: [String] = []
        let lowercased = text.lowercased()
        
        for marker in MedicalKnowledgeBase.compoundMarkers {
            if lowercased.contains(marker) {
                let parts = lowercased.components(separatedBy: marker)
                for part in parts {
                    let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let procedure = MedicalKnowledgeBase.findProcedure(by: trimmed) {
                        components.append(procedure.canonical)
                    }
                }
            }
        }
        
        return components
    }
}
