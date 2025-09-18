import Foundation

struct OPMERequirement {
    let needed: Bool
    let items: String
    
    static let none = OPMERequirement(needed: false, items: "")
}

class OPMEConfiguration {
    
    // MARK: - OPME Requirements Mapping
    
    // Exact procedure-to-OPME mapping based on medical requirements
    private static let requirements: [String: OPMERequirement] = [
        // Procedures requiring OPME
        "Implante de Cateter Duplo J": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J"
        ),
        
        "UTL Flexível": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J + Fibra Laser + 01 Bainha Ureteral + 01 Basket"
        ),
        
        "Ureterolitotripsia Flexível": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J + Fibra Laser + 01 Bainha Ureteral + 01 Basket"
        ),
        
        "UTL Rígida": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J + Fibra Laser"
        ),
        
        "Ureterolitotripsia Rígida": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J + Fibra Laser"
        ),
        
        "Cistolitotripsia": OPMERequirement(
            needed: true,
            items: "01 Fio Hidrofílico + 01 Cateter Duplo J + Fibra Laser"
        ),
        
        "Nefrolitotripsia Percutânea": OPMERequirement(
            needed: true,
            items: "01 Kit de Nefrostomia + 01 Cateter Duplo J + Fibra Laser"
        ),
        
        "ECIRS": OPMERequirement(
            needed: true,
            items: "01 Kit de Nefrostomia + 01 Cateter Duplo J + Fibra Laser + 01 Bainha Ureteral + 01 Basket"
        ),
        
        "Implante de Prótese Peniana": OPMERequirement(
            needed: true,
            items: "01 Prótese Peniana Inflável ou Semi-rígida"
        ),
        
        "IPP": OPMERequirement(
            needed: true,
            items: "01 Prótese Peniana Inflável ou Semi-rígida"
        ),
        
        "Sling": OPMERequirement(
            needed: true,
            items: "01 Kit de Sling Suburetral (TVT ou TOT)"
        ),
        
        "TVT": OPMERequirement(
            needed: true,
            items: "01 Kit TVT (Tension-free Vaginal Tape)"
        ),
        
        "TOT": OPMERequirement(
            needed: true,
            items: "01 Kit TOT (Transobturator Tape)"
        ),
        
        // Procedures with Duplo J removal (may need replacement)
        "Retirada de Cateter Duplo J": OPMERequirement(
            needed: false,
            items: ""
        ),
        
        // Procedures typically NOT requiring OPME
        "RTU de Bexiga": OPMERequirement.none,
        "RTU de Próstata": OPMERequirement.none,
        "RTU de Colo Vesical": OPMERequirement.none,
        "Orquiectomia Subcapsular Bilateral": OPMERequirement.none,
        "Orquiectomia Radical Unilateral": OPMERequirement.none,
        "Prostatectomia Radical": OPMERequirement.none,
        "Prostatectomia Aberta Millin": OPMERequirement.none,
        "Nefrectomia Radical": OPMERequirement.none,
        "Nefrectomia Parcial": OPMERequirement.none,
        "Nefroureterectomia Radical": OPMERequirement.none,
        "Nefrectomia Parcial VLP": OPMERequirement.none,
        "Nefrectomia Radical VLP": OPMERequirement.none,
        "Nefroureterectomia Radical VLP": OPMERequirement.none,
        "Nefrostomia Percutânea Unilateral": OPMERequirement.none,
        "Uretrotomia Interna": OPMERequirement.none,
        "UI": OPMERequirement.none,
        "Cistoscopia": OPMERequirement.none,
        "Cistectomia Radical": OPMERequirement.none,
        "Cistectomia Parcial": OPMERequirement.none,
        "Cistolitotomia": OPMERequirement.none,
        "Ureterostomia Cutanea Bilateral": OPMERequirement.none,
        "Postectomia": OPMERequirement.none,
        "Nesbit": OPMERequirement.none,
        "Herniorrafia Inguinal": OPMERequirement.none,
        "Vasectomia": OPMERequirement.none,
        "Varicocelectomia": OPMERequirement.none,
        "Meatoplastia": OPMERequirement.none
    ]
    
    // Alternative names and abbreviations mapping
    private static let alternativeNames: [String: String] = [
        "OSC": "Orquiectomia Subcapsular Bilateral",
        "RTUP": "RTU de Próstata",
        "PR": "Prostatectomia Radical",
        "NLP": "Nefrolitotripsia Percutânea",
        "PCNL": "Nefrolitotripsia Percutânea",
        "DJ": "Implante de Cateter Duplo J",
        "Duplo J": "Implante de Cateter Duplo J"
    ]
    
    // MARK: - Main Configuration Method
    
    static func getConfiguration(for procedure: String) -> OPMERequirement {
        let normalizedProcedure = normalizeProcedureName(procedure)
        
        // Try exact match first
        if let requirement = requirements[normalizedProcedure] {
            return requirement
        }
        
        // Try alternative names
        if let canonicalName = alternativeNames[normalizedProcedure],
           let requirement = requirements[canonicalName] {
            return requirement
        }
        
        // Try fuzzy matching with known procedures
        for (knownProcedure, requirement) in requirements {
            if fuzzyMatch(normalizedProcedure, with: knownProcedure) {
                return requirement
            }
        }
        
        // Check for compound procedures (e.g., "RTU de Próstata + OSC")
        if isCompoundProcedure(procedure) {
            return getCompoundProcedureConfiguration(procedure)
        }
        
        // Default: no OPME required
        return OPMERequirement.none
    }
    
    // MARK: - Helper Methods
    
    private static func normalizeProcedureName(_ procedure: String) -> String {
        var normalized = procedure.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Expand common abbreviations
        normalized = normalized
            .replacingOccurrences(of: "VLP", with: "Videolaparoscópica")
            .replacingOccurrences(of: "UTL", with: "Ureterolitotripsia")
        
        // Find the best match from MedicalKnowledgeBase
        let matchResult = IntelligentMatcher.matchProcedure(normalized)
        if matchResult.isKnownEntity && matchResult.confidence > 0.8 {
            return matchResult.value
        }
        
        return normalized
    }
    
    private static func fuzzyMatch(_ input: String, with target: String) -> Bool {
        let inputLower = input.lowercased()
        let targetLower = target.lowercased()
        
        // Check if input contains significant parts of target
        let targetWords = targetLower.split(separator: " ").map(String.init)
        let significantWords = targetWords.filter { $0.count > 3 } // Skip small words
        
        var matchCount = 0
        for word in significantWords {
            if inputLower.contains(word) {
                matchCount += 1
            }
        }
        
        // Consider it a match if we match at least 70% of significant words
        let matchRatio = Double(matchCount) / Double(max(significantWords.count, 1))
        return matchRatio >= 0.7
    }
    
    private static func isCompoundProcedure(_ procedure: String) -> Bool {
        let compoundIndicators = ["+", "mais", "com", "e", "associado"]
        let lowercased = procedure.lowercased()
        
        for indicator in compoundIndicators {
            if lowercased.contains(indicator) {
                return true
            }
        }
        
        return false
    }
    
    private static func getCompoundProcedureConfiguration(_ procedure: String) -> OPMERequirement {
        // Parse compound procedure and check if any component requires OPME
        let components = parseCompoundProcedure(procedure)
        var allItems: [String] = []
        var needsOPME = false
        
        for component in components {
            let config = getConfiguration(for: component)
            if config.needed {
                needsOPME = true
                if !config.items.isEmpty && !allItems.contains(config.items) {
                    allItems.append(config.items)
                }
            }
        }
        
        if needsOPME {
            // Combine unique items
            let combinedItems = combineOPMEItems(allItems)
            return OPMERequirement(needed: true, items: combinedItems)
        }
        
        return OPMERequirement.none
    }
    
    private static func parseCompoundProcedure(_ procedure: String) -> [String] {
        var components: [String] = []
        let separators = ["+", " mais ", " com ", " e ", " associado "]
        
        let remaining = procedure
        for separator in separators {
            let parts = remaining.components(separatedBy: separator)
            if parts.count > 1 {
                components.append(contentsOf: parts.map { $0.trimmingCharacters(in: .whitespaces) })
                break
            }
        }
        
        if components.isEmpty {
            components = [procedure]
        }
        
        return components
    }
    
    private static func combineOPMEItems(_ items: [String]) -> String {
        // Parse and combine OPME items, removing duplicates
        var uniqueItems = Set<String>()
        
        for itemList in items {
            let individualItems = itemList.components(separatedBy: "+").map { $0.trimmingCharacters(in: .whitespaces) }
            uniqueItems.formUnion(individualItems)
        }
        
        // Sort for consistent output
        let sortedItems = uniqueItems.sorted()
        return sortedItems.joined(separator: " + ")
    }
    
    // MARK: - Validation
    
    static func requiresOPME(for procedure: String) -> Bool {
        return getConfiguration(for: procedure).needed
    }
    
    static func getOPMEDescription(for procedure: String) -> String? {
        let config = getConfiguration(for: procedure)
        return config.needed ? config.items : nil
    }
    
    // MARK: - Statistics
    
    static func getProceduresRequiringOPME() -> [String] {
        return requirements.compactMap { (procedure, requirement) in
            requirement.needed ? procedure : nil
        }.sorted()
    }
}
