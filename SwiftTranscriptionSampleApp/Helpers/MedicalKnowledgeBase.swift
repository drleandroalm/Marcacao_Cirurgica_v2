import Foundation

struct SurgeonEntity: Sendable {
    let canonical: String
    let variations: [String]
    let phonetic: [String]
    let specialties: [String]
    
    init(canonical: String, variations: [String] = [], phonetic: [String] = [], specialties: [String] = []) {
        self.canonical = canonical
        self.variations = variations
        self.phonetic = phonetic
        self.specialties = specialties
    }
}

extension SurgeonEntity: Decodable {
    private enum CodingKeys: String, CodingKey {
        case canonical
        case variations
        case phonetic
        case specialties
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let canonical = try container.decode(String.self, forKey: .canonical)
        let variations = try container.decodeIfPresent([String].self, forKey: .variations) ?? []
        let phonetic = try container.decodeIfPresent([String].self, forKey: .phonetic) ?? []
        let specialties = try container.decodeIfPresent([String].self, forKey: .specialties) ?? []
        self.init(canonical: canonical, variations: variations, phonetic: phonetic, specialties: specialties)
    }
}

struct ProcedureEntity: Sendable {
    let canonical: String
    let abbreviations: [String]
    let spokenVariations: [String]
    let components: [String]
    let typicalDuration: ClosedRange<Int> // in minutes

    // Phase 1 enhancements
    let snomedCT: String?              // SNOMED-CT code
    let icd10pcs: [String]?            // ICD-10-PCS codes
    let specialty: String?              // "Urologia", "Cirurgia Geral", etc.
    let requiredOPME: [String]?        // Required special equipment
    let typicalAnesthesia: String?      // "Raquianestesia", "Anestesia Geral", etc.

    init(
        canonical: String,
        abbreviations: [String] = [],
        spokenVariations: [String] = [],
        components: [String] = [],
        typicalDuration: ClosedRange<Int> = 30...240,
        snomedCT: String? = nil,
        icd10pcs: [String]? = nil,
        specialty: String? = nil,
        requiredOPME: [String]? = nil,
        typicalAnesthesia: String? = nil
    ) {
        self.canonical = canonical
        self.abbreviations = abbreviations
        self.spokenVariations = spokenVariations
        self.components = components
        self.typicalDuration = typicalDuration
        self.snomedCT = snomedCT
        self.icd10pcs = icd10pcs
        self.specialty = specialty
        self.requiredOPME = requiredOPME
        self.typicalAnesthesia = typicalAnesthesia
    }
}

extension ProcedureEntity: Decodable {
    private enum CodingKeys: String, CodingKey {
        case canonical
        case abbreviations
        case spokenVariations
        case components
        case typicalDurationMinutes
        case snomedCT
        case icd10pcs
        case specialty
        case requiredOPME
        case typicalAnesthesia
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let canonical = try container.decode(String.self, forKey: .canonical)
        let abbreviations = try container.decodeIfPresent([String].self, forKey: .abbreviations) ?? []
        let spoken = try container.decodeIfPresent([String].self, forKey: .spokenVariations) ?? []
        let components = try container.decodeIfPresent([String].self, forKey: .components) ?? []
        let durationArray = try container.decodeIfPresent([Int].self, forKey: .typicalDurationMinutes) ?? [30, 240]
        let minDuration = durationArray.first ?? 30
        let maxDuration = durationArray.count > 1 ? durationArray[1] : minDuration

        // Phase 1 enhancements
        let snomedCT = try container.decodeIfPresent(String.self, forKey: .snomedCT)
        let icd10pcs = try container.decodeIfPresent([String].self, forKey: .icd10pcs)
        let specialty = try container.decodeIfPresent(String.self, forKey: .specialty)
        let requiredOPME = try container.decodeIfPresent([String].self, forKey: .requiredOPME)
        let typicalAnesthesia = try container.decodeIfPresent(String.self, forKey: .typicalAnesthesia)

        self.init(
            canonical: canonical,
            abbreviations: abbreviations,
            spokenVariations: spoken,
            components: components,
            typicalDuration: minDuration...maxDuration,
            snomedCT: snomedCT,
            icd10pcs: icd10pcs,
            specialty: specialty,
            requiredOPME: requiredOPME,
            typicalAnesthesia: typicalAnesthesia
        )
    }
}

/// Loads curated knowledge base datasets from `Resources/KnowledgeBase/*.json`.
final class MedicalKnowledgeBase {
    private enum Resource: String {
        case surgeons
        case procedures
        case abbreviations
    }
    
    private final class BundleToken {}
    
    // MARK: - Loaded datasets
    static let surgeons: [SurgeonEntity] = loadSurgeons()
    static let procedures: [ProcedureEntity] = loadProcedures()
    static let abbreviationExpansions: [String: String] = loadAbbreviations()
    static let compoundMarkers = ["+", "mais", "com", "e", "associado", "combinado"]
    
    // MARK: - Helper Methods
    static func getAllSurgeonNames() -> Set<String> {
        var names = Set<String>()
        for surgeon in surgeons {
            names.insert(surgeon.canonical.lowercased())
            names.formUnion(surgeon.variations.map { $0.lowercased() })
            names.formUnion(surgeon.phonetic.map { $0.lowercased() })
        }
        return names
    }
    
    static func getAllProcedureNames() -> Set<String> {
        var names = Set<String>()
        for procedure in procedures {
            names.insert(procedure.canonical.lowercased())
            names.formUnion(procedure.abbreviations.map { $0.lowercased() })
            names.formUnion(procedure.spokenVariations.map { $0.lowercased() })
        }
        return names
    }
    
    static func findSurgeon(by name: String) -> SurgeonEntity? {
        let lowercased = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        for surgeon in surgeons {
            if surgeon.canonical.lowercased() == lowercased {
                return surgeon
            }
            if surgeon.variations.contains(where: { $0.lowercased() == lowercased }) {
                return surgeon
            }
            if surgeon.phonetic.contains(where: { $0.lowercased() == lowercased }) {
                return surgeon
            }
        }
        return nil
    }
    
    static func findProcedure(by name: String) -> ProcedureEntity? {
        let lowercased = name.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        for procedure in procedures {
            if procedure.canonical.lowercased() == lowercased {
                return procedure
            }
            if procedure.abbreviations.contains(where: { $0.lowercased() == lowercased }) {
                return procedure
            }
            if procedure.spokenVariations.contains(where: { $0.lowercased() == lowercased }) {
                return procedure
            }
        }
        return nil
    }
    
    static func expandAbbreviation(_ text: String) -> String {
        let uppercased = text.uppercased()
        return abbreviationExpansions[uppercased] ?? text
    }
    
    static func isCompoundProcedure(_ text: String) -> Bool {
        let lowercased = text.lowercased()
        return compoundMarkers.contains { marker in
            lowercased.contains(marker)
        }
    }
}

private extension MedicalKnowledgeBase {
    static func loadSurgeons() -> [SurgeonEntity] {
        load([SurgeonEntity].self, resource: .surgeons, fallback: [])
    }
    
    static func loadProcedures() -> [ProcedureEntity] {
        load([ProcedureEntity].self, resource: .procedures, fallback: [])
    }
    
    static func loadAbbreviations() -> [String: String] {
        load([String: String].self, resource: .abbreviations, fallback: [:])
    }
    
    private static func load<T: Decodable>(_ type: T.Type, resource: Resource, fallback: T) -> T {
        guard let url = resourceURL(for: resource) else {
            print("⚠️ MedicalKnowledgeBase: missing resource \(resource.rawValue).json")
            return fallback
        }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            return try decoder.decode(T.self, from: data)
        } catch {
            print("⚠️ MedicalKnowledgeBase: failed decoding \(resource.rawValue).json: \(error)")
            return fallback
        }
    }
    
    private static func resourceURL(for resource: Resource) -> URL? {
        let primaryBundle: Bundle
        #if SWIFT_PACKAGE
        primaryBundle = .module
        #else
        primaryBundle = Bundle(for: BundleToken.self)
        #endif
        let searchPaths: [String?] = ["Resources/KnowledgeBase", "KnowledgeBase", nil]
        let candidateBundles: [Bundle] = primaryBundle == Bundle.main ? [primaryBundle] : [primaryBundle, Bundle.main]
        for bundle in candidateBundles {
            for subdir in searchPaths {
                if let url = bundle.url(forResource: resource.rawValue, withExtension: "json", subdirectory: subdir) {
                    return url
                }
            }
        }
        return nil
    }
}
