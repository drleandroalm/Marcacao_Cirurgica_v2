import Foundation
import FoundationModels
import os

// Helper function for timeout
private struct _Box<T>: @unchecked Sendable { let value: T }

func withTimeout<T>(seconds: TimeInterval, operation: @Sendable @escaping () async throws -> T) async throws -> T {
    try await withThrowingTaskGroup(of: _Box<T>.self) { group in
        group.addTask {
            let value = try await operation()
            return _Box(value: value)
        }
        
        group.addTask {
            try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            throw EntityExtractionError.timeout
        }
        
        let result = try await group.next()!
        group.cancelAll()
        return result.value
    }
}

struct ExtractedEntity: Sendable {
    let fieldId: String
    let value: String
    let confidence: Double
    let alternatives: [String]
    let originalText: String
}

struct ExtractionResult: Sendable {
    var entities: [ExtractedEntity]
    let unprocessedText: String
    let confidence: Double
}

@Observable
@MainActor
/// Extracts structured entities from transcripts using on-device language models and deterministic fallbacks.
class EntityExtractor {
    private var configuration: ExtractionConfiguration
    private let model: SystemLanguageModel
    private var session: LanguageModelSession?
    private var isSessionReady = false
    private static let metricsLog = OSLog(subsystem: "SwiftTranscriptionSampleApp", category: "EntityExtractor")
    
    private static func redactedSummary(for text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let wordCount = trimmed.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        return "<redacted len=\(trimmed.count) words=\(wordCount)>"
    }
    
    private static func redactedValue(_ value: String) -> String {
        "<len=\(value.count)>"
    }
    
    static let shared = EntityExtractor()
    
    init(configuration: ExtractionConfiguration = ExtractionConfiguration(), model: SystemLanguageModel = .default) {
        self.configuration = configuration
        self.model = model
        Task {
            await setupSession()
        }
    }
    
    private func setupSession() async {
        guard model.isAvailable else { 
            print("⚠️ Foundation Models not available")
            return 
        }
        
        session = LanguageModelSession(
            model: model,
            instructions: """
            You are a medical form extraction specialist.
            Extract entities from Portuguese medical transcriptions.
            Always respond with valid JSON only, no explanations.
            Match entities to known values when possible.
            """
        )
        isSessionReady = true
        print("✅ EntityExtractor session ready")
    }
    
    var isAvailable: Bool {
        return model.isAvailable
    }
    
    func updateConfiguration(_ configuration: ExtractionConfiguration) {
        self.configuration = configuration
    }

    // MARK: - Abbreviation expansion utility (shared with fallback)
    nonisolated private static func expandAbbreviations(in text: String) -> String {
        var processed = text
        for (abbr, expansion) in MedicalKnowledgeBase.abbreviationExpansions {
            let pattern = "\\b\(abbr)\\b"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
                let range = NSRange(location: 0, length: processed.utf16.count)
                processed = regex.stringByReplacingMatches(
                    in: processed,
                    options: [],
                    range: range,
                    withTemplate: expansion
                )
            }
        }
        return processed
    }

    func extractEntities(from transcription: String, for form: SurgicalRequestForm) async throws -> ExtractionResult {
        let signpostID = OSSignpostID(log: Self.metricsLog)
        os_signpost(.begin, log: Self.metricsLog, name: "extractEntities()", signpostID: signpostID)
        defer { os_signpost(.end, log: Self.metricsLog, name: "extractEntities()", signpostID: signpostID) }

        print("🔍 EntityExtractor: Starting extraction \(Self.redactedSummary(for: transcription))")
        
        guard isAvailable else {
            print("❌ EntityExtractor: Model unavailable")
            throw EntityExtractionError.modelUnavailable
        }
        
        // Skip extraction for very short texts (less than ~5 words / ~20 chars)
        let trimmedText = transcription.trimmingCharacters(in: .whitespacesAndNewlines)
        let wordCount = trimmedText.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        guard trimmedText.count > 20 || wordCount >= 5 else {
            print("⚠️ EntityExtractor: Text too short (\(trimmedText.count) chars, \(wordCount) words) - skipping extraction")
            return ExtractionResult(entities: [], unprocessedText: transcription, confidence: 0)
        }
        
        // Preprocess transcription to expand abbreviations
        let preprocessID = OSSignpostID(log: Self.metricsLog)
        os_signpost(.begin, log: Self.metricsLog, name: "preprocessTranscription", signpostID: preprocessID)
        let preprocessed = preprocessTranscription(transcription)
        os_signpost(.end, log: Self.metricsLog, name: "preprocessTranscription", signpostID: preprocessID)
        
        // Ensure session is ready
        if !isSessionReady || session == nil {
            print("🔄 Setting up EntityExtractor session...")
            await setupSession()
        }
        
        guard let session = session else {
            throw EntityExtractionError.sessionUnavailable
        }
        
        let promptBuildID = OSSignpostID(log: Self.metricsLog)
        os_signpost(.begin, log: Self.metricsLog, name: "buildExtractionPrompt", signpostID: promptBuildID)
        let prompt = buildExtractionPrompt(for: preprocessed, fields: form.fields)
        os_signpost(.end, log: Self.metricsLog, name: "buildExtractionPrompt", signpostID: promptBuildID)

        do {
            print("🤖 EntityExtractor: Sending prompt to FoundationModels...")
            print("📝 Prompt length: \(prompt.count) characters")
            
            // Add timeout for the model response
            let llmID = OSSignpostID(log: Self.metricsLog)
            os_signpost(.begin, log: Self.metricsLog, name: "LLMRespond", signpostID: llmID)
            let response = try await withTimeout(seconds: configuration.responseTimeout) {
                try await session.respond(to: prompt)
            }
            os_signpost(.end, log: Self.metricsLog, name: "LLMRespond", signpostID: llmID)
            
            print("✅ EntityExtractor: Got response from model (len=\(response.content.count) chars)")
            
            let parseID = OSSignpostID(log: Self.metricsLog)
            os_signpost(.begin, log: Self.metricsLog, name: "parseExtractionResponse", signpostID: parseID)
            let parsedResult = try parseExtractionResponse(response.content, originalText: transcription)
            os_signpost(.end, log: Self.metricsLog, name: "parseExtractionResponse", signpostID: parseID)
            print("📊 EntityExtractor: Parsed \(parsedResult.entities.count) entities")
            
            // Enhance with knowledge base validation off the main actor
            print("🔧 EntityExtractor: Enhancing with knowledge base...")
            let kbID = OSSignpostID(log: Self.metricsLog)
            os_signpost(.begin, log: Self.metricsLog, name: "enhanceWithKnowledgeBase", signpostID: kbID)
            let enhancedResult = await Task.detached(priority: .userInitiated) {
                Self.enhanceWithKnowledgeBase(parsedResult)
            }.value
            os_signpost(.end, log: Self.metricsLog, name: "enhanceWithKnowledgeBase", signpostID: kbID)
            var result = enhancedResult

            // If coverage is low, supplement with deterministic fallback
            let requiredIds = configuration.requiredFieldIds
            let presentIds = Set(result.entities.map { $0.fieldId })
            if !requiredIds.isSubset(of: presentIds) {
                let supplementID = OSSignpostID(log: Self.metricsLog)
                os_signpost(.begin, log: Self.metricsLog, name: "supplementWithFallback", signpostID: supplementID)
                print("🧩 Supplementing entities with fallback extraction for missing fields…")
                let fallback = try await performFallbackExtraction(for: transcription)
                var merged: [String: ExtractedEntity] = [:]
                // Prefer model entities
                for e in result.entities { merged[e.fieldId] = e }
                // Fill missing from fallback
                for e in fallback.entities where merged[e.fieldId] == nil {
                    merged[e.fieldId] = e
                }
                let mergedEntities = Array(merged.values)
                var supplemented = ExtractionResult(
                    entities: mergedEntities,
                    unprocessedText: result.unprocessedText.isEmpty ? fallback.unprocessedText : result.unprocessedText,
                    confidence: max(result.confidence, fallback.confidence)
                )
                // Re-run KB enhancement for any added items
                supplemented = Self.enhanceWithKnowledgeBase(supplemented)
                result = supplemented
                os_signpost(.end, log: Self.metricsLog, name: "supplementWithFallback", signpostID: supplementID)
            }
            
            print("🏁 EntityExtractor: Final result - \(result.entities.count) entities, confidence: \(result.confidence)")
            for entity in result.entities {
                if let field = form.fields.first(where: { $0.id == entity.fieldId }) {
                    let confidencePercent = Int(entity.confidence * 100)
                    print("  - \(field.label): \(Self.redactedValue(entity.value)) (confidence: \(confidencePercent)%)")
                }
            }
            
            return result
        } catch {
            print("❌ EntityExtractor: Extraction failed: \(error)")
            print("🔄 Attempting fallback extraction...")
            
            // Try fallback extraction
            return try await performFallbackExtraction(for: transcription)
        }
    }
    
    private func preprocessTranscription(_ text: String) -> String {
        var processed = text
        
        // Expand known abbreviations
        for (abbr, expansion) in MedicalKnowledgeBase.abbreviationExpansions {
            let pattern = "\\b\(abbr)\\b"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) {
                let range = NSRange(location: 0, length: processed.utf16.count)
                processed = regex.stringByReplacingMatches(
                    in: processed,
                    options: [],
                    range: range,
                    withTemplate: expansion
                )
            }
        }
        
        return processed
    }
    
    private func performFallbackExtraction(for text: String) async throws -> ExtractionResult {
        // Run on the main actor to respect actor isolation for helper methods
        let id = OSSignpostID(log: Self.metricsLog)
        os_signpost(.begin, log: Self.metricsLog, name: "performFallbackExtraction", signpostID: id)
        let r = try Self.fallbackExtraction(from: text)
        os_signpost(.end, log: Self.metricsLog, name: "performFallbackExtraction", signpostID: id)
        return r
    }
    
    nonisolated private static func enhanceWithKnowledgeBase(_ result: ExtractionResult) -> ExtractionResult {
        print("🔍 EntityExtractor: Enhancing entities with knowledge base...")
        var enhancedEntities = result.entities
        
        for (index, entity) in enhancedEntities.enumerated() {
            switch entity.fieldId {
            case "surgeonName":
                let matchResult = IntelligentMatcher.matchSurgeon(entity.value)
                if matchResult.isKnownEntity && matchResult.confidence > entity.confidence {
                    enhancedEntities[index] = ExtractedEntity(
                        fieldId: entity.fieldId,
                        value: matchResult.value,
                        confidence: matchResult.confidence,
                        alternatives: matchResult.alternatives,
                        originalText: entity.originalText
                    )
                }
                
            case "procedureName":
                let matchResult = IntelligentMatcher.matchProcedure(entity.value)
                if matchResult.isKnownEntity && matchResult.confidence > entity.confidence {
                    enhancedEntities[index] = ExtractedEntity(
                        fieldId: entity.fieldId,
                        value: matchResult.value,
                        confidence: matchResult.confidence,
                        alternatives: matchResult.alternatives,
                        originalText: entity.originalText
                    )
                }
                
            default:
                break
            }
        }
        
        // Recalculate overall confidence
        let confidenceSum = enhancedEntities.reduce(0.0) { $0 + $1.confidence }
        let avgConfidence = enhancedEntities.isEmpty ? 0 : confidenceSum / Double(enhancedEntities.count)
        
        return ExtractionResult(
            entities: enhancedEntities,
            unprocessedText: result.unprocessedText,
            confidence: avgConfidence
        )
    }
    
    private func buildExtractionPrompt(for text: String, fields: [TemplateField]) -> String {
        let fieldsDescription = fields.map { field in
            let typeDescription = getFieldTypeDescription(field.fieldType)
            return "- \(field.id): \(field.label) (\(typeDescription))"
        }.joined(separator: "\n")
        
        // Include known entities for better recognition
        let knownSurgeons = MedicalKnowledgeBase.surgeons.map { $0.canonical }.joined(separator: ", ")
        let commonProcedures = MedicalKnowledgeBase.procedures.prefix(10).map { $0.canonical }.joined(separator: ", ")
        
        return """
        Você é um assistente especializado em extrair informações médicas de transcrições em português brasileiro.

        CAMPOS DO FORMULÁRIO:
        \(fieldsDescription)

        CONHECIMENTO PRÉVIO DOS CIRURGIÕES COMUNS:
        \(knownSurgeons)

        PROCEDIMENTOS MÉDICOS CONHECIDOS (exemplos):
        \(commonProcedures)

        ABREVIAÇÕES MÉDICAS:
        - OSC = Orquiectomia Subcapsular Bilateral
        - VLP = Videolaparoscópica
        - IPP = Implante de Prótese Peniana
        - UI = Uretrotomia Interna
        - RTU = Ressecção Transuretral

        REGRAS DE EXTRAÇÃO:
        1. Priorize correspondências com entidades conhecidas
        2. Use correspondência fonética para nomes similares
        3. Expanda abreviações automaticamente
        4. Extraia informações mesmo quando ditas fora de ordem
        5. Todos os nomes próprios devem começar com letra maiúscula
        6. Datas no formato DD/MM/AAAA
        7. Horários no formato HH:MM
        8. Telefones apenas números (10-11 dígitos)
        9. Idades apenas números
        10. Se não encontrar uma informação, retorne "VAZIO"
        11. Indique confiança: ALTA (90-100%), MEDIA (70-89%), BAIXA (0-69%)
        12. Mantenha confiança ALTA (95%+) para correspondências exatas com entidades conhecidas

        TEXTO DA TRANSCRIÇÃO:
        "\(text)"

        RESPONDA EXATAMENTE NESTE FORMATO JSON:
        {
            "entities": [
                {
                    "fieldId": "patientName",
                    "value": "João Silva",
                    "confidence": 95,
                    "alternatives": ["João da Silva", "João Santos"]
                }
            ],
            "unprocessed": "texto não processado",
            "overallConfidence": 87
        }
        
        Não adicione texto explicativo, apenas o JSON válido.
        """
    }
    
    private func getFieldTypeDescription(_ fieldType: FieldType) -> String {
        switch fieldType {
        case .text:
            return "Nome próprio ou texto livre"
        case .age:
            return "Idade em anos (ex: 45)"
        case .number:
            return "Número (ex: idade)"
        case .date:
            return "Data no formato DD/MM/AAAA"
        case .time:
            return "Horário no formato HH:MM"
        case .duration:
            return "Duração em HH:MM (ex: 01:30)"
        case .phone:
            return "Telefone (10-11 dígitos)"
        }
    }
    
    private func parseExtractionResponse(_ response: String, originalText: String) throws -> ExtractionResult {
        print("🔍 Parsing response of \(response.count) characters")
        
        // Try to robustly extract JSON from the response (in case there's extra text)
        if let firstBrace = response.firstIndex(of: "{"),
           let lastBrace = response.lastIndex(of: "}") {
            let jsonString = String(response[firstBrace...lastBrace])
            if let jsonData = jsonString.data(using: .utf8) {
                do {
                    return try parseJSONResponse(jsonData, originalText: originalText)
                } catch {
                    print("⚠️ JSON slicing parse failed, falling back to direct parse/error: \(error)")
                }
            }
        }
        
        // Fallback: try direct parse of the whole string
        guard let jsonData = response.data(using: .utf8) else {
            print("⚠️ Could not convert response to data, trying fallback extraction")
            return try Self.fallbackExtraction(from: originalText)
        }
        
        do {
            return try parseJSONResponse(jsonData, originalText: originalText)
        } catch {
            print("⚠️ Direct JSON parse failed: \(error). Falling back.")
            return try Self.fallbackExtraction(from: originalText)
        }
    }
    
    private func parseJSONResponse(_ jsonData: Data, originalText: String) throws -> ExtractionResult {
        do {
            let raw = try JSONSerialization.jsonObject(with: jsonData)
            guard let json = raw as? [String: Any] else {
                throw EntityExtractionError.invalidResponse
            }
            print("📦 Successfully parsed JSON")
            
            guard let entitiesArray = json["entities"] as? [[String: Any]] else {
                throw EntityExtractionError.invalidResponse
            }
            
            // Accept overallConfidence as Double or Int
            var overallConfidence: Double = 0
            if let ocDouble = json["overallConfidence"] as? Double {
                overallConfidence = ocDouble
            } else if let ocInt = json["overallConfidence"] as? Int {
                overallConfidence = Double(ocInt)
            } else {
                // If not provided, estimate later from entities
                overallConfidence = 0
            }
            
            let entities = entitiesArray.compactMap { entityDict -> ExtractedEntity? in
                guard let fieldId = entityDict["fieldId"] as? String,
                      let value = entityDict["value"] as? String else {
                    return nil
                }
                
                // Accept confidence as Double or Int
                var confidencePct: Double = 0
                if let cDouble = entityDict["confidence"] as? Double {
                    confidencePct = cDouble
                } else if let cInt = entityDict["confidence"] as? Int {
                    confidencePct = Double(cInt)
                } else {
                    // Default if missing
                    confidencePct = 70
                }
                
                // Alternatives may be missing or empty
                let alternatives = entityDict["alternatives"] as? [String] ?? []
                
                // Skip empty values
                guard value != "VAZIO" && !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    return nil
                }
                // Apply capitalization only to proper-name like fields
                let normalizedValue: String
                switch fieldId {
                case "patientName", "surgeonName", "procedureName":
                    normalizedValue = Self.capitalizeProperly(value)
                default:
                    normalizedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
                }

                return ExtractedEntity(
                    fieldId: fieldId,
                    value: normalizedValue,
                    confidence: max(0.0, min(1.0, confidencePct / 100.0)), // 0.0 ... 1.0
                    alternatives: alternatives.map { alt in
                        switch fieldId {
                        case "patientName", "surgeonName", "procedureName": return Self.capitalizeProperly(alt)
                        default: return alt
                        }
                    },
                    originalText: originalText
                )
            }
            
            // Unprocessed text may come under different keys
            let unprocessed = (json["unprocessed"] as? String)
                ?? (json["unprocessedText"] as? String)
                ?? ""
            
            // If overallConfidence wasn't provided, estimate from entities
            let finalOverall = overallConfidence > 0
                ? max(0.0, min(1.0, overallConfidence / 100.0))
                : (entities.isEmpty ? 0 : entities.map { $0.confidence }.reduce(0, +) / Double(entities.count))
            
            return ExtractionResult(
                entities: entities,
                unprocessedText: unprocessed,
                confidence: finalOverall
            )
            
        } catch {
            // Fallback: try to extract using basic pattern matching
            return try Self.fallbackExtraction(from: originalText)
        }
    }
    
    private static func capitalizeProperly(_ text: String) -> String {
        return text.split(separator: " ")
            .map { word in
                let lowercased = word.lowercased()
                
                // Don't capitalize articles, prepositions, and conjunctions in Portuguese
                let dontCapitalize = ["de", "da", "do", "das", "dos", "e", "em", "na", "no", "nas", "nos", "com", "para", "por", "a", "o", "as", "os"]
                
                if dontCapitalize.contains(lowercased) {
                    return String(lowercased)
                } else {
                    return String(word.prefix(1).uppercased() + word.dropFirst().lowercased())
                }
            }
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    static func fallbackExtraction(from text: String) throws -> ExtractionResult {
        print("🔍 Fallback extraction: Processing \(Self.redactedSummary(for: text))")
        let id = OSSignpostID(log: metricsLog)
        os_signpost(.begin, log: metricsLog, name: "fallbackExtraction", signpostID: id)
        
        var entities: [ExtractedEntity] = []
        // Expand known abbreviations first (e.g., RTU, RTUP, UTL, etc.)
        let expanded = expandAbbreviations(in: text)
        let lowercased = expanded.lowercased()
        let words = lowercased.split(separator: " ").map(String.init)
        print("📊 Starting pattern matching for all 8 fields...")
        
        // 1. Extract patient name - enhanced patterns
        let namePatterns = [
            ("paciente", 1, 3),  // "paciente João Silva"
            ("nome", 1, 3),      // "nome João Silva"
            ("senhor", 1, 3),    // "senhor João Silva"
            ("senhora", 1, 3),   // "senhora Maria Silva"
        ]
        
        for (keyword, startOffset, endOffset) in namePatterns {
            if let keyIndex = words.firstIndex(where: { $0.lowercased().contains(keyword) }),
               keyIndex + startOffset < words.count {
                let endIndex = min(keyIndex + endOffset, words.count - 1)
                let name = words[keyIndex + startOffset...endIndex]
                    .filter { !["de", "da", "do", "dos", "das"].contains($0.lowercased()) || $0.count > 2 }
                    .joined(separator: " ")
                
                if !name.isEmpty && name.count > 2 {
                    entities.append(ExtractedEntity(
                        fieldId: "patientName",
                        value: Self.capitalizeProperly(name),
                        confidence: 0.75,
                        alternatives: [],
                        originalText: text
                    ))
                    print("👤 Found patient name \(Self.redactedValue(name))")
                    break
                }
            }
        }
        
        // 2. Extract age - enhanced patterns
        let agePatterns = [
            #"(\d{1,3})\s*anos?"#,
            #"idade\s+(?:de\s+)?(\d{1,3})"#,
            #"(\d{1,3})\s*(?:anos?\s+)?(?:de\s+)?idade"#
        ]
        
        for pattern in agePatterns {
            if let ageMatch = text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
                let ageStr = String(text[ageMatch])
                if let ageNumber = ageStr.components(separatedBy: CharacterSet.decimalDigits.inverted)
                    .compactMap({ Int($0) })
                    .first,
                   ageNumber > 0 && ageNumber < 150 {
                    entities.append(ExtractedEntity(
                        fieldId: "patientAge",
                        value: String(ageNumber),
                        confidence: 0.85,
                        alternatives: [],
                        originalText: text
                    ))
                    print("🎂 Found age \(Self.redactedValue(String(ageNumber)))")
                    break
                }
            }
        }
        
        // 3. Extract phone - enhanced patterns (allow any non-digit separators like ')' or '.')
        let phonePatterns = [
            #"(?:telefone|celular|contato)\s*(?:é\s*)?(?:o\s*)?(\d{2}\D*\d{4,5}\D*\d{4})"#,
            #"(\d{2}\D*\d{4,5}\D*\d{4})"#,
            #"(\d{10,11})"#
        ]
        
        for pattern in phonePatterns {
            if let phoneMatch = expanded.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
                let phoneStr = String(expanded[phoneMatch])
                let phone = phoneStr.components(separatedBy: CharacterSet.decimalDigits.inverted).joined()
                if (8...11).contains(phone.count) {
                    let conf: Double = (phone.count >= 10) ? 0.8 : 0.65
                    entities.append(ExtractedEntity(
                        fieldId: "patientPhone",
                        value: phone,
                        confidence: conf,
                        alternatives: [],
                        originalText: text
                    ))
                    print("📞 Found phone \(Self.redactedValue(phone))")
                    break
                }
            }
        }
        
        // 4. Extract date - enhanced patterns
        let dateKeywords = [
            "amanhã": 1,
            "depois de amanhã": 2,
            "hoje": 0
        ]
        
        var dateFound = false
        for (keyword, daysToAdd) in dateKeywords {
            if lowercased.contains(keyword) {
                let targetDate = Calendar.current.date(byAdding: .day, value: daysToAdd, to: Date())!
                
                let formatter = DateFormatter()
                formatter.dateFormat = "dd/MM/yyyy"
                entities.append(ExtractedEntity(
                    fieldId: "surgeryDate",
                    value: formatter.string(from: targetDate),
                    confidence: 0.9,
                    alternatives: [],
                    originalText: text
                ))
                print("📅 Found date via keyword \(keyword)")
                dateFound = true
                break
            }
        }
        
        if !dateFound {
            // Weekday mapping (e.g., próxima segunda)
            if let weekdayDate = TranscriptionProcessor.computeWeekdayDate(from: lowercased) {
                let formatter = DateFormatter(); formatter.dateFormat = "dd/MM/yyyy"
                entities.append(ExtractedEntity(
                    fieldId: "surgeryDate",
                    value: formatter.string(from: weekdayDate),
                    confidence: 0.8,
                    alternatives: [],
                    originalText: text
                ))
                dateFound = true
            }
        }
        
        if !dateFound {
            let datePatterns = [
                #"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})"#,
                #"dia\s+(\d{1,2})\s+de\s+(\w+)"#,
                #"(\d{1,2})\s+de\s+(\w+)"#
            ]
            
            for pattern in datePatterns {
                if let dateMatch = text.range(of: pattern, options: .regularExpression) {
                    let date = String(text[dateMatch])
                    entities.append(ExtractedEntity(
                        fieldId: "surgeryDate",
                        value: formatDate(date),
                        confidence: 0.8,
                        alternatives: [],
                        originalText: text
                    ))
                    print("📅 Found date \(Self.redactedValue(date))")
                    break
                }
            }
        }
        
        // 5. Extract time - enhanced patterns
        let timePatterns = [
            #"(?:às\s*)?(\d{1,2}):(\d{2})"#,
            #"(?:às\s*)?(\d{1,2})[hH](\d{2})"#,
            #"(\d{1,2})\s*(?:e\s*)?(\d{2})?\s*horas?"#,
            #"(?:horário|hora)\s*(?:é\s*)?(?:às\s*)?(\d{1,2})[:hH]?(\d{2})?"#
        ]
        
        var foundTime = false
        for pattern in timePatterns {
            if let timeMatch = text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
                let timeStr = String(text[timeMatch])
                let time = formatTime(timeStr)
                if !time.isEmpty {
                    entities.append(ExtractedEntity(
                        fieldId: "surgeryTime",
                        value: time,
                        confidence: 0.85,
                        alternatives: [],
                        originalText: text
                    ))
                    print("🕰 Found time \(Self.redactedValue(time))")
                    foundTime = true
                    break
                }
            }
        }
        
        // 6. Extract surgeon - enhanced patterns
        let surgeonPatterns = [
            ("doutor", 1, 3),
            ("doutora", 1, 3),
            ("dr", 1, 3),
            ("dra", 1, 3),
            ("médico", 1, 3),
            ("cirurgião", 1, 3),
            ("preceptor", 1, 3)
        ]

        for (keyword, startOffset, endOffset) in surgeonPatterns {
            if let keyIndex = words.firstIndex(where: { 
                $0 == keyword || $0 == "\(keyword)."
            }),
               keyIndex + startOffset < words.count {
                let endIndex = min(keyIndex + endOffset, words.count - 1)
                let surgeonName = words[keyIndex + startOffset...endIndex]
                    .filter { !["de", "da", "do", "dos", "das"].contains($0) || $0.count > 2 }
                    .joined(separator: " ")

                if !surgeonName.isEmpty && surgeonName.count > 2 {
                    entities.append(ExtractedEntity(
                        fieldId: "surgeonName",
                        value: Self.capitalizeProperly(surgeonName),
                        confidence: 0.75,
                        alternatives: [],
                        originalText: text
                    ))
                    print("👨‍⚕️ Found surgeon \(Self.redactedValue(surgeonName))")
                    break
                }
            }
        }
        
        // 7. Extract procedure - enhanced patterns
        let procedureKeywords = [
            "apendicectomia", "colecistectomia", "hernioplastia", "laparoscopia",
            "artroscopia", "endoscopia", "colonoscopia", "biópsia", "ressecção",
            "implante", "prótese", "cirurgia", "operação", "procedimento",
            "intervenção", "tratamento"
        ]

        for keyword in procedureKeywords {
            if lowercased.contains(keyword) {
                if let procIndex = words.firstIndex(where: { $0.contains(keyword) }) {
                    // Get more context around the procedure
                    let startIdx = max(0, procIndex - 2)
                    let endIdx = min(procIndex + 2, words.count - 1)
                    let procName = words[startIdx...endIdx]
                        .filter { !["de", "da", "do", "a", "o", "para"].contains($0) || $0.contains(keyword) }
                        .joined(separator: " ")
                    
                    entities.append(ExtractedEntity(
                        fieldId: "procedureName",
                        value: Self.capitalizeProperly(procName),
                        confidence: 0.7,
                        alternatives: [],
                        originalText: text
                    ))
                    print("🔧 Found procedure \(Self.redactedValue(procName))")
                    break
                }
            }
        }

        // KB-assisted detection when prefixes/keywords are not present
        func hasEntity(_ id: String) -> Bool { entities.contains { $0.fieldId == id } }

        if !hasEntity("surgeonName") {
            if let kbSurgeon = detectSurgeonViaKnowledgeBase(in: lowercased) {
                entities.append(ExtractedEntity(
                    fieldId: "surgeonName",
                    value: kbSurgeon.value,
                    confidence: kbSurgeon.confidence,
                    alternatives: kbSurgeon.alternatives,
                    originalText: text
                ))
                print("👨‍⚕️ KB matched surgeon \(Self.redactedValue(kbSurgeon.value))")
            }
        }

        if !hasEntity("procedureName") {
            if let kbProcedure = detectProcedureViaKnowledgeBase(in: lowercased) {
                entities.append(ExtractedEntity(
                    fieldId: "procedureName",
                    value: kbProcedure.value,
                    confidence: kbProcedure.confidence,
                    alternatives: kbProcedure.alternatives,
                    originalText: text
                ))
                print("🔧 KB matched procedure \(Self.redactedValue(kbProcedure.value))")
            }
        }
        
        // 8. Extract duration - new field handling
        let durationPatterns = [
            #"(\d+)\s*(?:a\s*)?(\d+)?\s*horas?"#,
            #"(\d+)\s*minutos?"#,
            #"duração\s*(?:de\s*)?(\d+)\s*(?:a\s*)?(\d+)?\s*horas?"#,
            #"tempo\s*(?:estimado\s*)?(?:de\s*)?(\d+)\s*horas?"#
        ]
        
        let hasDurationKeyword = lowercased.contains("duração") || lowercased.contains("duracao") || lowercased.contains("tempo") || lowercased.contains("estimad")
        for pattern in durationPatterns {
            if let durationMatch = text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
                let durationStr = String(text[durationMatch])
                // Disambiguate phrases like "uma hora da tarde" when time already detected
                let segmentLower = durationStr.lowercased()
                let looksLikeClockPhrase = segmentLower.contains("hora") && (lowercased.contains("da tarde") || lowercased.contains("da noite") || lowercased.contains("da manhã") || lowercased.contains("de manhã"))
                if foundTime && !hasDurationKeyword && looksLikeClockPhrase {
                    continue
                }
                // If time was found and there is no duration keyword, be conservative for hours-only matches
                if foundTime && !hasDurationKeyword && (segmentLower.contains("hora") && !segmentLower.contains("minuto")) {
                    continue
                }
                let duration = formatDuration(durationStr)
                if !duration.isEmpty {
                    entities.append(ExtractedEntity(
                        fieldId: "procedureDuration",
                        value: duration,
                        confidence: 0.75,
                        alternatives: [],
                        originalText: text
                    ))
                    print("⏱ Found duration \(Self.redactedValue(duration))")
                    break
                }
            }
        }
        
        print("📊 Fallback extraction complete: \(entities.count) entities found")
        for entity in entities {
            print("  - \(entity.fieldId): \(Self.redactedValue(entity.value)) (confidence: \(Int(entity.confidence * 100))%)")
        }
        
        let result = ExtractionResult(
            entities: entities,
            unprocessedText: "",
            confidence: entities.isEmpty ? 0.3 : Double(entities.count) / 8.0
        )
        os_signpost(.end, log: metricsLog, name: "fallbackExtraction", signpostID: id)
        return result
    }

    // MARK: - KB-assisted recognition helpers
    private static func detectSurgeonViaKnowledgeBase(in lowercasedText: String) -> (value: String, confidence: Double, alternatives: [String])? {
        let tokens = lowercasedText.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map(String.init)
        let windows = ngrams(tokens: tokens, minLen: 1, maxLen: 4)
        var best: (value: String, confidence: Double, alternatives: [String])?
        for w in windows {
            let res = IntelligentMatcher.matchSurgeon(w)
            if res.isKnownEntity && res.confidence >= 0.85 {
                if best == nil || res.confidence > best!.confidence {
                    best = (res.value, res.confidence, res.alternatives)
                }
            }
        }
        return best
    }

    private static func detectProcedureViaKnowledgeBase(in lowercasedText: String) -> (value: String, confidence: Double, alternatives: [String])? {
        let tokens = lowercasedText.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map(String.init)
        let windows = ngrams(tokens: tokens, minLen: 1, maxLen: 6)
        var best: (value: String, confidence: Double, alternatives: [String])?
        for w in windows {
            let res = IntelligentMatcher.matchProcedure(w)
            if res.isKnownEntity && res.confidence >= 0.80 {
                if best == nil || res.confidence > best!.confidence {
                    best = (res.value, res.confidence, res.alternatives)
                }
            }
        }
        return best
    }

    private static func ngrams(tokens: [String], minLen: Int, maxLen: Int) -> [String] {
        var result: [String] = []
        let n = tokens.count
        let maxL = min(maxLen, n)
        for l in minLen...maxL {
            if l <= 0 { continue }
            for i in 0..<(n - l + 1) {
                let slice = tokens[i..<(i + l)]
                result.append(slice.joined(separator: " "))
            }
        }
        return result
    }
    
    private static func formatTime(_ timeStr: String) -> String {
        // Extract hours and minutes from various formats
        let numbers = timeStr.components(separatedBy: CharacterSet.decimalDigits.inverted)
            .compactMap { Int($0) }
        
        if numbers.count >= 2 {
            return String(format: "%02d:%02d", numbers[0], numbers[1])
        } else if numbers.count == 1 && numbers[0] < 24 {
            return String(format: "%02d:00", numbers[0])
        }
        
        return ""
    }
    
    private static func formatDate(_ dateStr: String) -> String {
        let s = dateStr.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        // 1) Numeric date like 1/4/25 or 01-04-2025
        if let re = try? NSRegularExpression(pattern: #"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"#),
           let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)),
           let r1 = Range(m.range(at: 1), in: s), let r2 = Range(m.range(at: 2), in: s), let r3 = Range(m.range(at: 3), in: s) {
            var d = String(s[r1]); var mth = String(s[r2]); var y = String(s[r3])
            if d.count == 1 { d = "0" + d }
            if mth.count == 1 { mth = "0" + mth }
            if y.count == 2 { y = "20" + y }
            return "\(d)/\(mth)/\(y)"
        }
        // 2) Spoken month: 11 de abril de 2025
        let months = ["janeiro": "01", "fevereiro": "02", "março": "03", "abril": "04",
                      "maio": "05", "junho": "06", "julho": "07", "agosto": "08",
                      "setembro": "09", "outubro": "10", "novembro": "11", "dezembro": "12"]
        if let dayMatch = s.range(of: #"\b(\d{1,2})\b"#, options: .regularExpression) {
            let dayRaw = String(s[dayMatch])
            var day = dayRaw.count == 1 ? "0" + dayRaw : dayRaw
            for (name, num) in months where s.contains(name) {
                // Find 4-digit year if present
                var year = Calendar.current.component(.year, from: Date())
                if let yMatch = s.range(of: #"\b(\d{4})\b"#, options: .regularExpression) {
                    year = Int(String(s[yMatch])) ?? year
                }
                return "\(day)/\(num)/\(year)"
            }
        }
        return dateStr
    }
    
    private static func formatDuration(_ durationStr: String) -> String {
        return DurationFormatter.format(durationStr)
    }
    
    func refineEntity(fieldId: String, originalValue: String, context: String) async throws -> ExtractedEntity? {
        guard isAvailable else { return nil }
        
        let session = LanguageModelSession(model: model)
        
        let prompt = """
        Refine esta informação médica extraída:
        
        Campo: \(fieldId)
        Valor original: \(originalValue)
        Contexto: \(context)
        
        Forneça uma versão melhorada seguindo as regras:
        - Nomes próprios com primeira letra maiúscula
        - Datas no formato DD/MM/AAAA
        - Horários no formato HH:MM
        - Telefones apenas números
        
        Responda apenas com o valor refinado, sem explicações.
        """
        
        do {
            let response = try await session.respond(to: prompt)
            let refinedValue = Self.capitalizeProperly(response.content.trimmingCharacters(in: .whitespacesAndNewlines))
            
            return ExtractedEntity(
                fieldId: fieldId,
                value: refinedValue,
                confidence: 0.9,
                alternatives: [originalValue],
                originalText: context
            )
        } catch {
            return nil
        }
    }
}

enum EntityExtractionError: LocalizedError {
    case modelUnavailable
    case sessionUnavailable
    case extractionFailed(Error)
    case invalidResponse
    case timeout
    
    var errorDescription: String? {
        switch self {
        case .modelUnavailable:
            return "Foundation Models não está disponível"
        case .sessionUnavailable:
            return "Sessão do modelo não está disponível"
        case .extractionFailed(let error):
            return "Falha na extração: \(error.localizedDescription)"
        case .invalidResponse:
            return "Resposta inválida do modelo"
        case .timeout:
            return "Tempo limite excedido - usando extração alternativa"
        }
    }
}
