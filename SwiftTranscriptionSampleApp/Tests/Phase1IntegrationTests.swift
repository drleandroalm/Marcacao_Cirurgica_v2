import XCTest
@testable import SwiftTranscriptionSampleApp

/// Unit tests for Phase 1 entity recognition enhancements
final class Phase1IntegrationTests: XCTestCase {

    // MARK: - Pattern Configuration Tests

    func testPatternConfigurationLoadsSuccessfully() {
        let config = PatternConfiguration.shared

        XCTAssertEqual(config.locale, "pt-BR", "Configuration should be Brazilian Portuguese")
        XCTAssertFalse(config.patterns.isEmpty, "Patterns should not be empty")
    }

    func testPatternConfigurationHasAllEntityTypes() {
        let config = PatternConfiguration.shared

        let expectedEntityTypes = [
            "patientName",
            "patientAge",
            "patientPhone",
            "surgeryDate",
            "surgeryTime",
            "surgeonName",
            "procedureName",
            "procedureDuration"
        ]

        for entityType in expectedEntityTypes {
            XCTAssertNotNil(config.patterns[entityType], "Missing pattern for \(entityType)")
        }
    }

    func testPatternConfidenceScoresAreValid() {
        let config = PatternConfiguration.shared

        for (entityType, patternConfig) in config.patterns {
            // Test regex patterns
            for regexPattern in patternConfig.regexPatterns {
                XCTAssertTrue((0.0...1.0).contains(regexPattern.confidence),
                             "\(entityType): Confidence score \(regexPattern.confidence) out of range")
            }

            // Test context keywords
            for keyword in patternConfig.contextKeywords {
                XCTAssertTrue((0.0...1.0).contains(keyword.confidence),
                             "\(entityType): Context keyword confidence \(keyword.confidence) out of range")
            }

            // Test relative keywords
            for relKeyword in patternConfig.relativeKeywords {
                XCTAssertTrue((0.0...1.0).contains(relKeyword.confidence),
                             "\(entityType): Relative keyword confidence \(relKeyword.confidence) out of range")
            }
        }
    }

    // MARK: - Enhanced Knowledge Base Tests

    func testEnhancedProceduresLoad() {
        let procedures = MedicalKnowledgeBase.procedures

        XCTAssertFalse(procedures.isEmpty, "Procedures should not be empty")
        XCTAssertGreaterThanOrEqual(procedures.count, 36, "Should have at least 36 procedures")
    }

    func testProceduresHaveSNOMEDCTCodes() {
        let procedures = MedicalKnowledgeBase.procedures

        var proceduresWithSNOMED = 0
        for procedure in procedures {
            if let snomedCT = procedure.snomedCT {
                // SNOMED-CT codes are numeric strings
                XCTAssertTrue(snomedCT.allSatisfy { $0.isNumber },
                             "\(procedure.canonical): SNOMED-CT code should be numeric")
                XCTAssertGreaterThan(snomedCT.count, 3,
                                    "\(procedure.canonical): SNOMED-CT code too short")
                proceduresWithSNOMED += 1
            }
        }

        print("✅ \(proceduresWithSNOMED)/\(procedures.count) procedures have SNOMED-CT codes")
        XCTAssertGreaterThan(proceduresWithSNOMED, 0, "At least some procedures should have SNOMED-CT codes")
    }

    func testProceduresHaveICD10Codes() {
        let procedures = MedicalKnowledgeBase.procedures

        var proceduresWithICD10 = 0
        for procedure in procedures {
            if let icd10codes = procedure.icd10pcs, !icd10codes.isEmpty {
                for code in icd10codes {
                    // ICD-10-PCS codes are 7 alphanumeric characters
                    XCTAssertEqual(code.count, 7,
                                  "\(procedure.canonical): ICD-10-PCS code should be 7 characters")
                    proceduresWithICD10 += 1
                }
            }
        }

        print("✅ \(proceduresWithICD10)/\(procedures.count) procedures have ICD-10-PCS codes")
        XCTAssertGreaterThan(proceduresWithICD10, 0, "At least some procedures should have ICD-10-PCS codes")
    }

    func testProceduresHaveMetadata() {
        let procedures = MedicalKnowledgeBase.procedures

        var proceduresWithSpecialty = 0
        var proceduresWithAnesthesia = 0

        for procedure in procedures {
            if procedure.specialty != nil {
                proceduresWithSpecialty += 1
            }
            if procedure.typicalAnesthesia != nil {
                proceduresWithAnesthesia += 1
            }
        }

        print("✅ \(proceduresWithSpecialty)/\(procedures.count) procedures have specialty")
        print("✅ \(proceduresWithAnesthesia)/\(procedures.count) procedures have anesthesia type")
    }

    func testExpandedAbbreviationsLoad() {
        let abbreviations = MedicalKnowledgeBase.abbreviationExpansions

        XCTAssertFalse(abbreviations.isEmpty, "Abbreviations should not be empty")
        XCTAssertGreaterThanOrEqual(abbreviations.count, 16, "Should have at least 16 abbreviations")
        print("✅ Loaded \(abbreviations.count) abbreviations")
    }

    func testAbbreviationExpansion() {
        let testCases: [(String, String)] = [
            ("OSC", "Orquiectomia Subcapsular Bilateral"),
            ("RTU", "Ressecção Transuretral"),
            ("VLP", "Videolaparoscópica"),
            ("IPP", "Implante de Prótese Peniana"),
            ("LECO", "Litotripsia Extracorpórea por Ondas de Choque")
        ]

        for (abbr, expectedExpansion) in testCases {
            let expanded = MedicalKnowledgeBase.expandAbbreviation(abbr)
            XCTAssertEqual(expanded, expectedExpansion,
                          "Abbreviation \(abbr) should expand to \(expectedExpansion)")
        }
    }

    // MARK: - Enhanced Confidence Scoring Tests

    func testAutoAcceptThreshold() {
        // High confidence across all factors
        let highConfidence = EnhancedConfidence(
            transcriptionQuality: 0.98,
            entityMatch: 0.95,
            contextConsistency: 0.92,
            historicalAccuracy: 0.95
        )

        XCTAssertTrue(highConfidence.shouldAutoAccept(),
                     "Should auto-accept with high confidence across all factors")
        XCTAssertEqual(highConfidence.uiColor, .green)
    }

    func testRequiresConfirmationThreshold() {
        // Medium confidence
        let mediumConfidence = EnhancedConfidence(
            transcriptionQuality: 0.85,
            entityMatch: 0.80,
            contextConsistency: 0.75,
            historicalAccuracy: 0.80
        )

        XCTAssertTrue(mediumConfidence.requiresConfirmation(),
                     "Should require confirmation with medium confidence")
        XCTAssertEqual(mediumConfidence.uiColor, .orange)
    }

    func testShouldRejectThreshold() {
        // Low confidence
        let lowConfidence = EnhancedConfidence(
            transcriptionQuality: 0.60,
            entityMatch: 0.55,
            contextConsistency: 0.50,
            historicalAccuracy: 0.60
        )

        XCTAssertTrue(lowConfidence.shouldReject(),
                     "Should reject with low confidence")
        XCTAssertEqual(lowConfidence.uiColor, .red)
    }

    func testDateEntityScoring() {
        // Relative keyword (high confidence)
        let relativeConf = ConfidenceScorer.scoreDateEntity(
            extractedDate: "14/11/2025",
            isRelativeKeyword: true,
            hasExplicitContext: false
        )

        XCTAssertGreaterThan(relativeConf.entityMatch, 0.90,
                           "Relative keywords should have high entity match confidence")

        // Explicit context (medium-high confidence)
        let explicitConf = ConfidenceScorer.scoreDateEntity(
            extractedDate: "15/03/2025",
            isRelativeKeyword: false,
            hasExplicitContext: true
        )

        XCTAssertGreaterThan(explicitConf.entityMatch, 0.80,
                           "Explicit context should have medium-high confidence")
        XCTAssertLessThan(explicitConf.entityMatch, relativeConf.entityMatch,
                         "Relative keywords should be more confident than explicit context")
    }

    func testPhoneEntityScoring() {
        // 11 digits with context (high confidence)
        let fullPhoneConf = ConfidenceScorer.scorePhoneEntity(
            extractedPhone: "11987654321",
            digitCount: 11,
            hasContext: true
        )

        XCTAssertGreaterThan(fullPhoneConf.entityMatch, 0.80,
                           "11 digits with context should have high confidence")

        // 8 digits no context (lower confidence)
        let shortPhoneConf = ConfidenceScorer.scorePhoneEntity(
            extractedPhone: "87654321",
            digitCount: 8,
            hasContext: false
        )

        XCTAssertLessThan(shortPhoneConf.entityMatch, fullPhoneConf.entityMatch,
                         "Shorter phone without context should have lower confidence")
    }

    func testKnowledgeBaseEntityScoring() {
        // Exact canonical match
        let exactConf = ConfidenceScorer.scoreKnowledgeBaseEntity(
            matchType: .exactCanonical
        )

        XCTAssertGreaterThan(exactConf.entityMatch, 0.90,
                           "Exact canonical match should have very high confidence")

        // Fuzzy match
        let fuzzyConf = ConfidenceScorer.scoreKnowledgeBaseEntity(
            matchType: .fuzzyMatch,
            fuzzyMatchScore: 0.85
        )

        XCTAssertLessThan(fuzzyConf.entityMatch, exactConf.entityMatch,
                         "Fuzzy match should have lower confidence than exact match")
    }

    // MARK: - Advanced Temporal Expression Tests

    func testRelativeDaysExtraction() {
        let testCases: [(String, Int)] = [
            ("A cirurgia será daqui a 3 dias", 3),
            ("Agendar em 5 dias", 5),
            ("Dentro de 10 dias", 10)
        ]

        for (text, expectedDays) in testCases {
            if let result = AdvancedTemporalExtractor.extractAdvancedDate(from: text) {
                let expectedDate = Calendar.current.date(byAdding: .day, value: expectedDays, to: Date())!
                let calendar = Calendar.current

                XCTAssertEqual(
                    calendar.component(.day, from: result.date),
                    calendar.component(.day, from: expectedDate),
                    "Day should match for: \(text)"
                )

                XCTAssertEqual(
                    calendar.component(.month, from: result.date),
                    calendar.component(.month, from: expectedDate),
                    "Month should match for: \(text)"
                )

                XCTAssertGreaterThan(result.confidence, 0.80,
                                   "Confidence should be high for: \(text)")
            } else {
                XCTFail("Failed to extract date from: \(text)")
            }
        }
    }

    func testRelativeWeeksExtraction() {
        let testCases: [(String, Int)] = [
            ("Daqui a 2 semanas", 14),
            ("Em 1 semana", 7),
            ("Dentro de 3 semanas", 21)
        ]

        for (text, expectedDays) in testCases {
            if let result = AdvancedTemporalExtractor.extractAdvancedDate(from: text) {
                let expectedDate = Calendar.current.date(byAdding: .day, value: expectedDays, to: Date())!
                let calendar = Calendar.current

                XCTAssertEqual(
                    calendar.component(.day, from: result.date),
                    calendar.component(.day, from: expectedDate),
                    "Day should match for: \(text)"
                )

                XCTAssertGreaterThan(result.confidence, 0.80,
                                   "Confidence should be high for: \(text)")
            } else {
                XCTFail("Failed to extract date from: \(text)")
            }
        }
    }

    func testNextDayOfMonthExtraction() {
        let testCases = [
            "Próximo dia 15",
            "Dia 20 do próximo mês",
            "No dia 25 do próximo"
        ]

        for text in testCases {
            if let result = AdvancedTemporalExtractor.extractAdvancedDate(from: text) {
                // Just verify we got a future date
                XCTAssertGreaterThan(result.date, Date(),
                                   "Extracted date should be in the future for: \(text)")

                XCTAssertGreaterThan(result.confidence, 0.80,
                                   "Confidence should be high for: \(text)")
            } else {
                XCTFail("Failed to extract date from: \(text)")
            }
        }
    }

    func testTemporalExpressionFormatting() {
        let date = Calendar.current.date(from: DateComponents(year: 2025, month: 3, day: 15))!
        let formatted = AdvancedTemporalExtractor.formatDate(date)

        XCTAssertEqual(formatted, "15/03/2025",
                      "Date should be formatted as dd/MM/yyyy (Brazilian format)")
    }

    // MARK: - Integration Tests

    func testBackwardCompatibility() {
        // Verify that loading procedures without enhanced fields still works
        let procedures = MedicalKnowledgeBase.procedures

        for procedure in procedures {
            XCTAssertFalse(procedure.canonical.isEmpty,
                          "All procedures should have canonical name")
            XCTAssertTrue(procedure.typicalDuration.lowerBound > 0,
                         "\(procedure.canonical): Duration should be positive")

            // Enhanced fields are optional - shouldn't crash if nil
            _ = procedure.snomedCT
            _ = procedure.icd10pcs
            _ = procedure.specialty
            _ = procedure.requiredOPME
            _ = procedure.typicalAnesthesia
        }
    }

    func testAbbreviationExpansionCaseInsensitive() {
        let testCases: [(String, String)] = [
            ("osc", "Orquiectomia Subcapsular Bilateral"),
            ("OSC", "Orquiectomia Subcapsular Bilateral"),
            ("rtu", "Ressecção Transuretral"),
            ("RTU", "Ressecção Transuretral")
        ]

        for (abbr, expectedExpansion) in testCases {
            let expanded = MedicalKnowledgeBase.expandAbbreviation(abbr)
            XCTAssertEqual(expanded, expectedExpansion,
                          "Abbreviation \(abbr) should expand case-insensitively")
        }
    }

    // MARK: - Performance Tests

    func testPatternConfigurationLoadPerformance() {
        measure {
            _ = PatternConfiguration.shared
        }
    }

    func testAdvancedTemporalExtractionPerformance() {
        let testText = "A cirurgia será daqui a 5 dias no hospital"

        measure {
            _ = AdvancedTemporalExtractor.extractAdvancedDate(from: testText)
        }
    }
}
