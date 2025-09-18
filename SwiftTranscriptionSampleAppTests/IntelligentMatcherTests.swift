import XCTest
@testable import SwiftTranscriptionSampleApp

final class IntelligentMatcherTests: XCTestCase {
    func test_WhenMatchingKnownSurgeon_ShouldReturnHighConfidence() {
        let result = IntelligentMatcher.matchSurgeon("Dr. Wadson")
        XCTAssertTrue(result.isKnownEntity)
        XCTAssertGreaterThanOrEqual(result.confidence, 0.7)
        XCTAssertEqual(result.value, "Wadson Miconi")
    }
    
    func test_WhenMatchingUnknownSurgeon_ShouldReturnUnknownEntity() {
        let result = IntelligentMatcher.matchSurgeon("Fulano Exemplo")
        XCTAssertFalse(result.isKnownEntity)
        XCTAssertLessThan(result.confidence, 0.8)
    }
    
    func test_WhenMatchingProcedureAbbreviation_ShouldExpandToCanonicalName() {
        let result = IntelligentMatcher.matchProcedure("RTUP")
        XCTAssertTrue(result.isKnownEntity)
        XCTAssertEqual(result.value, "RTU de Próstata")
        XCTAssertGreaterThan(result.confidence, 0.7)
    }
}
