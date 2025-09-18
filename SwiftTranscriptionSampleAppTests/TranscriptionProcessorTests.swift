import XCTest
@testable import SwiftTranscriptionSampleApp

@MainActor
final class TranscriptionProcessorTests: XCTestCase {
    func test_WhenPortugueseNumberPhraseProvided_ShouldReturnDigits() {
        let input = "setenta e dois anos"
        let processed = TranscriptionProcessor.processText(input, fieldType: .age)
        XCTAssertEqual(processed, "72")
    }
    
    func test_WhenPortugueseDatePhraseProvided_ShouldReturnNormalizedDate() {
        let input = "vinte e sete de setembro de dois mil e vinte e quatro"
        let processed = TranscriptionProcessor.processText(input, fieldType: .date)
        XCTAssertEqual(processed, "27/09/2024")
    }
    
    func test_WhenDurationSpoken_ShouldReturnHHMMFormat() {
        let input = "uma hora e quinze"
        let processed = TranscriptionProcessor.processText(input, fieldType: .duration)
        XCTAssertEqual(processed, "01:15")
    }
    
    func test_WhenPhoneDictated_ShouldStripFormatting() {
        let input = "(21) 99988-7766"
        let processed = TranscriptionProcessor.processText(input, fieldType: .phone)
        XCTAssertEqual(processed, "21999887766")
    }
}
