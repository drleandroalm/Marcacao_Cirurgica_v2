import XCTest
@testable import SwiftTranscriptionSampleApp

final class DurationFormatterTests: XCTestCase {
    func test_WhenUmaHoraEMeia_ShouldReturn0130() {
        XCTAssertEqual(DurationFormatter.format("uma hora e meia"), "01:30")
    }

    func test_WhenHColonM_ShouldReturnNormalized() {
        XCTAssertEqual(DurationFormatter.format("1:05"), "01:05")
        XCTAssertEqual(DurationFormatter.format("1h30"), "01:30")
    }

    func test_WhenMinutesOnly_ShouldReturn00MM() {
        XCTAssertEqual(DurationFormatter.format("45"), "00:45")
    }

    func test_WhenNumbersWithUnits_ShouldParse() {
        XCTAssertEqual(DurationFormatter.format("2 horas 15 minutos"), "02:15")
    }

    func test_WhenWordsOnly_ShouldParse() {
        XCTAssertEqual(DurationFormatter.format("duas horas"), "02:00")
    }
}

