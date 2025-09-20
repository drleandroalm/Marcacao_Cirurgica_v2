import XCTest
@testable import SwiftTranscriptionSampleApp

@MainActor
final class FormPreviewConfidenceTests: XCTestCase {

    // Redacts any value for safe logging in tests
    private func redacted(_ value: String) -> String { "<len=\(value.count)>" }

    private func sanitizeForStorage(_ type: FieldType, _ raw: String) -> String {
        switch type {
        case .age:
            return raw.filter { $0.isNumber }
        case .date:
            return TemplateField(id: "tmp", label: "", placeholder: "", value: raw, fieldType: .date, isRequired: true).formattedValue()
        case .time:
            return MilitaryTimeFormatter.format(raw)
        case .duration:
            return DurationFormatter.format(raw)
        case .phone:
            var digits = raw.filter { $0.isNumber }
            if digits.first == "0" { digits.removeFirst() }
            return digits
        default:
            return raw
        }
    }

    func test_WhenSampleTranscriptProvided_ShouldProduceRedactedConfidencePreviewOutput() throws {
        // Sample BR-PT transcription (values will be redacted in logs)
        let transcript = """
        Paciente Joao da Silva, 45 anos, telefone 21 99988 7766.\n
        Cirurgiao Wadson Miconi. Procedimento RTU de Bexiga.\n
        Data 27/09/2025, horario 14:30. Duracao 1 hora e 30 minutos.
        """

        // Use deterministic fallback to extract entities
        let result = try EntityExtractor.fallbackExtraction(from: transcript)
        XCTAssertGreaterThan(result.entities.count, 0, "Expected entities from fallback extraction")

        // Fill form like the preview .task does
        let form = SurgicalRequestForm()
        for e in result.entities {
            if let idx = form.fields.firstIndex(where: { $0.id == e.fieldId }) {
                let type = form.fields[idx].fieldType
                form.fields[idx].value = sanitizeForStorage(type, e.value)
            }
        }

        let validation = ComplianceValidator.validate(form: form)

        // Summarize (redacted) so Tools script can grep
        let overallPct = Int(result.confidence * 100)
        let validationPct = Int(validation.overallConfidence * 100)
        let warnCount = validation.issues.filter { $0.severity == .warning }.count
        let errCount = validation.issues.filter { $0.severity == .error }.count
        print("[PREVIEW] entities=\(result.entities.count) overall=\(overallPct)% validation_overall=\(validationPct)% warnings=\(warnCount) errors=\(errCount)")
        for e in result.entities {
            print("[CONF] field=\(e.fieldId) confidence=\(Int(e.confidence * 100))% value=\(redacted(e.value))")
        }

        // Threshold assertions (configurable via environment variables)
        let env = ProcessInfo.processInfo.environment
        let minEntities = Int(env["PREVIEW_MIN_ENTITIES"] ?? "6") ?? 6
        let minValidationPct = Int(env["PREVIEW_MIN_VALIDATION_OVERALL_PCT"] ?? "70") ?? 70
        let maxWarnings = Int(env["PREVIEW_MAX_WARNINGS"] ?? "1") ?? 1
        let maxErrors = Int(env["PREVIEW_MAX_ERRORS"] ?? "0") ?? 0

        XCTAssertGreaterThanOrEqual(result.entities.count, minEntities, "Preview extracted too few entities")
        XCTAssertGreaterThanOrEqual(validationPct, minValidationPct, "Validation overall confidence below threshold")
        XCTAssertLessThanOrEqual(warnCount, maxWarnings, "Too many validation warnings")
        XCTAssertLessThanOrEqual(errCount, maxErrors, "Validation errors present")
        }

    func test_Performance_PreviewPipeline_ShouldReportMetrics() throws {
        let transcript = """
        Paciente Maria dos Santos, 63 anos, telefone 31 98877 6655.\n
        Cirurgiao Wadson Miconi. Procedimento RTUP.\n
        Data 28/09/2025, horario 08:45. Duracao 90 minutos.
        """

        let iterations = 10
        var samples: [Double] = [] // milliseconds
        samples.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            let result = try EntityExtractor.fallbackExtraction(from: transcript)
            XCTAssertGreaterThan(result.entities.count, 0)
            let form = SurgicalRequestForm()
            for e in result.entities {
                if let idx = form.fields.firstIndex(where: { $0.id == e.fieldId }) {
                    let type = form.fields[idx].fieldType
                    form.fields[idx].value = sanitizeForStorage(type, e.value)
                }
            }
            _ = ComplianceValidator.validate(form: form)
            let end = CFAbsoluteTimeGetCurrent()
            samples.append((end - start) * 1000.0)
        }

        let avg = samples.reduce(0, +) / Double(samples.count)
        let sorted = samples.sorted()
        let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
        print(String(format: "[PERF] iterations=%d avg_ms=%.2f p95_ms=%.2f", iterations, avg, p95))

        // Threshold assertions (configurable via environment variables)
        let env = ProcessInfo.processInfo.environment
        let avgMax = Double(env["PREVIEW_PERF_AVG_MAX_MS"] ?? "3000") ?? 3000
        let p95Max = Double(env["PREVIEW_PERF_P95_MAX_MS"] ?? "3500") ?? 3500
        XCTAssertLessThanOrEqual(avg, avgMax, "Average latency regression: avg_ms=\(avg) > \(avgMax)")
        XCTAssertLessThanOrEqual(p95, p95Max, "P95 latency regression: p95_ms=\(p95) > \(p95Max)")
    }
}
