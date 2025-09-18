import XCTest
@testable import SwiftTranscriptionSampleApp

@MainActor
final class ComplianceValidatorTests: XCTestCase {
    func test_WhenRequiredFieldsMissing_ShouldReportError() {
        let form = SurgicalRequestForm()
        let result = ComplianceValidator.validate(form: form)
        XCTAssertTrue(result.issues.contains(where: { $0.type == .missingRequired && $0.field == "Nome do Paciente" }))
        XCTAssertTrue(result.issues.contains(where: { $0.severity == .error }))
    }
    
    func test_WhenFieldsMatchKnowledgeBase_ShouldVerifyEntities() {
        let form = SurgicalRequestForm()
        update(form: form, fieldId: "patientName", value: "João Silva")
        update(form: form, fieldId: "patientAge", value: "45")
        update(form: form, fieldId: "patientPhone", value: "21999887766")
        update(form: form, fieldId: "surgeonName", value: "Wadson Miconi")
        update(form: form, fieldId: "surgeryDate", value: "27/09/2024")
        update(form: form, fieldId: "surgeryTime", value: "14:30")
        update(form: form, fieldId: "procedureName", value: "RTU de Bexiga")
        update(form: form, fieldId: "procedureDuration", value: "01:30")
        
        let result = ComplianceValidator.validate(form: form)
        XCTAssertTrue(result.issues.isEmpty)
        if case .verified(let confidence)? = result.fieldStatuses["surgeonName"] {
            XCTAssertGreaterThanOrEqual(confidence, 0.9)
        } else {
            XCTFail("Expected surgeon to be verified")
        }
        XCTAssertGreaterThan(result.overallConfidence, 0.0)
    }
    
    private func update(form: SurgicalRequestForm, fieldId: String, value: String) {
        if let index = form.fields.firstIndex(where: { $0.id == fieldId }) {
            form.fields[index].value = value
        }
    }
}
