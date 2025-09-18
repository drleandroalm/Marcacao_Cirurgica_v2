import Foundation

struct SurgerySession: Identifiable, Codable, Sendable {
    let id: UUID
    let createdAt: Date
    let patientName: String
    let surgeonName: String
    let procedureName: String
    let surgeryDate: String
    let surgeryTime: String
    let exportedTemplate: String
    let exportedJSON: String?
    let needsCTI: Bool?
    let needsOPME: Bool?
    let needsHemocomponents: Bool?
    let hemocomponentsSpecification: String?
}

extension SurgerySession {
    @MainActor static func from(form: SurgicalRequestForm) -> SurgerySession {
        let fields = form.fields
        func val(_ id: String) -> String {
            fields.first(where: { $0.id == id })?.value ?? ""
        }
        // Build JSON export similar to FormExporter.exportAsJSON
        var json: [String: Any] = [:]
        for field in fields { json[field.id] = field.value.isEmpty ? nil : field.value }
        json["needsCTI"] = form.needsCTI
        json["needsOPME"] = form.needsOPME
        json["needsHemocomponents"] = form.needsHemocomponents
        json["hemocomponentsSpecification"] = form.hemocomponentsSpecification
        json["exportDate"] = ISO8601DateFormatter().string(from: Date())
        let exportedJSON: String? = (try? JSONSerialization.data(withJSONObject: json, options: .prettyPrinted)).flatMap { String(data: $0, encoding: .utf8) }
        return SurgerySession(
            id: UUID(),
            createdAt: Date(),
            patientName: val("patientName"),
            surgeonName: val("surgeonName"),
            procedureName: val("procedureName"),
            surgeryDate: val("surgeryDate"),
            surgeryTime: val("surgeryTime"),
            exportedTemplate: form.generateFilledTemplate(),
            exportedJSON: exportedJSON,
            needsCTI: form.needsCTI,
            needsOPME: form.needsOPME,
            needsHemocomponents: form.needsHemocomponents,
            hemocomponentsSpecification: form.hemocomponentsSpecification
        )
    }
}
