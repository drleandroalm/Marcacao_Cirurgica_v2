import Foundation
import SwiftUI
import os

enum ComplianceStatus {
    case verified(confidence: Double)
    case suggested(alternatives: [String])
    case unverified
    case warning(reason: String)
    
    var icon: String {
        switch self {
        case .verified: return "checkmark.seal.fill"
        case .suggested: return "exclamationmark.triangle.fill"
        case .unverified: return "info.circle"
        case .warning: return "exclamationmark.octagon.fill"
        }
    }
    
    var color: Color {
        switch self {
        case .verified: return .green
        case .suggested: return .orange
        case .unverified: return .blue
        case .warning: return .red
        }
    }
    
    var description: String {
        switch self {
        case .verified(let confidence):
            return "Verificado (\(Int(confidence * 100))%)"
        case .suggested:
            return "Sugestões disponíveis"
        case .unverified:
            return "Não verificado"
        case .warning(let reason):
            return reason
        }
    }
    
    var hasAlternatives: Bool {
        if case .suggested = self {
            return true
        }
        return false
    }
}

struct ComplianceIssue {
    enum IssueType {
        case unusualCombination
        case atypicalDuration
        case unknownEntity
        case invalidFormat
        case missingRequired
    }
    
    let type: IssueType
    let field: String
    let message: String
    let suggestion: String?
    let severity: Severity
    
    enum Severity {
        case warning
        case error
        case info
        
        var color: Color {
            switch self {
            case .warning: return .orange
            case .error: return .red
            case .info: return .blue
            }
        }
    }
}

struct ValidationResult {
    let issues: [ComplianceIssue]
    let overallConfidence: Double
    let fieldStatuses: [String: ComplianceStatus]
    
    var isValid: Bool {
        return !issues.contains { $0.severity == .error }
    }
    
    var hasWarnings: Bool {
        return issues.contains { $0.severity == .warning }
    }
}

class ComplianceValidator {
    private static let metricsLog = OSLog(subsystem: "SwiftTranscriptionSampleApp", category: "ComplianceValidator")
    
    // MARK: - Main Validation Method
    @MainActor
    static func validate(form: SurgicalRequestForm) -> ValidationResult {
        let id = OSSignpostID(log: metricsLog)
        os_signpost(.begin, log: metricsLog, name: "validate(form:)", signpostID: id)
        var issues: [ComplianceIssue] = []
        var fieldStatuses: [String: ComplianceStatus] = [:]
        var confidenceScores: [Double] = []
        
        // Validate each field
        for field in form.fields {
            let fieldID = OSSignpostID(log: metricsLog)
            os_signpost(.begin, log: metricsLog, name: "validateField", signpostID: fieldID, "id=%{public}s", field.id)
            let (status, fieldIssues, confidence) = validateField(field)
            os_signpost(.end, log: metricsLog, name: "validateField", signpostID: fieldID, "id=%{public}s", field.id)
            fieldStatuses[field.id] = status
            issues.append(contentsOf: fieldIssues)
            confidenceScores.append(confidence)
        }
        
        // Cross-field validation
        let xID = OSSignpostID(log: metricsLog)
        os_signpost(.begin, log: metricsLog, name: "validateCrossFields", signpostID: xID)
        let crossFieldIssues = validateCrossFields(form: form)
        os_signpost(.end, log: metricsLog, name: "validateCrossFields", signpostID: xID)
        issues.append(contentsOf: crossFieldIssues)
        
        // Calculate overall confidence
        let overallConfidence = confidenceScores.isEmpty ? 0 : confidenceScores.reduce(0, +) / Double(confidenceScores.count)
        
        let result = ValidationResult(
            issues: issues,
            overallConfidence: overallConfidence,
            fieldStatuses: fieldStatuses
        )
        os_signpost(.end, log: metricsLog, name: "validate(form:)", signpostID: id)
        return result
    }
    
    // MARK: - Field Validation
    private static func validateField(_ field: TemplateField) -> (ComplianceStatus, [ComplianceIssue], Double) {
        var issues: [ComplianceIssue] = []
        var confidence: Double = 0
        var status: ComplianceStatus = .unverified
        
        // Check if field is empty
        if field.value.isEmpty {
            if field.isRequired {
                issues.append(ComplianceIssue(
                    type: .missingRequired,
                    field: field.label,
                    message: "\(field.label) é obrigatório",
                    suggestion: nil,
                    severity: .error
                ))
                status = .warning(reason: "Campo obrigatório vazio")
            }
            return (status, issues, 0)
        }
        
        // Field-specific validation
        switch field.id {
        case "surgeonName":
            let matchResult = IntelligentMatcher.matchSurgeon(field.value)
            confidence = matchResult.confidence
            
            if matchResult.isKnownEntity {
                if matchResult.confidence >= 0.9 {
                    status = .verified(confidence: matchResult.confidence)
                } else if matchResult.confidence >= 0.7 {
                    status = .suggested(alternatives: matchResult.alternatives)
                    issues.append(ComplianceIssue(
                        type: .unusualCombination,
                        field: field.label,
                        message: "Cirurgião pode estar incorreto",
                        suggestion: "Você quis dizer '\(matchResult.value)'?",
                        severity: .warning
                    ))
                } else {
                    status = .unverified
                }
            } else {
                status = .unverified
                issues.append(ComplianceIssue(
                    type: .unknownEntity,
                    field: field.label,
                    message: "Cirurgião não reconhecido",
                    suggestion: "Verifique o nome do cirurgião",
                    severity: .info
                ))
            }
            
        case "procedureName":
            let matchResult = IntelligentMatcher.matchProcedure(field.value)
            confidence = matchResult.confidence
            
            if matchResult.isKnownEntity {
                if matchResult.confidence >= 0.9 {
                    status = .verified(confidence: matchResult.confidence)
                } else if matchResult.confidence >= 0.7 {
                    status = .suggested(alternatives: matchResult.alternatives)
                    issues.append(ComplianceIssue(
                        type: .unusualCombination,
                        field: field.label,
                        message: "Procedimento pode estar incorreto",
                        suggestion: "Você quis dizer '\(matchResult.value)'?",
                        severity: .warning
                    ))
                } else {
                    status = .unverified
                }
            } else {
                status = .unverified
                issues.append(ComplianceIssue(
                    type: .unknownEntity,
                    field: field.label,
                    message: "Procedimento não reconhecido",
                    suggestion: "Verifique o nome do procedimento",
                    severity: .info
                ))
            }
            
        case "patientAge":
            if let age = Int(field.value) {
                confidence = 1.0
                if age < 0 || age > 120 {
                    status = .warning(reason: "Idade inválida")
                    issues.append(ComplianceIssue(
                        type: .invalidFormat,
                        field: field.label,
                        message: "Idade fora do intervalo esperado",
                        suggestion: "Verifique se a idade está correta",
                        severity: .warning
                    ))
                } else {
                    status = .verified(confidence: 1.0)
                }
            } else {
                status = .warning(reason: "Formato inválido")
                issues.append(ComplianceIssue(
                    type: .invalidFormat,
                    field: field.label,
                    message: "Idade deve ser um número",
                    suggestion: nil,
                    severity: .error
                ))
            }
            
        case "surgeryDate":
            if isValidDate(field.value) {
                confidence = 1.0
                status = .verified(confidence: 1.0)
            } else {
                status = .warning(reason: "Formato de data inválido")
                issues.append(ComplianceIssue(
                    type: .invalidFormat,
                    field: field.label,
                    message: "Use o formato DD/MM/AAAA",
                    suggestion: nil,
                    severity: .error
                ))
            }
            
        case "surgeryTime":
            if isValidTime(field.value) {
                confidence = 1.0
                status = .verified(confidence: 1.0)
            } else {
                status = .warning(reason: "Formato de horário inválido")
                issues.append(ComplianceIssue(
                    type: .invalidFormat,
                    field: field.label,
                    message: "Use o formato HH:MM",
                    suggestion: nil,
                    severity: .error
                ))
            }
            
        case "patientPhone":
            let cleaned = field.value.filter { $0.isNumber }
            if cleaned.count == 10 || cleaned.count == 11 {
                confidence = 1.0
                status = .verified(confidence: 1.0)
            } else {
                status = .warning(reason: "Telefone inválido")
                issues.append(ComplianceIssue(
                    type: .invalidFormat,
                    field: field.label,
                    message: "Telefone deve ter 10 ou 11 dígitos",
                    suggestion: nil,
                    severity: .error
                ))
            }
            
        case "procedureDuration":
            if let _ = extractDurationInMinutes(field.value) {
                confidence = 0.8
                status = .verified(confidence: 0.8)
            } else {
                status = .unverified
                confidence = 0.5
            }
            
        default:
            // Generic text field validation
            if !field.value.isEmpty {
                status = .verified(confidence: 0.8)
                confidence = 0.8
            }
        }
        
        return (status, issues, confidence)
    }
    
    // MARK: - Cross-Field Validation
    @MainActor
    private static func validateCrossFields(form: SurgicalRequestForm) -> [ComplianceIssue] {
        var issues: [ComplianceIssue] = []
        
        // Get surgeon and procedure
        let surgeonField = form.fields.first { $0.id == "surgeonName" }
        let procedureField = form.fields.first { $0.id == "procedureName" }
        
        if let surgeon = surgeonField?.value,
           let procedure = procedureField?.value,
           !surgeon.isEmpty && !procedure.isEmpty {
            
            // Check if this is a common combination
            if !isCommonCombination(surgeon: surgeon, procedure: procedure) {
                issues.append(ComplianceIssue(
                    type: .unusualCombination,
                    field: "Combinação",
                    message: "Combinação incomum de cirurgião e procedimento",
                    suggestion: "Verifique se \(surgeon) realiza \(procedure)",
                    severity: .info
                ))
            }
        }
        
        // Check date and time consistency
        let dateField = form.fields.first { $0.id == "surgeryDate" }
        let timeField = form.fields.first { $0.id == "surgeryTime" }
        
        if let date = dateField?.value,
           let time = timeField?.value,
           !date.isEmpty && !time.isEmpty {
            
            if let surgeryDateTime = combineDateAndTime(date: date, time: time) {
                let now = Date()
                let hoursDifference = surgeryDateTime.timeIntervalSince(now) / 3600
                
                if hoursDifference < 24 && hoursDifference > 0 {
                    issues.append(ComplianceIssue(
                        type: .unusualCombination,
                        field: "Agendamento",
                        message: "Cirurgia agendada com menos de 24 horas de antecedência",
                        suggestion: nil,
                        severity: .warning
                    ))
                }
            }
        }
        
        return issues
    }
    
    // MARK: - Helper Methods
    private static func isValidDate(_ dateString: String) -> Bool {
        let components = dateString.split(separator: "/")
        guard components.count == 3,
              let day = Int(components[0]), day >= 1, day <= 31,
              let month = Int(components[1]), month >= 1, month <= 12,
              let year = Int(components[2]), year >= 2024, year <= 2100 else {
            return false
        }
        return true
    }
    
    private static func parseDate(_ dateString: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd/MM/yyyy"
        formatter.locale = Locale(identifier: "pt_BR")
        return formatter.date(from: dateString)
    }
    
    private static func isValidTime(_ timeString: String) -> Bool {
        let components = timeString.split(separator: ":")
        guard components.count == 2,
              let hour = Int(components[0]), hour >= 0, hour <= 23,
              let minute = Int(components[1]), minute >= 0, minute <= 59 else {
            return false
        }
        return true
    }
    
    private static func extractDurationInMinutes(_ text: String) -> Int? {
        // Try to extract duration in various formats
        let lowercased = text.lowercased()
        
        // "2 horas" -> 120 minutes
        if let hoursMatch = lowercased.range(of: #"(\d+)\s*hora"#, options: .regularExpression) {
            let hoursString = String(lowercased[hoursMatch]).filter { $0.isNumber }
            if let hours = Int(hoursString) {
                return hours * 60
            }
        }
        
        // "90 minutos" -> 90 minutes
        if let minutesMatch = lowercased.range(of: #"(\d+)\s*minuto"#, options: .regularExpression) {
            let minutesString = String(lowercased[minutesMatch]).filter { $0.isNumber }
            if let minutes = Int(minutesString) {
                return minutes
            }
        }
        
        // "1h30" or "1:30" -> 90 minutes
        if let timeMatch = lowercased.range(of: #"(\d+)[h:](\d+)"#, options: .regularExpression) {
            let timeString = String(lowercased[timeMatch])
            let components = timeString.split { $0 == "h" || $0 == ":" }
            if components.count == 2,
               let hours = Int(components[0]),
               let minutes = Int(components[1]) {
                return hours * 60 + minutes
            }
        }
        
        // Just a number -> assume minutes
        if let minutes = Int(text.filter { $0.isNumber }) {
            return minutes
        }
        
        return nil
    }
    
    private static func isCommonCombination(surgeon: String, procedure: String) -> Bool {
        // In a real implementation, this would check a database of surgeon specialties
        // For now, we'll assume all urologists can perform all urological procedures
        let surgeonEntity = MedicalKnowledgeBase.findSurgeon(by: surgeon)
        let procedureEntity = MedicalKnowledgeBase.findProcedure(by: procedure)
        
        // If both are known entities, assume it's a valid combination
        if surgeonEntity != nil && procedureEntity != nil {
            return true
        }
        
        // If either is unknown, flag for review
        return false
    }
    
    private static func combineDateAndTime(date: String, time: String) -> Date? {
        let formatter = DateFormatter()
        formatter.dateFormat = "dd/MM/yyyy HH:mm"
        formatter.locale = Locale(identifier: "pt_BR")
        return formatter.date(from: "\(date) \(time)")
    }
}
