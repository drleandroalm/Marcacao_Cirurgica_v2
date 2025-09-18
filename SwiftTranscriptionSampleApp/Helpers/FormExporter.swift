import Foundation
import UIKit

@MainActor
class FormExporter {
    
    static func copyToClipboard(form: SurgicalRequestForm) -> Bool {
        let filledTemplate = form.generateFilledTemplate()
        UIPasteboard.general.string = filledTemplate
        return true
    }
    
    static func shareForm(form: SurgicalRequestForm, from viewController: UIViewController) {
        let text = form.generateFilledTemplate()
        let activityVC = UIActivityViewController(activityItems: [text], applicationActivities: nil)
        
        if let popoverController = activityVC.popoverPresentationController {
            popoverController.sourceView = viewController.view
            popoverController.sourceRect = CGRect(x: viewController.view.bounds.midX, y: viewController.view.bounds.midY, width: 0, height: 0)
            popoverController.permittedArrowDirections = []
        }
        
        viewController.present(activityVC, animated: true)
    }
    
    static func saveToDocuments(form: SurgicalRequestForm) -> URL? {
        let text = form.generateFilledTemplate()
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = dateFormatter.string(from: Date())
        let fileName = "solicitacao_cirurgica_\(timestamp).txt"
        
        guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
            return nil
        }
        
        let fileURL = documentsDirectory.appendingPathComponent(fileName)
        
        do {
            try text.write(to: fileURL, atomically: true, encoding: .utf8)
            return fileURL
        } catch {
            print("Failed to save file: \(error)")
            return nil
        }
    }
    
    static func exportAsJSON(form: SurgicalRequestForm) -> Data? {
        var jsonDict: [String: Any] = [:]
        
        for field in form.fields {
            jsonDict[field.id] = field.value.isEmpty ? nil : field.value
        }
        
        jsonDict["exportDate"] = ISO8601DateFormatter().string(from: Date())
        jsonDict["formType"] = "SOLICITAÇÃO DE AGENDAMENTO CIRÚRGICO"
        
        do {
            let jsonData = try JSONSerialization.data(withJSONObject: jsonDict, options: .prettyPrinted)
            return jsonData
        } catch {
            print("Failed to create JSON: \(error)")
            return nil
        }
    }
}
