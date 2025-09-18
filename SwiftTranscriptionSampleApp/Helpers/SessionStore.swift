import Foundation

@MainActor
final class SessionStore: ObservableObject {
    static let shared = SessionStore()
    @Published private(set) var sessions: [SurgerySession] = []

    private var url: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return dir.appendingPathComponent("surgery_sessions.json")
    }

    init() {
        load()
    }

    func add(_ session: SurgerySession) {
        sessions.insert(session, at: 0)
        save()
    }

    func delete(at offsets: IndexSet) {
        sessions.remove(atOffsets: offsets)
        save()
    }

    func delete(_ session: SurgerySession) {
        if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions.remove(at: idx)
            save()
        }
    }

    func clear() {
        sessions.removeAll()
        save()
    }

    // MARK: - Bulk Export

    func exportJSON(anonymize: Bool) -> URL? {
        let exportObjects: [[String: Any]] = sessions.map { s in
            if anonymize {
                return [
                    "id": s.id.uuidString,
                    "createdAt": ISO8601DateFormatter().string(from: s.createdAt),
                    "surgeonName": s.surgeonName,
                    "procedureName": s.procedureName,
                    "surgeryDate": s.surgeryDate,
                    "surgeryTime": s.surgeryTime,
                    "needsCTI": s.needsCTI as Any,
                    "needsOPME": s.needsOPME as Any,
                    "needsHemocomponents": s.needsHemocomponents as Any
                ]
            } else {
                return [
                    "id": s.id.uuidString,
                    "createdAt": ISO8601DateFormatter().string(from: s.createdAt),
                    "patientName": s.patientName,
                    "surgeonName": s.surgeonName,
                    "procedureName": s.procedureName,
                    "surgeryDate": s.surgeryDate,
                    "surgeryTime": s.surgeryTime,
                    "needsCTI": s.needsCTI as Any,
                    "needsOPME": s.needsOPME as Any,
                    "needsHemocomponents": s.needsHemocomponents as Any,
                    "hemocomponentsSpecification": s.hemocomponentsSpecification as Any,
                    "exportedTemplate": s.exportedTemplate
                ]
            }
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: exportObjects, options: .prettyPrinted)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("history_export.json")
            try data.write(to: url)
            return url
        } catch {
            return nil
        }
    }

    func exportCSV(anonymize: Bool) -> URL? {
        var lines: [String] = []
        if anonymize {
            lines.append("id,createdAt,surgeon,procedure,date,time,needsCTI,needsOPME,needsHem")
            for s in sessions {
                lines.append("\(s.id.uuidString),\(iso8601(s.createdAt)),\(csv(s.surgeonName)),\(csv(s.procedureName)),\(csv(s.surgeryDate)),\(csv(s.surgeryTime)),\(csv(flag(s.needsCTI))),\(csv(flag(s.needsOPME))),\(csv(flag(s.needsHemocomponents)))")
            }
        } else {
            lines.append("id,createdAt,patient,surgeon,procedure,date,time,needsCTI,needsOPME,needsHem")
            for s in sessions {
                lines.append("\(s.id.uuidString),\(iso8601(s.createdAt)),\(csv(s.patientName)),\(csv(s.surgeonName)),\(csv(s.procedureName)),\(csv(s.surgeryDate)),\(csv(s.surgeryTime)),\(csv(flag(s.needsCTI))),\(csv(flag(s.needsOPME))),\(csv(flag(s.needsHemocomponents)))")
            }
        }
        let csvData = lines.joined(separator: "\n").data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("history_export.csv")
        do { try csvData.write(to: url); return url } catch { return nil }
    }

    // Helpers
    private func csv(_ s: String) -> String {
        let escaped = s.replacingOccurrences(of: "\"", with: "\"\"")
        return "\"\(escaped)\""
    }
    private func flag(_ b: Bool?) -> String { b == true ? "1" : "0" }
    private func iso8601(_ d: Date) -> String { ISO8601DateFormatter().string(from: d) }

    private func load() {
        do {
            let data = try Data(contentsOf: url)
            sessions = try JSONDecoder().decode([SurgerySession].self, from: data)
        } catch {
            sessions = []
        }
    }

    private func save() {
        do {
            let data = try JSONEncoder().encode(sessions)
            try data.write(to: url)
        } catch {
            // Avoid logging PHI; no-op on error
        }
    }
}
