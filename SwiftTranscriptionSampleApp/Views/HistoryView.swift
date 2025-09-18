import SwiftUI

struct HistoryView: View {
    @EnvironmentObject private var sessionStore: SessionStore
    @State private var searchText: String = ""
    @AppStorage("history.search") private var persistedSearch: String = ""
    @AppStorage("history.filterSurgeon") private var persistedFilterSurgeon: String = ""
    @AppStorage("history.filterProcedure") private var persistedFilterProcedure: String = ""
    @State private var filterSurgeon: String = ""
    @State private var filterProcedure: String = ""
    @State private var isEditing: Bool = false
    @State private var showClearAllConfirm: Bool = false
    @State private var showExportSheet: Bool = false
    @State private var exportAnonymize: Bool = true
    @State private var exportFormat: ExportFormat = .json

    enum ExportFormat: String, CaseIterable, Identifiable { case json, csv; var id: String { rawValue } }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                filtersBar
                listContent
            }
            .navigationTitle("Histórico")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Limpar Tudo") { showClearAllConfirm = true }
                        .disabled(sessionStore.sessions.isEmpty)
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 12) {
                        Button {
                            showExportSheet = true
                        } label: {
                            Image(systemName: "square.and.arrow.up")
                        }
                        Button(isEditing ? "OK" : "Editar") { isEditing.toggle() }
                    }
                }
            }
            .alert("Apagar todo o histórico?", isPresented: $showClearAllConfirm) {
                Button("Cancelar", role: .cancel) {}
                Button("Apagar", role: .destructive) { sessionStore.clear() }
            } message: {
                Text("Esta ação não pode ser desfeita.")
            }
            .sheet(isPresented: $showExportSheet) {
                ExportSheet(isPresented: $showExportSheet,
                            anonymize: $exportAnonymize,
                            format: $exportFormat,
                            onExport: { format, anonymize in
                    if let url = (format == .json ? sessionStore.exportJSON(anonymize: anonymize)
                                                  : sessionStore.exportCSV(anonymize: anonymize)) {
                        let av = UIActivityViewController(activityItems: [url], applicationActivities: nil)
                        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                           let root = scene.windows.first?.rootViewController {
                            root.present(av, animated: true)
                        }
                    }
                })
            }
        }
        .onAppear {
            searchText = persistedSearch
            filterSurgeon = persistedFilterSurgeon
            filterProcedure = persistedFilterProcedure
        }
        .onChange(of: searchText) { _, v in persistedSearch = v }
        .onChange(of: filterSurgeon) { _, v in persistedFilterSurgeon = v }
        .onChange(of: filterProcedure) { _, v in persistedFilterProcedure = v }
    }

    private var filtersBar: some View {
        VStack(spacing: 8) {
            // Search
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundColor(.secondary)
                TextField("Buscar por Paciente, Cirurgião, Procedimento", text: $searchText)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled(true)
                if !searchText.isEmpty {
                    Button(action: { searchText = "" }) {
                        Image(systemName: "xmark.circle.fill").foregroundColor(.secondary)
                    }
                }
            }
            .padding(10)
            .background(Color(.systemGray6))
            .cornerRadius(10)

            // Toggles / pickers for surgeon and procedure
            HStack(spacing: 8) {
                Menu {
                    Button("Todos") { filterSurgeon = "" }
                    ForEach(Array(uniqueSurgeons.sorted()), id: \.self) { name in
                        Button(name) { filterSurgeon = name }
                    }
                } label: {
                    labelChip(title: filterSurgeon.isEmpty ? "Cirurgião: Todos" : filterSurgeon,
                              systemImage: "person.text.rectangle")
                }

                Menu {
                    Button("Todos") { filterProcedure = "" }
                    ForEach(Array(uniqueProcedures.sorted()), id: \.self) { name in
                        Button(name) { filterProcedure = name }
                    }
                } label: {
                    labelChip(title: filterProcedure.isEmpty ? "Procedimento: Todos" : filterProcedure,
                              systemImage: "stethoscope")
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
    }

    private func labelChip(title: String, systemImage: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: systemImage)
            Text(title).lineLimit(1)
        }
        .font(.caption)
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(.systemGray6))
        .cornerRadius(16)
    }

    private var listContent: some View {
        List {
            ForEach(groupedSessions.keys.sorted(by: >), id: \.self) { day in
                if let items = groupedSessions[day] {
                    Section(header: Text(sectionTitle(for: day))) {
                        ForEach(items) { session in
                            NavigationLink(destination: SessionDetailView(session: session)) {
                                SessionRow(session: session)
                            }
                        }
                        .onDelete { offsets in
                            let toDelete = offsets.map { items[$0] }
                            toDelete.forEach { sessionStore.delete($0) }
                        }
                    }
                }
            }
        }
        .environment(\.editMode, isEditing ? .constant(EditMode.active) : .constant(EditMode.inactive))
        .listStyle(InsetGroupedListStyle())
    }

    private var filtered: [SurgerySession] {
        sessionStore.sessions.filter { session in
            // Text query across patient, surgeon, procedure
            let q = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let matchesQuery = q.isEmpty || [session.patientName, session.surgeonName, session.procedureName]
                .map { $0.lowercased() }.contains { $0.contains(q) }
            // Pickers
            let matchesSurgeon = filterSurgeon.isEmpty || session.surgeonName == filterSurgeon
            let matchesProcedure = filterProcedure.isEmpty || session.procedureName == filterProcedure
            return matchesQuery && matchesSurgeon && matchesProcedure
        }
    }

    private var groupedSessions: [Date: [SurgerySession]] {
        let calendar = Calendar.current
        let groups = Dictionary(grouping: filtered) { session in
            calendar.startOfDay(for: session.createdAt)
        }
        // Sort each group by time desc
        var sortedGroups: [Date: [SurgerySession]] = [:]
        for (k, v) in groups {
            sortedGroups[k] = v.sorted { $0.createdAt > $1.createdAt }
        }
        return sortedGroups
    }

    private var uniqueSurgeons: Set<String> {
        Set(sessionStore.sessions.map { $0.surgeonName }.filter { !$0.isEmpty })
    }

    private var uniqueProcedures: Set<String> {
        Set(sessionStore.sessions.map { $0.procedureName }.filter { !$0.isEmpty })
    }

    private func sectionTitle(for day: Date) -> String {
        let cal = Calendar.current
        if cal.isDateInToday(day) { return "Hoje" }
        if cal.isDateInYesterday(day) { return "Ontem" }
        let df = DateFormatter(); df.dateStyle = .medium; df.timeStyle = .none
        return df.string(from: day)
    }
}

struct SessionRow: View {
    let session: SurgerySession
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(session.patientName.isEmpty ? "[Paciente]" : session.patientName)
                    .font(.headline)
                Spacer()
                Text("\(session.surgeryDate) • \(session.surgeryTime)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(session.procedureName.isEmpty ? "[Procedimento]" : session.procedureName)
                    .font(.subheadline)
                    .foregroundColor(.primary)
            }
            HStack(spacing: 8) {
                Text(session.surgeonName.isEmpty ? "[Cirurgião]" : session.surgeonName)
                    .font(.footnote)
                    .foregroundColor(.secondary)
                Spacer()
                // Compact flags
                if session.needsCTI == true {
                    flagChip(system: "bed.double.fill", text: "CTI", color: .orange)
                }
                if session.needsOPME == true {
                    flagChip(system: "briefcase.fill", text: "OPME", color: .blue)
                }
                if session.needsHemocomponents == true {
                    flagChip(system: "drop.fill", text: "Hem", color: .red)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func flagChip(system: String, text: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: system)
            Text(text)
        }
        .font(.caption2)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(color.opacity(0.12))
        .foregroundColor(color)
        .cornerRadius(8)
    }
}

struct SessionDetailView: View {
    let session: SurgerySession
    @EnvironmentObject private var store: SessionStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 16) {
            ScrollView {
                Text(session.exportedTemplate)
                    .font(.system(.body, design: .monospaced))
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    .padding()
            }
            HStack(spacing: 12) {
                Button(role: .destructive) {
                    store.delete(session)
                    dismiss()
                } label: {
                    Label("Apagar", systemImage: "trash")
                }

                Spacer()

                Button {
                    UIPasteboard.general.string = session.exportedTemplate
                } label: {
                    Label("Copiar", systemImage: "doc.on.doc")
                }
                Button {
                    let av = UIActivityViewController(activityItems: [session.exportedTemplate], applicationActivities: nil)
                    if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let root = scene.windows.first?.rootViewController {
                        root.present(av, animated: true)
                    }
                } label: {
                    Label("Compartilhar", systemImage: "square.and.arrow.up")
                }
                if let json = session.exportedJSON {
                    Button {
                        let av = UIActivityViewController(activityItems: [json], applicationActivities: nil)
                        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                           let root = scene.windows.first?.rootViewController {
                            root.present(av, animated: true)
                        }
                    } label: {
                        Label("Compartilhar JSON", systemImage: "curlybraces.square")
                    }
                }
            }
            .padding(.horizontal)
            .padding(.bottom)
        }
        .navigationTitle("Sessão")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ExportSheet: View {
    @Binding var isPresented: Bool
    @Binding var anonymize: Bool
    @Binding var format: HistoryView.ExportFormat
    var onExport: (HistoryView.ExportFormat, Bool) -> Void

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Formato")) {
                    Picker("Formato", selection: $format) {
                        Text("JSON").tag(HistoryView.ExportFormat.json)
                        Text("CSV").tag(HistoryView.ExportFormat.csv)
                    }.pickerStyle(.segmented)
                }
                Section(header: Text("Opções")) {
                    Toggle("Anonimizar (remove paciente)", isOn: $anonymize)
                }
            }
            .navigationTitle("Exportar Histórico")
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancelar") { isPresented = false }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Exportar") {
                        onExport(format, anonymize)
                        isPresented = false
                    }
                }
            }
        }
    }
}
