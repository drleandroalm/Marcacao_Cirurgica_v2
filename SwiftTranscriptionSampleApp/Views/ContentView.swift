/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The app's main view.
*/

import SwiftUI
import Speech

struct ContentView: View {
    var body: some View {
        MainTabs()
    }
}

struct MainTabs: View {
    @StateObject private var sessionStore = SessionStore.shared

    var body: some View {
        TabView {
            FormFillerView()
                .tabItem {
                    Label("Formulário", systemImage: "square.and.pencil")
                }
                .environmentObject(sessionStore)

            HistoryView()
                .tabItem {
                    Label("Histórico", systemImage: "clock.fill")
                }
                .environmentObject(sessionStore)
        }
    }
}
