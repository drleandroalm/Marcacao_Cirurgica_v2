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
            .preferredColorScheme(.dark)
    }
}

struct MainTabs: View {
    @StateObject private var sessionStore = SessionStore.shared
    @State private var selectedTab = 0

    var body: some View {
        ZStack {
            // Dark background
            DarkTheme.background
                .ignoresSafeArea()

            VStack(spacing: 0) {
                // Content
                Group {
                    if selectedTab == 0 {
                        FormFillerView()
                            .environmentObject(sessionStore)
                    } else {
                        HistoryView()
                            .environmentObject(sessionStore)
                    }
                }

                // Custom Tab Bar
                DarkTabBar(
                    selectedTab: $selectedTab,
                    tabs: [
                        ("Formulário", "square.and.pencil"),
                        ("Histórico", "clock.fill")
                    ]
                )
            }
        }
        .onAppear {
            // Customize appearance
            UITabBar.appearance().backgroundColor = UIColor.black
            UITabBar.appearance().tintColor = UIColor.cyan
            UINavigationBar.appearance().backgroundColor = UIColor.black
            UINavigationBar.appearance().tintColor = UIColor.cyan
        }
    }
}
