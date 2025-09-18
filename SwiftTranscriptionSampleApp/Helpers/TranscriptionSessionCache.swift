import Foundation
import AVFoundation

actor TranscriptionSessionCache {
    static let shared = TranscriptionSessionCache()
    
    private var localeRetainCount: [String: Int] = [:]
    private var analyzerFormats: [String: AVAudioFormat] = [:]
    
    func retain(locale: Locale) -> Bool {
        let key = cacheKey(for: locale)
        let currentCount = localeRetainCount[key, default: 0]
        localeRetainCount[key] = currentCount + 1
        return currentCount == 0
    }
    
    func release(locale: Locale) -> Bool {
        let key = cacheKey(for: locale)
        guard let currentCount = localeRetainCount[key] else { return false }
        let newCount = max(0, currentCount - 1)
        if newCount == 0 {
            localeRetainCount.removeValue(forKey: key)
            analyzerFormats.removeValue(forKey: key)
            return true
        } else {
            localeRetainCount[key] = newCount
            return false
        }
    }
    
    func cachedFormat(for locale: Locale) -> AVAudioFormat? {
        analyzerFormats[cacheKey(for: locale)]
    }
    
    func cache(format: AVAudioFormat, for locale: Locale) {
        analyzerFormats[cacheKey(for: locale)] = format
    }
    
    private func cacheKey(for locale: Locale) -> String {
        locale.identifier(.bcp47)
    }
}
