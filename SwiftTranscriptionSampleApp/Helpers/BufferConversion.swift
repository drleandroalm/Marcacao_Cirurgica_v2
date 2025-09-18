/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Conversion code for audio inputs.
*/

import Foundation
@preconcurrency import AVFoundation
import Speech
import CoreMedia

final class BufferConverter: @unchecked Sendable {
    enum Error: Swift.Error {
        case failedToCreateConverter
        case failedToCreateConversionBuffer
        case conversionFailed(NSError?)
    }
    
    private var converter: AVAudioConverter?
    private var currentTime: TimeInterval = 0
    private let startTime = Date()
    func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format
        guard inputFormat != format else {
            return buffer
        }
        
        if converter == nil || converter?.outputFormat != format {
            converter = AVAudioConverter(from: inputFormat, to: format)
            converter?.primeMethod = .none // Sacrifice quality of first samples in order to avoid any timestamp drift from source
        }
        
        guard let converter else {
            throw Error.failedToCreateConverter
        }
        
        let sampleRateRatio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let scaledInputFrameLength = Double(buffer.frameLength) * sampleRateRatio
        let frameCapacity = AVAudioFrameCount(scaledInputFrameLength.rounded(.up))
        guard let conversionBuffer = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: frameCapacity) else {
            throw Error.failedToCreateConversionBuffer
        }
        
        var nsError: NSError?
        
        // Use a pointer-backed flag to avoid "Mutation of captured var in concurrently-executing code"
        let processedPtr = UnsafeMutablePointer<Bool>.allocate(capacity: 1)
        processedPtr.initialize(to: false)
        defer {
            processedPtr.deinitialize(count: 1)
            processedPtr.deallocate()
        }
        
        nonisolated(unsafe) let unsafeProcessedPtr = processedPtr
        
        let status = converter.convert(to: conversionBuffer, error: &nsError) { _, inputStatusPointer in
            if unsafeProcessedPtr.pointee {
                inputStatusPointer.pointee = .noDataNow
                return nil
            } else {
                unsafeProcessedPtr.pointee = true
                inputStatusPointer.pointee = .haveData
                return buffer
            }
        }
        
        guard status != .error else {
            throw Error.conversionFailed(nsError)
        }
        
        return conversionBuffer
    }
    
    func calculateTimeRange(for buffer: AVAudioPCMBuffer) -> CMTimeRange {
        let duration = Double(buffer.frameLength) / buffer.format.sampleRate
        let start = CMTime(seconds: currentTime, preferredTimescale: 1000000)
        let durationTime = CMTime(seconds: duration, preferredTimescale: 1000000)
        
        currentTime += duration
        
        return CMTimeRange(start: start, duration: durationTime)
    }
    
    func reset() {
        currentTime = 0
    }
}
