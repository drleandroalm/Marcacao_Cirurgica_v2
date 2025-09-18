import Foundation
import AVFoundation

// Lightweight smoke test for converter timing and conversion

final class SmokeBufferConverter {
    enum Err: Swift.Error { case failedToCreateConverter, failedToCreateBuffer }
    private var converter: AVAudioConverter?
    private var currentTime: TimeInterval = 0
    func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format
        if inputFormat == format { return buffer }
        if converter == nil || converter?.outputFormat != format || converter?.inputFormat != inputFormat {
            converter = AVAudioConverter(from: inputFormat, to: format)
            converter?.primeMethod = .none
        }
        guard let converter else { throw Err.failedToCreateConverter }
        let ratio = converter.outputFormat.sampleRate / converter.inputFormat.sampleRate
        let frames = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up))
        guard let out = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: frames) else { throw Err.failedToCreateBuffer }
        var err: NSError?
        var fed = false
        let status = converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return buffer
        }
        if status == .error { throw err ?? Err.failedToCreateConverter }
        return out
    }
    func calculateTimeRange(for buffer: AVAudioPCMBuffer) -> CMTimeRange {
        let duration = Double(buffer.frameLength) / buffer.format.sampleRate
        let start = CMTime(seconds: currentTime, preferredTimescale: 1_000_000)
        let dur = CMTime(seconds: duration, preferredTimescale: 1_000_000)
        currentTime += duration
        return CMTimeRange(start: start, duration: dur)
    }
}

final class Smoke {
    let converter = SmokeBufferConverter()

    func run() throws {
        print("🔎 SmokeTest: BufferConverter timing")

        // Target format (what analyzer would want)
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false) else {
            fatalError("Failed to create target format")
        }

        // Source 1: 44.1kHz stereo, ~0.1s
        guard let src1 = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100, channels: 2, interleaved: false),
              let buf1 = AVAudioPCMBuffer(pcmFormat: src1, frameCapacity: 4_410) else { fatalError("Failed to create src1") }
        buf1.frameLength = 4_410

        let conv1 = try converter.convertBuffer(buf1, to: target)
        let tr1 = converter.calculateTimeRange(for: conv1)
        print(String(format: "✅ Buf1 start=%.3fs dur=%.3fs frames=%u", CMTimeGetSeconds(tr1.start), CMTimeGetSeconds(tr1.duration), conv1.frameLength))

        // Source 2: 48kHz mono, ~0.2s
        guard let src2 = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 1, interleaved: false),
              let buf2 = AVAudioPCMBuffer(pcmFormat: src2, frameCapacity: 9_600) else { fatalError("Failed to create src2") }
        buf2.frameLength = 9_600

        let conv2 = try converter.convertBuffer(buf2, to: target)
        let tr2 = converter.calculateTimeRange(for: conv2)
        print(String(format: "✅ Buf2 start=%.3fs dur=%.3fs frames=%u", CMTimeGetSeconds(tr2.start), CMTimeGetSeconds(tr2.duration), conv2.frameLength))

        // Zero-length buffer
        guard let zero = AVAudioPCMBuffer(pcmFormat: src2, frameCapacity: 1) else { fatalError("Failed to create zero buffer") }
        zero.frameLength = 0
        let convZero = try converter.convertBuffer(zero, to: target)
        if convZero.frameLength == 0 {
            print("⏭️ Zero-length buffer remains 0 frames after conversion (expected)")
        }
        let trZero = converter.calculateTimeRange(for: convZero)
        print(String(format: "ℹ️ Zero buf start=%.3fs dur=%.3fs frames=%u", CMTimeGetSeconds(trZero.start), CMTimeGetSeconds(trZero.duration), convZero.frameLength))

        print("🎉 SmokeTest complete")
    }
}

try Smoke().run()
