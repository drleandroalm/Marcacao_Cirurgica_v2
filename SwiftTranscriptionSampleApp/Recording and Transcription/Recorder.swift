/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Audio input code
*/

import Foundation
import AVFoundation
import SwiftUI
import os

/// Streams microphone audio into the speech pipeline while optionally persisting a waveform for playback or debugging.
final class Recorder: @unchecked Sendable {
    private var outputContinuation: AsyncStream<AudioData>.Continuation? = nil
    private let audioEngine: AVAudioEngine
    private let transcriber: SpokenWordTranscriber
    private let configuration: RecordingConfiguration
    private var audioPlayer: AVAudioPlayer?
    private static let metricsLog = OSLog(subsystem: "SwiftTranscriptionSampleApp", category: "Recorder")
    
    var file: AVAudioFile?
    private let url: URL

    init(transcriber: SpokenWordTranscriber, configuration: RecordingConfiguration = RecordingConfiguration()) {
        audioEngine = AVAudioEngine()
        self.transcriber = transcriber
        self.configuration = configuration
        self.url = FileManager.default.temporaryDirectory
            .appending(component: UUID().uuidString)
            .appendingPathExtension(for: .wav)
    }
    
    private func isAuthorized() async -> Bool {
    #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        switch session.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            return await withCheckedContinuation { continuation in
                session.requestRecordPermission { granted in
                    continuation.resume(returning: granted)
                }
            }
        @unknown default:
            return false
        }
    #else
        return true
    #endif
    }
    
    func record() async throws {
        let signpostID = OSSignpostID(log: Self.metricsLog)
        os_signpost(.begin, log: Self.metricsLog, name: "record()", signpostID: signpostID)
        defer { os_signpost(.end, log: Self.metricsLog, name: "record()", signpostID: signpostID) }

        guard await isAuthorized() else {
            print("user denied mic permission")
            return
        }
#if os(iOS)
        try setUpAudioSession()
#endif
        try await transcriber.setUpTranscriber()
        
        // Capture a stable reference to avoid capturing task-isolated self later.
        let transcriber = self.transcriber
                
        for await input in try await audioStream() {
            do {
                // If the method is MainActor-isolated, the await will hop to the main actor automatically.
                try await transcriber.streamAudioToTranscriber(input.buffer)
            } catch {
                print("streamAudioToTranscriber error: \(error)")
            }
        }
    }
    
    func stopRecording() async throws {
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()
        outputContinuation?.finish()
        outputContinuation = nil
        try await transcriber.finishTranscribing()
        await transcriber.finishContinuousTranscription()
    #if os(iOS)
        do {
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to deactivate audio session: \(error)")
        }
    #endif
    }
    
    func pauseRecording() {
        audioEngine.pause()
    }
    
    func resumeRecording() throws {
        try audioEngine.start()
    }
#if os(iOS)
    func setUpAudioSession() throws {
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .spokenAudio)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
    }
#endif

    private func writeBufferToDisk(buffer: AVAudioPCMBuffer) {
        guard let file = self.file else { return }
        do {
            try file.write(from: buffer)
        } catch {
            print("Failed to write buffer to disk: \(error)")
        }
    }
    
    private func audioStream() async throws -> AsyncStream<AudioData> {
        try setupAudioEngine()
        audioEngine.inputNode.installTap(onBus: 0,
                                         bufferSize: configuration.bufferSize,
                                         format: audioEngine.inputNode.outputFormat(forBus: 0)) { [weak self] (buffer, time) in
            guard let self else { return }
            writeBufferToDisk(buffer: buffer)
            // Wrap in AudioData which is @unchecked Sendable.
            let audioData = AudioData(buffer: buffer, time: time)
            self.outputContinuation?.yield(audioData)
        }
        
        audioEngine.prepare()
        if !audioEngine.isRunning {
            try audioEngine.start()
        }
        
        return AsyncStream(AudioData.self, bufferingPolicy: .bufferingNewest(configuration.streamBacklogDepth)) { continuation in
            outputContinuation = continuation
        }
    }
    
    private func setupAudioEngine() throws {
        let inputSettings = audioEngine.inputNode.inputFormat(forBus: 0).settings
        if configuration.shouldWriteToDisk {
            self.file = try AVAudioFile(forWriting: url, settings: inputSettings)
        } else {
            self.file = nil
        }
        audioEngine.inputNode.removeTap(onBus: 0)
    }
        
    func playRecording() {
        guard let url = file?.url else { return }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
        } catch {
            print("AVAudioPlayer error: \(error)")
        }
    }
    
    func stopPlaying() {
        audioPlayer?.stop()
        audioPlayer = nil
    }
}
