/// apple-vision-ocr-helper
///
/// A thin CLI wrapper around Apple Vision's VNRecognizeTextRequest.
/// Reads a PNG/JPEG image from stdin, runs on-device OCR, and writes
/// a UTF-8 JSON result to stdout.
///
/// Usage:
///   echo -n <image-bytes> | apple-vision-ocr-helper [--lang ko] [--lang ja]
///
/// Output JSON schema:
///   { "text": "...", "error": null }
///   { "text": null,  "error": "..." }

import Foundation
import Vision

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

var requestedLanguages: [String] = []
var args = CommandLine.arguments.dropFirst()
while let arg = args.first {
    if arg == "--lang", let lang = args.dropFirst().first {
        requestedLanguages.append(lang)
        args = args.dropFirst(2)
    } else {
        args = args.dropFirst()
    }
}

// ---------------------------------------------------------------------------
// Read stdin
// ---------------------------------------------------------------------------

var imageData = Data()
let bufferSize = 65536
var buffer = [UInt8](repeating: 0, count: bufferSize)
while true {
    let count = read(STDIN_FILENO, &buffer, bufferSize)
    if count <= 0 { break }
    imageData.append(contentsOf: buffer.prefix(count))
}

// ---------------------------------------------------------------------------
// Vision OCR
// ---------------------------------------------------------------------------

struct Output: Codable {
    let text: String?
    let error: String?
}

func emit(_ output: Output) -> Never {
    let encoder = JSONEncoder()
    let data = (try? encoder.encode(output)) ?? Data("{\"text\":null,\"error\":\"encode failed\"}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(0)
}

guard !imageData.isEmpty else {
    emit(Output(text: nil, error: "no image data received on stdin"))
}

guard let cgImageSource = CGImageSourceCreateWithData(imageData as CFData, nil),
      let cgImage = CGImageSourceCreateImageAtIndex(cgImageSource, 0, nil) else {
    emit(Output(text: nil, error: "failed to decode image from stdin"))
}

var recognizedStrings: [String] = []
var visionError: String? = nil

let semaphore = DispatchSemaphore(value: 0)
let request = VNRecognizeTextRequest { request, error in
    defer { semaphore.signal() }
    if let error {
        visionError = error.localizedDescription
        return
    }
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    recognizedStrings = observations.compactMap { $0.topCandidates(1).first?.string }
}

request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

if !requestedLanguages.isEmpty {
    request.recognitionLanguages = requestedLanguages
} else {
    // Default: prefer Korean and Japanese if supported, fall back to all
    let supported = (try? VNRecognizeTextRequest.supportedRecognitionLanguages(
        for: .accurate,
        revision: VNRecognizeTextRequestRevision3
    )) ?? (try? VNRecognizeTextRequest().supportedRecognitionLanguages()) ?? []
    let preferred = ["ko-KR", "ja-JP", "zh-Hans", "zh-Hant", "en-US"]
    request.recognitionLanguages = preferred.filter { supported.contains($0) }
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    visionError = error.localizedDescription
}
semaphore.wait()

if let err = visionError {
    emit(Output(text: nil, error: err))
}

let combined = recognizedStrings.joined(separator: "\n")
emit(Output(text: combined, error: nil))
