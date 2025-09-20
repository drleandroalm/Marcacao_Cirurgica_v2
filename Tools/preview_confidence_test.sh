#!/usr/bin/env bash
set -euo pipefail

# Runs the Preview Confidence output + performance tests and summarizes key lines.
# Usage:
#   Tools/preview_confidence_test.sh [--device "iPhone 16 Pro"]

DEVICE="iPhone 16 Pro"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

PROJECT="SwiftTranscriptionSampleApp.xcodeproj"
SCHEME="SwiftTranscriptionSampleApp"

mkdir -p build

echo "Running tests focusing on preview confidence + perf…"
xcodebuild test \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "platform=iOS Simulator,name=$DEVICE" \
  -only-testing:SwiftTranscriptionSampleAppTests/FormPreviewConfidenceTests/test_WhenSampleTranscriptProvided_ShouldProduceRedactedConfidencePreviewOutput \
  -only-testing:SwiftTranscriptionSampleAppTests/FormPreviewConfidenceTests/test_Performance_PreviewPipeline_ShouldReportMetrics \
  | tee build/preview_perf_raw.log | xcbeautify || true

echo "\n==== Preview Confidence Summary ===="
rg -n "^\[PREVIEW\]|^\[CONF\]|^\[PERF\]" build/preview_perf_raw.log || true

echo "\nTip: Use --device to pick a specific simulator."
