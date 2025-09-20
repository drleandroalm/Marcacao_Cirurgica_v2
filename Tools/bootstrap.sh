#!/usr/bin/env bash
set -euo pipefail

# Tiny bootstrap to build and test the Xcode project.
# Usage:
#   Tools/bootstrap.sh [--device "iPhone 16 Pro"] [--open]

DEVICE="iPhone 16 Pro"
OPEN_PROJECT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"; shift 2;;
    --open)
      OPEN_PROJECT=true; shift;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

PROJECT="SwiftTranscriptionSampleApp.xcodeproj"
SCHEME="SwiftTranscriptionSampleApp"

if [[ ! -d "$PROJECT" ]]; then
  echo "Project not found: $PROJECT" >&2
  exit 1
fi

echo "Selecting Xcode (if needed)…"
if command -v xcode-select >/dev/null 2>&1; then
  XCODE_PATH=$(xcode-select -p 2>/dev/null || true)
  echo "xcode-select: $XCODE_PATH"
fi

if $OPEN_PROJECT; then
  echo "Opening project in Xcode…"
  open "$PROJECT"
fi

echo "Resolving packages (if any)…"
xcodebuild -resolvePackageDependencies \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  >/dev/null

echo "Building for iOS Simulator…"
mkdir -p build
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -sdk iphonesimulator \
  -configuration Debug \
  build | xcbeautify || true

echo "Running unit tests on device: $DEVICE…"
xcodebuild test \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -destination "platform=iOS Simulator,name=$DEVICE" | tee build/ci_test.log | xcbeautify || true

echo "Done. Logs at build/ci_test.log"
