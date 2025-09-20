IOS_DEVICE ?= iPhone 16 Pro

.PHONY: bootstrap build test perf clean

bootstrap:
	Tools/bootstrap.sh --device "$(IOS_DEVICE)"

build:
	xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj \
	  -scheme SwiftTranscriptionSampleApp \
	  -sdk iphonesimulator build | xcbeautify || true

test:
	xcodebuild test -project SwiftTranscriptionSampleApp.xcodeproj \
	  -scheme SwiftTranscriptionSampleApp \
	  -destination 'platform=iOS Simulator,name=$(IOS_DEVICE)' | xcbeautify || true

perf:
	Tools/preview_confidence_test.sh --device "$(IOS_DEVICE)"

clean:
	xcodebuild -project SwiftTranscriptionSampleApp.xcodeproj \
	  -scheme SwiftTranscriptionSampleApp \
	  clean >/dev/null
	rm -f build/ci_test.log build/preview_perf_raw.log || true

