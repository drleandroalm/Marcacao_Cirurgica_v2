# Entity Recognition Implementation Guide
## Quick Start for Marcação Cirúrgica Project

**Date:** 2025-11-13

---

## TL;DR - What You Need to Know

**Best Approach for This Project:** Hybrid multi-stage pipeline combining:
1. **Rule-Based** (fast, high precision for structured data)
2. **Gazetteer** (medical terminology dictionary)
3. **Apple NaturalLanguage** (person/place names)
4. **Custom CoreML Model** (Portuguese medical entities)

**Expected Performance:**
- Precision: 85-90%
- Recall: 90-95%
- Latency: <100ms per transcript

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│            Transcribed Text (Portuguese)                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Rule-Based Extraction (High Confidence)       │
│  - Dates/Times: Regex patterns                          │
│  - Patient IDs: Structured patterns                     │
│  - Common phrases: "cirurgia de", "Dr.", etc.          │
│  Output: ~40% of entities (confidence > 0.95)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Gazetteer Matching (Medical Terms)           │
│  - Surgical procedures dictionary                       │
│  - Medical equipment list                               │
│  - Medication names                                     │
│  Output: ~30% of entities (confidence > 0.90)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Apple NaturalLanguage (General Entities)     │
│  - Person names (doctors, patients)                     │
│  - Place names (rooms, departments)                     │
│  Output: ~15% of entities (confidence > 0.70)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Custom CoreML (Edge Cases)                   │
│  - Complex medical entities                             │
│  - Context-dependent classification                     │
│  Output: ~15% of entities (confidence > 0.75)           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: Ensemble Voting & Deduplication              │
│  - Merge overlapping entities                           │
│  - Weighted voting by confidence                        │
│  - Resolve conflicts                                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 6: Validation Layer                             │
│  - Database cross-reference                             │
│  - Required fields check                                │
│  - Date/time feasibility                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STAGE 7: Confidence-Based UI                          │
│  - High confidence (>0.9): Auto-fill                    │
│  - Medium confidence (0.6-0.9): Show with highlight     │
│  - Low confidence (<0.6): Leave blank, suggest          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
**Goal:** Build rule-based core

**Tasks:**
1. Create `SurgicalRuleEngine.swift`
   ```swift
   class SurgicalRuleEngine {
       func extractDates(from text: String) -> [Entity]
       func extractTimes(from text: String) -> [Entity]
       func extractPatientIDs(from text: String) -> [Entity]
       func extractDoctorNames(from text: String) -> [Entity]
   }
   ```

2. Build regex patterns for Portuguese medical text:
   - Dates: "20 de novembro", "20/11/2025", "vinte de novembro"
   - Times: "14:30", "14h30", "duas da tarde"
   - Procedures: "cirurgia de [X]", "procedimento [X]"

3. Add validation rules:
   - Dates must be future (for scheduling)
   - Times must be during hospital hours
   - Required fields: procedure, doctor, patient, date

**Deliverable:** Working rule-based extractor with test cases

---

### Phase 2: Gazetteer Integration (Week 2-3)
**Goal:** Add medical terminology matching

**Tasks:**
1. Create `MedicalGazetteer.swift`
   ```swift
   class MedicalGazetteer {
       private var procedures: Set<String>
       private var equipment: Set<String>
       private var medications: Set<String>

       func match(in text: String) -> [Entity]
       func fuzzyMatch(term: String, threshold: Double) -> String?
   }
   ```

2. Build terminology lists:
   - Surgical procedures (Portuguese): "apendicectomia", "colecistectomia", "herniorrafia"
   - Medical equipment: "bisturi elétrico", "monitor cardíaco", "anestesia geral"
   - Common medications: "propofol", "fentanil", "midazolam"

3. Sources for terminology:
   - Hospital's procedure catalog
   - Medical terminology databases (DeCS/MeSH Portuguese)
   - Existing scheduling system exports

4. Implement fuzzy matching for typos/variations:
   - Levenshtein distance < 2
   - Phonetic matching (Metaphone for Portuguese)

**Deliverable:** Gazetteer with 500+ terms, fuzzy matching

---

### Phase 3: NaturalLanguage Integration (Week 3-4)
**Goal:** Leverage Apple's framework for general entities

**Tasks:**
1. Create `NaturalLanguageExtractor.swift`
   ```swift
   class NaturalLanguageExtractor {
       private let tagger = NLTagger(tagSchemes: [.nameType])

       func extractPersonNames(from text: String) -> [Entity]
       func extractPlaceNames(from text: String) -> [Entity]
   }
   ```

2. Map NaturalLanguage entity types to medical domain:
   - `NLTag.personalName` → EntityType.doctor or .patient (context-dependent)
   - `NLTag.placeName` → EntityType.location (operating room, department)

3. Handle Portuguese language nuances:
   - "Dr. Silva" vs "Dra. Silva" (gender)
   - "Sala 3" vs "Centro Cirúrgico 2" (location variations)

**Deliverable:** NaturalLanguage integration with domain mapping

---

### Phase 4: CoreML Custom Model (Week 4-6)
**Goal:** Train custom model for Portuguese medical entities

#### Step 4.1: Data Collection
```python
# Annotation format (JSON)
{
  "text": "Paciente Maria Silva necessita cirurgia de apendicectomia com Dr. João Santos no dia 20 de novembro.",
  "entities": [
    {"start": 9, "end": 20, "label": "PATIENT"},
    {"start": 32, "end": 60, "label": "PROCEDURE"},
    {"start": 64, "end": 80, "label": "DOCTOR"},
    {"start": 87, "end": 103, "label": "DATE"}
  ]
}
```

**Data Sources:**
- Anonymized past transcripts (with consent)
- Synthetic data generation (templates + variations)
- Crowdsourced annotation (medical staff)

**Minimum Data:** 500 annotated examples
**Target Data:** 2000+ annotated examples

#### Step 4.2: Model Training (Python)
```python
import spacy
from spacy.training import Example
import json

# Load base Portuguese model
nlp = spacy.load("pt_core_news_sm")

# Add custom entity recognizer
ner = nlp.get_pipe("ner")
ner.add_label("PROCEDURE")
ner.add_label("PATIENT")
ner.add_label("DOCTOR")
ner.add_label("DATE")
ner.add_label("EQUIPMENT")

# Train on your data
def train_model(train_data, n_iter=30):
    for i in range(n_iter):
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses)
        print(f"Iteration {i}, Loss: {losses['ner']}")

    return nlp

# Save model
nlp.to_disk("./surgical_ner_model")
```

#### Step 4.3: CoreML Conversion
```python
import coremltools as ct

# Convert spaCy model to CoreML (requires custom converter)
# Alternative: Train with CreateML text classifier

# For entity classification (after span extraction):
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Train classifier on entity features
X = extract_features(entities)  # word embeddings, context, etc.
y = entity_labels

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Convert to CoreML
coreml_model = ct.converters.sklearn.convert(
    clf,
    input_features=['feature_1', 'feature_2', ...],
    output_feature_names='entity_type'
)

coreml_model.save('SurgicalEntityClassifier.mlmodel')
```

#### Step 4.4: Swift Integration
```swift
import CoreML

class CoreMLEntityRecognizer {
    private let model: SurgicalEntityClassifier

    init() throws {
        self.model = try SurgicalEntityClassifier(configuration: MLModelConfiguration())
    }

    func classify(entity: String, context: String) async throws -> (type: EntityType, confidence: Double) {
        let prediction = try model.prediction(text: entity, context: context)
        return (
            type: EntityType(rawValue: prediction.label) ?? .unknown,
            confidence: prediction.labelProbabilities[prediction.label] ?? 0.0
        )
    }
}
```

**Deliverable:** Trained CoreML model with >80% F1 score

---

### Phase 5: Ensemble & Validation (Week 7-8)
**Goal:** Combine all models and add validation

**Tasks:**
1. Implement ensemble voting:
   ```swift
   class EnsembleNER {
       func mergeEntities(_ entities: [[Entity]]) -> [Entity] {
           // 1. Group overlapping entities
           // 2. Weighted voting (weights from validation performance)
           // 3. Resolve conflicts
       }
   }
   ```

2. Weight assignment (from validation data):
   ```swift
   let weights = [
       .ruleBased: 0.25,      // High precision, limited coverage
       .gazetteer: 0.30,      // Medical terms, very reliable
       .naturalLanguage: 0.15, // General entities, moderate accuracy
       .coreML: 0.30          // Custom model, best for edge cases
   ]
   ```

3. Validation layer:
   ```swift
   class SchedulingValidator {
       func validate(_ entities: [Entity]) -> ValidationResult {
           // Check required fields
           // Verify date is in future
           // Cross-reference with database
           // Flag impossibilities (doctor availability, room conflicts)
       }
   }
   ```

**Deliverable:** Complete hybrid NER system with validation

---

### Phase 6: UI Integration (Week 8-9)
**Goal:** User-friendly confidence-based interface

**SwiftUI Implementation:**
```swift
struct TranscriptEntityView: View {
    let entity: Entity

    var body: some View {
        HStack {
            Image(systemName: icon(for: entity.type))
                .foregroundColor(color(for: entity.confidence))

            Text(entity.text)
                .padding(8)
                .background(background(for: entity.confidence))
                .cornerRadius(4)

            if entity.confidence < 0.9 {
                Button("Confirmar") {
                    // User confirms entity
                }
                Button("Editar") {
                    // User corrects entity
                }
            }
        }
    }

    func color(for confidence: Double) -> Color {
        if confidence > 0.9 { return .green }
        if confidence > 0.7 { return .orange }
        return .red
    }

    func background(for confidence: Double) -> Color {
        if confidence > 0.9 { return .green.opacity(0.2) }
        if confidence > 0.7 { return .orange.opacity(0.2) }
        return .red.opacity(0.2)
    }
}
```

**UI States:**
- **High Confidence (>0.9):** Green highlight, auto-filled
- **Medium Confidence (0.7-0.9):** Orange highlight, show "Confirm" button
- **Low Confidence (<0.7):** Red highlight, show "Edit" button
- **Missing:** Gray placeholder, "Enter manually" prompt

**Deliverable:** Interactive UI for entity review and correction

---

### Phase 7: Testing & Refinement (Week 9-10)
**Goal:** Validate with real users, improve based on feedback

**Testing Strategy:**
1. **Unit Tests:** Each component independently
2. **Integration Tests:** Full pipeline on synthetic data
3. **User Acceptance Testing:** Medical staff with real transcripts
4. **A/B Testing:** Compare with manual entry baseline

**Metrics to Track:**
```swift
struct NERMetrics {
    var totalTranscripts: Int
    var entitiesExtracted: Int
    var highConfidence: Int  // > 0.9
    var userCorrections: Int
    var completionRate: Double  // % of fields successfully extracted
    var averageLatency: TimeInterval
    var userSatisfaction: Double  // 1-5 scale
}
```

**Refinement Loop:**
1. Collect user corrections
2. Add corrected examples to training data
3. Retrain CoreML model weekly
4. Update gazetteers with new terms
5. Adjust confidence thresholds based on precision/recall

**Deliverable:** Production-ready NER system with monitoring

---

## Code Templates

### Main NER System
```swift
// File: SurgicalSchedulingNER.swift

import Foundation
import NaturalLanguage
import CoreML

@Observable
class SurgicalSchedulingNER {
    // MARK: - Properties
    private let ruleEngine: SurgicalRuleEngine
    private let gazetteer: MedicalGazetteer
    private let nlExtractor: NaturalLanguageExtractor
    private let coreMLRecognizer: CoreMLEntityRecognizer?
    private let validator: SchedulingValidator
    private let ensembler: EnsembleNER

    // MARK: - Configuration
    struct Config {
        var confidenceThreshold: Double = 0.7
        var enableRules: Bool = true
        var enableGazetteer: Bool = true
        var enableNaturalLanguage: Bool = true
        var enableCoreML: Bool = true

        var ensembleWeights: [EntitySource: Double] = [
            .ruleBased: 0.25,
            .gazetteer: 0.30,
            .naturalLanguage: 0.15,
            .coreML: 0.30
        ]
    }

    private let config: Config

    // MARK: - Initialization
    init(config: Config = Config()) {
        self.config = config
        self.ruleEngine = SurgicalRuleEngine()
        self.gazetteer = MedicalGazetteer()
        self.nlExtractor = NaturalLanguageExtractor()
        self.coreMLRecognizer = try? CoreMLEntityRecognizer()
        self.validator = SchedulingValidator()
        self.ensembler = EnsembleNER(weights: config.ensembleWeights)

        // Load gazetteers
        gazetteer.loadProcedures()
        gazetteer.loadEquipment()
        gazetteer.loadMedications()
    }

    // MARK: - Main Processing
    func extractSchedulingEntities(from transcript: String) async throws -> SchedulingResult {
        let startTime = Date()
        var allEntities: [[Entity]] = []

        // Stage 1: Rule-Based
        if config.enableRules {
            let ruleEntities = ruleEngine.extractAll(from: transcript)
            allEntities.append(ruleEntities)
        }

        // Stage 2: Gazetteer
        if config.enableGazetteer {
            let gazEntities = gazetteer.matchAll(in: transcript)
            allEntities.append(gazEntities)
        }

        // Stage 3: NaturalLanguage
        if config.enableNaturalLanguage {
            let nlEntities = nlExtractor.extractAll(from: transcript)
            allEntities.append(nlEntities)
        }

        // Stage 4: CoreML
        if config.enableCoreML, let recognizer = coreMLRecognizer {
            let mlEntities = try await recognizer.extractAll(from: transcript)
            allEntities.append(mlEntities)
        }

        // Stage 5: Ensemble
        let merged = ensembler.merge(allEntities)

        // Stage 6: Validation
        let validated = validator.validate(merged, in: transcript)

        // Stage 7: Structure
        let schedulingData = structureSchedulingData(from: validated)

        let processingTime = Date().timeIntervalSince(startTime)

        return SchedulingResult(
            data: schedulingData,
            entities: validated,
            processingTime: processingTime,
            transcript: transcript
        )
    }

    // MARK: - Helper Methods
    private func structureSchedulingData(from entities: [Entity]) -> SchedulingData {
        var data = SchedulingData()

        for entity in entities where entity.confidence >= config.confidenceThreshold {
            switch entity.type {
            case .procedure:
                data.procedure = entity.text
                data.procedureConfidence = entity.confidence
            case .doctor:
                data.doctor = entity.text
                data.doctorConfidence = entity.confidence
            case .patient:
                data.patient = entity.text
                data.patientConfidence = entity.confidence
            case .date:
                data.date = parseDate(from: entity.text)
                data.dateConfidence = entity.confidence
            case .time:
                data.time = parseTime(from: entity.text)
                data.timeConfidence = entity.confidence
            case .location:
                data.location = entity.text
                data.locationConfidence = entity.confidence
            case .equipment:
                data.equipment.append(entity.text)
            case .medication:
                data.medications.append(entity.text)
            default:
                break
            }
        }

        return data
    }

    private func parseDate(from text: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "pt_BR")

        // Try multiple formats
        let formats = [
            "dd 'de' MMMM 'de' yyyy",
            "dd/MM/yyyy",
            "dd-MM-yyyy",
            "EEEE, dd 'de' MMMM"
        ]

        for format in formats {
            formatter.dateFormat = format
            if let date = formatter.date(from: text) {
                return date
            }
        }

        return nil
    }

    private func parseTime(from text: String) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "pt_BR")

        let formats = [
            "HH:mm",
            "HH'h'mm",
            "hh:mm a"
        ]

        for format in formats {
            formatter.dateFormat = format
            if let time = formatter.date(from: text) {
                return time
            }
        }

        return nil
    }
}

// MARK: - Result Types
struct SchedulingResult {
    let data: SchedulingData
    let entities: [Entity]
    let processingTime: TimeInterval
    let transcript: String

    var isComplete: Bool {
        data.isComplete
    }

    var needsReview: Bool {
        data.needsReview
    }
}

struct SchedulingData {
    var procedure: String?
    var procedureConfidence: Double = 0

    var doctor: String?
    var doctorConfidence: Double = 0

    var patient: String?
    var patientConfidence: Double = 0

    var date: Date?
    var dateConfidence: Double = 0

    var time: Date?
    var timeConfidence: Double = 0

    var location: String?
    var locationConfidence: Double = 0

    var equipment: [String] = []
    var medications: [String] = []

    var isComplete: Bool {
        procedure != nil && doctor != nil && patient != nil && date != nil
    }

    var needsReview: Bool {
        procedureConfidence < 0.9 ||
        doctorConfidence < 0.9 ||
        patientConfidence < 0.9 ||
        dateConfidence < 0.9
    }

    var missingFields: [String] {
        var missing: [String] = []
        if procedure == nil { missing.append("Procedimento") }
        if doctor == nil { missing.append("Médico") }
        if patient == nil { missing.append("Paciente") }
        if date == nil { missing.append("Data") }
        return missing
    }
}
```

### Entity Types
```swift
// File: EntityTypes.swift

import Foundation

struct Entity: Identifiable, Equatable {
    let id = UUID()
    let text: String
    let type: EntityType
    let range: Range<String.Index>
    let confidence: Double
    let source: EntitySource

    func overlaps(with other: Entity) -> Bool {
        range.overlaps(other.range)
    }

    static func == (lhs: Entity, rhs: Entity) -> Bool {
        lhs.id == rhs.id
    }
}

enum EntityType: String, CaseIterable {
    case procedure = "PROCEDURE"
    case doctor = "DOCTOR"
    case patient = "PATIENT"
    case date = "DATE"
    case time = "TIME"
    case location = "LOCATION"
    case equipment = "EQUIPMENT"
    case medication = "MEDICATION"
    case diagnosis = "DIAGNOSIS"
    case unknown = "UNKNOWN"

    var displayName: String {
        switch self {
        case .procedure: return "Procedimento"
        case .doctor: return "Médico"
        case .patient: return "Paciente"
        case .date: return "Data"
        case .time: return "Horário"
        case .location: return "Local"
        case .equipment: return "Equipamento"
        case .medication: return "Medicamento"
        case .diagnosis: return "Diagnóstico"
        case .unknown: return "Desconhecido"
        }
    }

    var icon: String {
        switch self {
        case .procedure: return "surgicalscalpel"
        case .doctor: return "stethoscope"
        case .patient: return "person.fill"
        case .date: return "calendar"
        case .time: return "clock.fill"
        case .location: return "mappin.circle.fill"
        case .equipment: return "wrench.and.screwdriver.fill"
        case .medication: return "pills.fill"
        case .diagnosis: return "cross.case.fill"
        case .unknown: return "questionmark.circle"
        }
    }
}

enum EntitySource {
    case ruleBased
    case gazetteer
    case gazetteerFuzzy
    case naturalLanguage
    case coreML
    case ensemble
}
```

---

## Testing Strategy

### Unit Tests
```swift
import XCTest
@testable import MarcacaoCirurgica

class NERTests: XCTestCase {
    var nerSystem: SurgicalSchedulingNER!

    override func setUp() {
        super.setUp()
        nerSystem = SurgicalSchedulingNER()
    }

    func testDateExtraction() async throws {
        let text = "Cirurgia agendada para 20 de novembro de 2025"
        let result = try await nerSystem.extractSchedulingEntities(from: text)

        XCTAssertNotNil(result.data.date)
        XCTAssertGreaterThan(result.data.dateConfidence, 0.8)
    }

    func testProcedureExtraction() async throws {
        let text = "Paciente necessita cirurgia de apendicectomia"
        let result = try await nerSystem.extractSchedulingEntities(from: text)

        XCTAssertEqual(result.data.procedure, "apendicectomia")
        XCTAssertGreaterThan(result.data.procedureConfidence, 0.9)
    }

    func testDoctorExtraction() async throws {
        let text = "Cirurgia com Dr. João Silva"
        let result = try await nerSystem.extractSchedulingEntities(from: text)

        XCTAssertEqual(result.data.doctor, "Dr. João Silva")
        XCTAssertGreaterThan(result.data.doctorConfidence, 0.7)
    }

    func testCompleteTranscript() async throws {
        let text = """
        Paciente Maria Santos necessita cirurgia de colecistectomia.
        Procedimento agendado com Dr. Pedro Oliveira no dia 25 de novembro às 14h30.
        Local: Centro Cirúrgico 2.
        Equipamento necessário: bisturi elétrico, anestesia geral.
        """

        let result = try await nerSystem.extractSchedulingEntities(from: text)

        XCTAssertTrue(result.isComplete, "All required fields should be extracted")
        XCTAssertEqual(result.data.patient, "Maria Santos")
        XCTAssertEqual(result.data.procedure, "colecistectomia")
        XCTAssertEqual(result.data.doctor, "Dr. Pedro Oliveira")
        XCTAssertNotNil(result.data.date)
        XCTAssertNotNil(result.data.time)
        XCTAssertEqual(result.data.location, "Centro Cirúrgico 2")
        XCTAssertEqual(result.data.equipment.count, 2)
    }

    func testLowConfidenceHandling() async throws {
        let text = "Cirurgia com doutor joao"  // Lowercase, informal
        let result = try await nerSystem.extractSchedulingEntities(from: text)

        // Should extract but with lower confidence
        if let doctor = result.data.doctor {
            XCTAssertLessThan(result.data.doctorConfidence, 0.9)
        }
    }

    func testPerformance() {
        let text = "Paciente João Silva necessita cirurgia de apendicectomia com Dr. Maria Santos no dia 20 de novembro."

        measure {
            Task {
                _ = try? await nerSystem.extractSchedulingEntities(from: text)
            }
        }
    }
}
```

---

## Performance Benchmarks

**Target Performance:**
- **Latency:** <100ms per transcript (average)
- **Precision:** >85% (minimize false positives)
- **Recall:** >90% (capture most entities)
- **F1 Score:** >87%
- **User Correction Rate:** <15% (most entities correct)

**Latency Breakdown:**
- Stage 1 (Rules): ~10ms
- Stage 2 (Gazetteer): ~20ms
- Stage 3 (NaturalLanguage): ~30ms
- Stage 4 (CoreML): ~40ms
- Stage 5-7 (Ensemble/Validation): ~10ms
- **Total:** ~110ms (slightly over target, optimize Stage 4)

**Optimization Strategies:**
- Cache CoreML model predictions
- Parallel execution of Stages 1-4
- Lazy loading of gazetteers
- Index gazetteer with trie data structure

---

## Monitoring & Maintenance

### Dashboard Metrics
```swift
struct NERDashboard {
    var totalTranscripts: Int
    var totalEntities: Int
    var averageConfidence: Double
    var completionRate: Double  // % with all required fields
    var userCorrectionRate: Double
    var averageLatency: TimeInterval

    var entityTypeBreakdown: [EntityType: Int]
    var confidenceDistribution: [ConfidenceBucket: Int]
    var errorPatterns: [String: Int]
}

enum ConfidenceBucket: String {
    case high = "High (>0.9)"
    case medium = "Medium (0.7-0.9)"
    case low = "Low (<0.7)"
}
```

### Weekly Maintenance Tasks
1. **Review user corrections** → Update training data
2. **Check confidence distributions** → Adjust thresholds
3. **Analyze error patterns** → Add rules/gazetteer entries
4. **Monitor latency** → Optimize slow stages
5. **Retrain CoreML model** → Incorporate new data

### Monthly Tasks
1. **Full model evaluation** on test set
2. **A/B test** new model versions
3. **Update gazetteers** with new terminology
4. **User satisfaction survey**
5. **Performance report** for stakeholders

---

## Troubleshooting Guide

### Low Recall (Missing Entities)
**Symptoms:** Many entities not detected
**Causes:**
- Incomplete gazetteer
- Weak regex patterns
- Insufficient training data

**Solutions:**
1. Add missing terms to gazetteer
2. Review transcripts for common patterns
3. Enable fuzzy matching
4. Collect more training examples

### Low Precision (False Positives)
**Symptoms:** Wrong entities extracted
**Causes:**
- Overly broad regex patterns
- Low confidence threshold
- Model overfitting

**Solutions:**
1. Tighten regex patterns
2. Increase confidence threshold
3. Add validation rules
4. More training data with negative examples

### High Latency
**Symptoms:** Slow processing (>200ms)
**Causes:**
- Large CoreML model
- Inefficient gazetteer lookup
- Too many regex patterns

**Solutions:**
1. Quantize CoreML model (reduce size)
2. Index gazetteer with trie/hash map
3. Optimize regex (combine patterns)
4. Parallel execution of stages

### Inconsistent Results
**Symptoms:** Same text gives different results
**Causes:**
- Ensemble voting ties
- Random CoreML behavior

**Solutions:**
1. Set deterministic ensemble voting (use entity text as tiebreaker)
2. Disable CoreML dropout at inference
3. Add consistency validation layer

---

## Resources & References

### Documentation
- **Main Research Report:** `/home/user/Marcacao_Cirurgica_v2/research_hybrid_entity_recognition.md`
- **Apple NaturalLanguage:** https://developer.apple.com/documentation/naturallanguage
- **CoreML:** https://developer.apple.com/documentation/coreml
- **spaCy (for training):** https://spacy.io/usage/training

### Tools
- **Annotation:** Label Studio, Prodigy, Doccano
- **Model Training:** spaCy, HuggingFace Transformers
- **Conversion:** coremltools, onnx-coreml
- **Testing:** XCTest, XCTMetric

### Medical Terminology
- **DeCS (Portuguese):** https://decs.bvsalud.org/
- **SNOMED CT:** https://www.snomed.org/
- **ICD-10:** https://www.who.int/standards/classifications/classification-of-diseases

---

## Quick Command Reference

```bash
# Run all NER tests
xcodebuild test -project SwiftTranscriptionSampleApp.xcodeproj -scheme SwiftTranscriptionSampleApp -destination 'platform=iOS Simulator,name=iPhone 15'

# Profile NER performance
instruments -t Time\ Profiler -D trace.trace path/to/YourApp.app

# Convert Python model to CoreML
python3 convert_to_coreml.py --input surgical_ner.pkl --output SurgicalNER.mlmodel

# Validate CoreML model
coremlcompiler compile SurgicalNER.mlmodel .

# Generate test data
python3 generate_synthetic_transcripts.py --output test_data.json --count 100
```

---

## Next Steps

1. **Start with Phase 1** - Build rule-based foundation
2. **Test incrementally** - Validate each stage before proceeding
3. **Collect real data** - Annotate actual transcripts for training
4. **Iterate quickly** - Weekly releases with improvements
5. **Monitor continuously** - Track metrics from day one

**Questions?** Refer to main research report for detailed explanations and academic references.
