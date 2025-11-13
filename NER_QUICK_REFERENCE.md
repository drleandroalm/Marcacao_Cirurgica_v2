# NER Implementation Quick Reference
## TL;DR for iOS/macOS Surgical Transcription

### Best Approach: Hybrid Rule-Based + CRF

**Why?**
- 85-90% F1 accuracy
- <25ms latency
- <250 MB memory
- 4-week implementation
- No server required

---

## Three-Tier Architecture

```
┌─────────────────────────────────────────┐
│ TIER 1: Rules (Regex + Gazetteer)      │
│ - Procedure codes, dates, dosages      │
│ - Latency: <5ms, Memory: <20 MB        │
│ - Precision: 90%+, Recall: 40-60%      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ TIER 2: CRF (Classical ML)              │
│ - General medical entities              │
│ - Latency: ~20ms, Memory: ~200 MB      │
│ - F1: 85-91%                            │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│ TIER 3: BiLSTM-CRF (Optional)           │
│ - Complex entities, high accuracy       │
│ - Latency: ~50ms, Memory: ~300 MB      │
│ - F1: 91-93%                            │
└─────────────────────────────────────────┘
```

---

## Quick Comparison

| Method | F1 Score | Latency | Memory | Dev Time | Complexity |
|--------|----------|---------|--------|----------|------------|
| **Regex + Gazetteer** | 70% | 5ms | 50 MB | 2 weeks | LOW |
| **CRF** | 88% | 20ms | 200 MB | 4 weeks | MEDIUM |
| **Hybrid (Rules+CRF)** | **90%** | **25ms** | **250 MB** | **4 weeks** | **MEDIUM** ⭐ |
| BiLSTM-CRF | 92% | 50ms | 300 MB | 8 weeks | HIGH |
| DistilBERT | 94% | 100ms | 400 MB | 10 weeks | HIGH |
| Apple NL Framework | 54% | 10ms | 10 MB | 1 day | NONE |

⭐ **Recommended for production**

---

## 4-Week Implementation Plan

### Week 1: Rule-Based Foundation
```swift
// Regex patterns
let patterns = [
    "PROCEDURE_CODE": "\\b\\d{5}\\b",      // CPT codes
    "ICD10": "[A-Z]\\d{2}\\.?\\d{0,4}",    // ICD-10
    "DATE": "\\d{1,2}/\\d{1,2}/\\d{2,4}",
    "TIME": "\\d{1,2}:\\d{2}(?:\\s?[AP]M)?",
    "DOSE": "\\d+(?:\\.\\d+)?\\s?(?:mg|ml|g)"
]

// Gazetteer (500-1000 procedures from ICD-10)
let procedures = loadGazetteer("surgical_procedures.txt")
```

**Deliverable:** 60-70% recall, <5ms latency

### Week 2: Gazetteer Expansion
- Build medication list (RxNorm - top 200 drugs)
- Build anatomy list (FMA - 100 common terms)
- Implement bigram matching
- Add post-processing

**Deliverable:** 70-80% recall, <10ms latency

### Week 3: CRF Training
- Collect training data (MTSamples, i2b2)
- Extract features (word shape, POS, context)
- Train CRF using sklearn-crfsuite
- Export model

**Deliverable:** Trained CRF model, 80-85% F1

### Week 4: CRF Integration
- Implement Swift feature extraction
- Port CRF inference or use bridging
- Combine with rule-based tier
- Benchmark and optimize

**Deliverable:** 85-90% F1, <25ms latency

---

## Essential Resources

### Datasets
1. **MTSamples** (Free) - 5,000 medical transcriptions
   https://www.mtsamples.com/

2. **i2b2 2010** (Registration required) - Clinical NER
   https://www.i2b2.org/NLP/DataSets/

3. **CoNLL 2003** (Free) - General NER benchmark
   https://www.clips.uantwerpen.be/conll2003/ner/

### Gazetteers
1. **ICD-10-PCS** (Free) - Procedure codes
   https://www.cms.gov/medicare/coding-billing/icd-10-codes

2. **UMLS** (Free with license) - 4M+ medical terms
   https://www.nlm.nih.gov/research/umls/

3. **RxNorm** (Free) - Medication names
   https://www.nlm.nih.gov/research/umls/rxnorm/

### Libraries

**Python (for training):**
```bash
pip install sklearn-crfsuite  # CRF training
pip install spacy              # Tokenization, POS tagging
pip install scispacy           # Medical NER models
```

**iOS/macOS:**
- Natural Language framework (built-in)
- Core ML (built-in)
- ONNX Runtime (https://onnxruntime.ai/docs/build/ios.html)
- Kafka (https://github.com/questo-ai/Kafka) - Pure Swift NLP

### GitHub Repos
1. **CRF++:** https://taku910.github.io/crfpp/
2. **NCRF++:** https://github.com/jiesutd/NCRFpp
3. **BiLSTM-CRF:** https://github.com/guillaumegenthial/sequence_tagging
4. **SciSpacy:** https://github.com/allenai/scispacy

---

## Code Templates

### 1. Minimal Rule-Based NER (Swift)
```swift
import Foundation

func extractEntities(from text: String) -> [(String, String)] {
    var entities: [(String, String)] = []

    // Regex patterns
    let patterns = [
        ("PROCEDURE_CODE", try! NSRegularExpression(pattern: "\\b\\d{5}\\b")),
        ("DATE", try! NSRegularExpression(pattern: "\\d{1,2}/\\d{1,2}/\\d{2,4}")),
        ("DOSE", try! NSRegularExpression(pattern: "\\d+(?:\\.\\d+)?\\s?mg", options: .caseInsensitive))
    ]

    for (type, regex) in patterns {
        let range = NSRange(text.startIndex..., in: text)
        regex.enumerateMatches(in: text, range: range) { match, _, _ in
            guard let match = match, let r = Range(match.range, in: text) else { return }
            entities.append((String(text[r]), type))
        }
    }

    return entities
}
```

### 2. CRF Feature Extraction (Swift)
```swift
func extractFeatures(token: String, index: Int, tokens: [String], posTags: [String]) -> [String: Any] {
    var features: [String: Any] = [:]

    // Word features
    features["word.lower"] = token.lowercased()
    features["word.isupper"] = token.uppercased() == token
    features["word.istitle"] = token.capitalized == token

    // Morphological features
    features["word.prefix-3"] = String(token.prefix(min(3, token.count)))
    features["word.suffix-3"] = String(token.suffix(min(3, token.count)))

    // Word shape (Aa0 for "Dr5")
    features["word.shape"] = token.map {
        $0.isUppercase ? "A" : $0.isLowercase ? "a" : $0.isNumber ? "0" : String($0)
    }.joined()

    // POS tag
    features["postag"] = posTags[index]

    // Context
    if index > 0 {
        features["word[-1].lower"] = tokens[index-1].lowercased()
    }
    if index < tokens.count - 1 {
        features["word[+1].lower"] = tokens[index+1].lowercased()
    }

    return features
}
```

### 3. Training CRF (Python)
```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# Extract features for all tokens
X_train = [extract_features_for_sentence(s) for s in train_sentences]
y_train = [get_labels_for_sentence(s) for s in train_sentences]

# Train CRF
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Evaluate
X_test = [extract_features_for_sentence(s) for s in test_sentences]
y_test = [get_labels_for_sentence(s) for s in test_sentences]
y_pred = crf.predict(X_test)

print(metrics.flat_f1_score(y_test, y_pred, average='weighted'))
# Expected: 85-91% F1
```

---

## Performance Optimization

### Quantization (4x size reduction)
```python
# TensorFlow Lite quantization
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

quantized_model = converter.convert()
# Result: 250 MB → 65 MB
```

### ONNX Export
```python
# PyTorch → ONNX
import torch

dummy_input = torch.randn(1, 128, 300)  # (batch, seq_len, embedding_dim)
torch.onnx.export(model, dummy_input, "ner_model.onnx",
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {1: 'seq_len'}})
```

### Core ML Conversion
```bash
# ONNX → Core ML
coremltools-convert --source onnx --target coreml \
  ner_model.onnx -o NERModel.mlmodel
```

---

## Common Pitfalls

### ❌ DON'T:
- Use HMM (superseded by CRF, 10-15% F1 lower)
- Train BERT from scratch on mobile (too large, use DistilBERT)
- Ignore medical gazetteers (critical for domain accuracy)
- Deploy unquantized models (4x larger than necessary)
- Use Apple NL Framework for medical NER (only 54% F1)

### ✅ DO:
- Start with rule-based MVP (immediate value)
- Use domain-specific gazetteers (ICD-10, UMLS, RxNorm)
- Quantize neural models to INT8
- Combine rules + ML for best results
- Benchmark on target devices (iPhone 12+)
- Use CRF for classical ML (best accuracy/complexity ratio)

---

## Accuracy Targets

| Use Case | Required F1 | Recommended Approach |
|----------|-------------|---------------------|
| Demo/Prototype | 60-70% | Rules + Gazetteer |
| Internal Tool | 80-85% | Hybrid (Rules + CRF) |
| **Production App** | **85-90%** | **Hybrid (Rules + CRF)** ⭐ |
| Clinical Decision Support | 90-95% | Hybrid (All 3 Tiers) + Human Review |
| Research | 95%+ | DistilBERT + Ensemble + Human Review |

---

## Decision Tree

```
Need NER for surgical transcription?
├─ Need >90% F1?
│  ├─ YES → BiLSTM-CRF (quantized) or DistilBERT
│  └─ NO → Continue
├─ Have 4+ weeks development time?
│  ├─ YES → Hybrid (Rules + CRF) ⭐ RECOMMENDED
│  └─ NO → Continue
├─ Have 2 weeks?
│  ├─ YES → Rules + Gazetteer
│  └─ NO → Apple Natural Language Framework (54% F1)
```

---

## Medical Entity Types

### Extractable with High Confidence (Rules + Gazetteer):
- ✅ Procedure codes (CPT, ICD-10)
- ✅ Medication names
- ✅ Dates and times
- ✅ Dosages
- ✅ Lab values
- ✅ Anatomical locations (when standardized)

### Require ML (CRF or Neural):
- 🔶 Symptoms/diagnoses (free text)
- 🔶 Procedure descriptions (narrative)
- 🔶 Adverse events
- 🔶 Patient history
- 🔶 Complex temporal expressions

---

## Memory Budget Guidelines

**iOS App Memory Limits:**
- iPhone 8: ~1.4 GB (foreground)
- iPhone 12+: ~2.8 GB (foreground)
- Background: ~200 MB (all devices)

**NER Model Allocation:**
- Rules + Gazetteer: <50 MB ✅
- CRF: 200-250 MB ✅
- BiLSTM-CRF (quantized): 300-400 MB ✅
- DistilBERT (quantized): 400-500 MB ⚠️
- DistilBERT (unquantized): 1+ GB ❌

**Recommendation:** Keep total NER memory <400 MB for reliable performance

---

## Getting Started Checklist

### Day 1:
- [ ] Download ICD-10 procedure codes
- [ ] Create surgical procedure gazetteer (500 terms)
- [ ] Implement regex patterns for codes, dates, dosages
- [ ] Test on sample transcripts

### Week 1:
- [ ] Expand gazetteers (medications, anatomy)
- [ ] Implement bigram matching
- [ ] Build evaluation harness
- [ ] Measure baseline recall

### Week 2:
- [ ] Download MTSamples dataset
- [ ] Annotate 100-200 transcripts (or use pre-annotated)
- [ ] Set up sklearn-crfsuite
- [ ] Implement feature extraction

### Week 3:
- [ ] Train CRF model
- [ ] Evaluate on test set
- [ ] Tune hyperparameters
- [ ] Export model

### Week 4:
- [ ] Port CRF to Swift (or bridge to Python)
- [ ] Integrate with rule-based tier
- [ ] Benchmark latency and memory
- [ ] Optimize and deploy

---

## Support and Community

### Forums:
- **Hugging Face:** https://discuss.huggingface.co/
- **spaCy Discussions:** https://github.com/explosion/spaCy/discussions
- **Stack Overflow:** Tag `named-entity-recognition`, `swift`, `coreml`

### Papers:
- **Survey (2024):** "Recent Advances in NER" - arXiv:2401.10825v3
- **Classical CRF:** "CRF for Sequence Labeling" - Lafferty et al., 2001
- **BiLSTM-CRF:** arXiv:1508.01991

---

## FAQ

**Q: Can I use spaCy directly on iOS?**
A: No. spaCy is Python-based. You need to convert models to Core ML or ONNX, but spaCy doesn't officially support export. Use alternative training frameworks.

**Q: Is Apple's Natural Language framework good enough?**
A: For general NER, maybe. For medical NER, no (54% F1 vs 90%+ for custom models).

**Q: How do I handle HIPAA compliance?**
A: Use on-device processing only (no cloud), use de-identified training data, implement proper data security.

**Q: CRF vs BiLSTM-CRF - which one?**
A: CRF for simplicity (4 weeks dev, 88% F1). BiLSTM-CRF for accuracy (8 weeks dev, 92% F1).

**Q: Can I skip the CRF and just use rules?**
A: Yes, but expect 65-75% recall vs 85-90% with CRF. Good for MVP.

---

**Last Updated:** 2025-11-13
**For Full Details:** See `NER_RESEARCH_REPORT.md`
