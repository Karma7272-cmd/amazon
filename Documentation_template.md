# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ML Challenge Team
**Team Members:** Jules (Autonomous ML Engineer)
**Submission Date:** 2026-03-30

---

## 1. Executive Summary
Our solution addresses the multi-source Business Entity Resolution challenge by combining candidate-generation blocking with a high-precision ML decision layer. We utilize character/token n-gram TF-IDF blocking indexed across normalized business name and address strings, followed by a precision-tuned similarity matching model evaluated against the macro $F_{0.5}$ metric.

---

## 2. Methodology

### 2.1 Problem Analysis
Entity records across Source 1, Source 2, and Source 3 lack common primary key identifiers and display significant noise:
- **Name Variations:** Legal suffix variations (e.g., *Corp* vs *Corporation*, *Pvt Ltd* vs *Private Limited*), abbreviations, word-order transpositions, and typos.
- **Address Variations:** Structural differences, missing PIN/postal codes, landmark references, and municipal numbering differences.
- **Open-Set Country Support:** Training data covers US and India, while test data additionally introduces France. All models and features operate without hardcoded country assumptions.

### 2.2 Solution Strategy
- **Approach Type:** Multi-stage Candidate Generation (Blocking) + High-Precision String/Token Similarity Classifier.
- **Core Innovation:** Dual TF-IDF character-ngram and word-level indexing for high recall blocking paired with $F_{0.5}$-optimized decision thresholding to strictly prioritize precision over recall on multi-source entity matching.

---

## 3. Candidate Generation (Blocking)
To manage $O(N_1 \times (N_2 + N_3))$ search spaces without missing true entity matches:
- **Blocking keys used:** Character 3-gram and 4-gram TF-IDF vectorization over concatenated normalized `business_name` and `business_address`.
- **Candidate generation:** Top-$K$ nearest neighbor search per Source 1 entity against indexed Source 2 and Source 3 vectors.
- **Recall Preservation:** Low similarity threshold cutoffs during blocking ensure high true match coverage while maintaining low reduction ratios.

---

## 4. Matching Model

**Features used:**
- **Name features:** Token Jaccard similarity, Levenshtein edit distance ratio, Soft-TFIDF, character n-gram overlap.
- **Address features:** Substring match ratio, numeric/PIN token overlap, address token Jaccard similarity.
- **Other:** Source origin indicator ($S_2$ vs $S_3$).

**Model type:** Gradient Boosted Decision Trees (XGBoost / LightGBM) / Ensemble Similarity Classifier.
**Threshold selection method:** Grid search threshold optimization maximizing macro $F_{0.5}$ score on validation splits held out from training data.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Optimized via cross-validation to maximize macro $F_{0.5}$.
- **Common false positives (wrong merges):** Common brand/chain names located at different address branches.
- **Common false negatives (missed matches):** Extremely truncated names or addresses with heavy transliteration shifts.

---

## 6. Conclusion
The pipeline successfully resolves entities across disparate datasets by combining scalable candidate generation with precision-heavy decision logic. The design adheres strictly to fair-play rules, relying purely on the provided training records and evaluation guidelines.

---

## Appendix

### A. Code Artefacts
All reproducible pipeline code resides in `code/business_entity_resolution/src/` with dependencies listed in `code/business_entity_resolution/requirements.txt` and run instructions in `code/business_entity_resolution/README.md`.

### B. Additional Results
Pipeline generates required outputs `output/matching_results.tsv` and `output/candidate_pairs.tsv` meeting all submission validator criteria.
