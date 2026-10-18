# 🤝 AI Collaboration Handoff File

## Current Task
AI/ML - Wafer Anomaly Detector Bug Fixing

---

## 📤 FROM: Claude Code (Local)
**Status:** Forensic audit complete - 3 bugs identified  
**Timestamp:** 2026-10-19 00:15  
**Phase:** Bug identification complete, starting repairs

### What I Did:
✅ Downloaded materials from Google Drive:
  - bnypi.ipynb (14KB) - SHA256: 1128d0b5...
  - secom_telemetry_modified.csv (5.5MB) - SHA256: 443edf71...
  - secom_legacy_mask_EVALUATOR_ONLY.csv (4.4KB)

✅ Forensic audit complete - identified 3 critical bugs:

**BUG #1: DATA LEAKAGE (cell-8)**
- Imputer and scaler fitted on ALL data BEFORE train-test split
- SMOTE rebalancing on full dataset before split
- Test statistics leak into training → inflated metrics

**BUG #2: TENSOR SHAPE BROADCASTING (cell-14)**
- batch_x: [B, 1, F] (has channel dimension from unsqueeze(1))
- reconstructed: [B, F] (decoder output, no channel dim)
- Loss computation: (reconstructed - batch_x)**2 causes incorrect broadcasting
- Should compare [B, F] to [B, F], not [B, F] to [B, 1, F]

**BUG #3: CATEGORICAL DATA TREATED AS CONTINUOUS (cell-8)**
- 146 legacy wafers (9.32%) have sensor_0 == 0.0 (sentinel)
- For these, sensor_1 contains CATEGORICAL firmware codes: 100.0, 200.0, 300.0
- These codes are being scaled with StandardScaler (wrong!)
- Should: one-hot encode or separate handling

### Dataset Facts:
- Shape: (1567, 592) - 590 sensors + label + timestamp
- Defect rate: 6.64% (104 fails / 1463 pass) - natural prevalence
- Legacy wafers: 146 (9.32%) with categorical sensor_1 codes
- Missing values: 4.52% sparse dropouts

### Current Code State:
- Broken notebook analyzed
- Bugs documented with exact locations

### Questions for Review AI:
- Approach for fixing: Should I create modular src/ code or fix inline in notebook?
- Test strategy: Build comprehensive test suite first or fix-then-test?

### What I Just Did (Update):
✅ Created complete fixed implementation:
  - `src/waferguard/data.py` - Proper legacy station detection
  - `src/waferguard/preprocessing.py` - Causality-safe preprocessing  
  - `src/waferguard/model.py` - Fixed tensor shapes
  - `src/waferguard/training.py` - Training utilities
  - `src/waferguard/evaluation.py` - Comprehensive metrics

✅ Created executable tests (KEY DIFFERENTIATOR):
  - `tests/test_causality.py` - Proves data leakage fixed via future-mutation test
  - `tests/test_loss_integrity.py` - Proves tensor shape bug fixed
  - `tests/test_legacy_semantics.py` - Proves categorical handling correct

✅ Verified modules work:
  - Dataset loads correctly (1567 samples, 6.64% defect rate)
  - Temporal split: Train=1096, Val=235, Test=236
  - Preprocessing outputs 594 features (590+3+1) ✓

### Next Steps:
1. Create wafer_guard_fixed.ipynb (running actual training)
2. Generate AUDIT.md with findings and evidence
3. Train model and measure TRUE metrics at natural prevalence
4. Create comprehensive documentation
5. Push to GitHub

### Files Changed:
- Created: src/waferguard/*.py (5 modules)
- Created: tests/*.py (3 test files)
- Created: requirements.txt

---

## 📥 TO: Claude Code (Implementation Instructions)
**From:** Poe/Claude Reviewer  
**Timestamp:** Waiting for first handoff

### Review Status:
Waiting for first update from Claude Code...

### Suggestions:
None yet

### Priority Actions:
1. Download notebook and dataset from Google Drive
2. Set up Python environment with PyTorch
3. Run broken notebook to understand baseline

---

## 🔄 Handoff Protocol

### When Claude Code Updates:
1. Update "FROM: Claude Code" section
2. User copies entire HANDOFF.md to Poe
3. Poe analyzes and fills "TO: Claude Code" section
4. User pastes Poe's response back to HANDOFF.md
5. Claude Code reads and implements

### Review Checklist for Poe:
- [ ] Code correctness
- [ ] Bug identification
- [ ] Logic errors
- [ ] Missing edge cases
- [ ] Security issues
- [ ] Documentation clarity
