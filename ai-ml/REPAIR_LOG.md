# Repair Verification Log

**Branch:** repair/systematic-verification  
**Date:** 2026-10-19  
**Reviewer Instructions:** Systematically verify and repair 9 priority areas

---

## PRIORITY 1: Checkpointing

**Hypothesis:** best_model_state uses shallow state_dict copy

**Investigation:**
- scripts/train.py:78 and :95 used `model.state_dict().copy()`
- Created test_checkpointing_debug.py to verify behavior
- Ran experiments showing shallow copy tensors share storage with model

**Finding:** ✅ CONFIRMED - Critical bug found!
- Shallow copy checkpoint gets CORRUPTED during training
- Test showed shallow checkpoint changed from `[0.2418, ...]` to `[0.4974, ...]`
- Deep checkpoint correctly preserved original values
- Restoration error: 0.1662 (shallow) vs 0.00e+00 (deep)

**Action:**
- Added `import copy` to train.py
- Changed line 78: `best_model_state = copy.deepcopy(model.state_dict())`
- Changed line 95: `best_model_state = copy.deepcopy(model.state_dict())`
- Created tests/test_checkpointing.py with two tests:
  - test_deepcopy_checkpoint_independence() - proves deep copy works
  - test_shallow_copy_fails() - documents the bug

**Status:** ✅ FIXED - Tests pass, bug documented

---

## PRIORITY 2: Loss Correctness

**Hypothesis:** Original [B,F] - [B,1,F] broadcast produces [B,B,F]; test expects wrong shape

**Investigation:**
- Verified actual broadcast: `[8,594] - [8,1,594] = [8,8,594]` (not [8,594,594])
- tests/test_loss_integrity.py:67 expected (B, F, F) - WRONG
- Reviewed loss function in model.py - already has correct flattening

**Finding:** ✅ CONFIRMED - Test had wrong expected shape
- Test expected [B, F, F] but actual broadcast is [B, B, F]
- Broadcasting: [B,F] → [1,B,F], then [1,B,F] - [B,1,F] → [B,B,F]
- Memory explosion: 0.15 MB vs 18.56 KB (8x bloat)

**Action:**
- Fixed test_loss_integrity.py:67 to expect (B, B, F)
- Removed unicode characters (✅ → [OK], ⚠️ removed)
- Added test_loss_values_and_gradients_reference() with B=2, F=3 manual verification
- Added test_partial_batch_handling() for edge cases
- All 7 tests PASS

**Status:** ✅ FIXED - Test corrected, additional verification added

---

## PRIORITY 3: Legacy Semantics

**Hypothesis:** Multiple issues with legacy sensor handling

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 4: Temporal Safety

**Hypothesis:** Future-mutation test needs strengthening; evaluator artifacts may be used

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 5: Tests

**Hypothesis:** Not running full pytest suite; tests don't call production code

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 6: Training and Metrics

**Hypothesis:** Multiple issues with batch handling, subgroup metrics, baseline evaluation

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 7: Experiment Integrity

**Hypothesis:** Threshold protocol mismatch, test data already inspected

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 8: Reproduce Results

**Hypothesis:** Need complete reproducibility artifacts

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## PRIORITY 9: Submission

**Hypothesis:** Missing notebook, stale docs, unclear permissions

**Investigation:**

**Finding:**

**Action:**

**Status:**

---

## Summary of Findings

**Confirmed Issues:**

**Rejected Hypotheses:**

**Repairs Made:**

**Tests Added:**

**Results Changed:**
