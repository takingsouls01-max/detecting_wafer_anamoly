# 🎨 AI Collaboration System Diagram

## **High-Level Architecture**

```
╔════════════════════════════════════════════════════════════════╗
║                    24-HOUR SPRINT SYSTEM                        ║
╚════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────┐
│                         YOU (Orchestrator)                       │
│  - Copies files between systems                                 │
│  - Makes final decisions                                        │
│  - Tracks time and progress                                     │
└────────┬───────────────────────────────────────────────┬────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────────────┐                   ┌──────────────────┐
│   CLAUDE CODE (Local)   │                   │  POE (Cloud)     │
│  ─────────────────────  │                   │  ──────────────  │
│  Role: Implementation   │◄─────────────────►│  Role: Review    │
│                         │   HANDOFF.md      │                  │
│  Capabilities:          │                   │  Capabilities:   │
│  • Write code           │                   │  • Code review   │
│  • Run tests            │                   │  • Bug finding   │
│  • Git operations       │                   │  • Architecture  │
│  • File management      │                   │  • Suggestions   │
│  • Terminal access      │                   │  • Analysis      │
└─────────────────────────┘                   └──────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────────────┐                   ┌──────────────────┐
│   Local Repository      │                   │  Review Output   │
│   ─────────────────     │                   │  ────────────    │
│  • Source code          │                   │  • Feedback      │
│  • HANDOFF.md (state)   │                   │  • Next steps    │
│  • Documentation        │                   │  • Warnings      │
│  • Test results         │                   │  • Guidance      │
└─────────────────────────┘                   └──────────────────┘
```

---

## **Data Flow Diagram**

```
ITERATION N (Repeat Loop)
═══════════════════════════════════════════════════════════════

Step 1: Implementation
┌──────────────┐
│ Claude Code  │  Does work (code/test/fix)
└──────┬───────┘
       │ writes
       ▼
┌──────────────┐
│ HANDOFF.md   │  Documents: What changed? What's next?
└──────────────┘


Step 2: Transfer to Review
┌──────────────┐
│    YOU       │  Copy HANDOFF.md content
└──────┬───────┘
       │ paste
       ▼
┌──────────────┐
│    Poe       │  Receives context for review
└──────────────┘


Step 3: Review
┌──────────────┐
│    Poe       │  Analyzes code, finds issues, suggests fixes
└──────┬───────┘
       │ outputs
       ▼
┌──────────────┐
│  Review Doc  │  Structured feedback + action items
└──────────────┘


Step 4: Transfer to Implementation  
┌──────────────┐
│    YOU       │  Copy review
└──────┬───────┘
       │ paste
       ▼
┌──────────────┐
│ Claude Code  │  Receives instructions
└──────────────┘


Step 5: Apply Feedback
┌──────────────┐
│ Claude Code  │  Implements review suggestions
└──────┬───────┘
       │
       └──────► LOOP BACK TO STEP 1

═══════════════════════════════════════════════════════════════
```

---

## **State Management**

```
HANDOFF.md Structure
══════════════════════════════════════════════════════════

┌────────────────────────────────────────────┐
│  📤 FROM: Claude Code                      │
│  ─────────────────────────────────────     │
│  • Current task/phase                      │
│  • Code changes made                       │
│  • Test results                            │
│  • Questions for reviewer                  │
│  • Files modified                          │
│  • Timestamp                               │
└────────────────────────────────────────────┘
                     │
                     │ State handoff
                     ▼
┌────────────────────────────────────────────┐
│  📥 TO: Claude Code                        │
│  ─────────────────────────────────────     │
│  • Review status                           │
│  • Issues found                            │
│  • Specific fixes needed                   │
│  • Priority action items                   │
│  • Next steps                              │
│  • Timestamp                               │
└────────────────────────────────────────────┘
```

---

## **Timeline View**

```
24-Hour Sprint Timeline
═══════════════════════════════════════════════════════════

Hour 0  ═══════════════════════════════════════════════════
│ Setup  │ ← You are here
│        │
Hour 2  ───────────────────────────────────────────────────
│        │
│ AI/ML  │  Iteration 1: Download + Initial analysis
│ Task   │  Iteration 2: Fix Bug #1 (Data leakage)
│        │  Iteration 3: Fix Bug #2 (Tensor shapes)
Hour 10 ───  Iteration 4: Fix Bug #3 (Categorical)
│        │  Iteration 5: Evaluation
│        │  Iteration 6: Documentation
Hour 16 ───────────────────────────────────────────────────
│        │
│Cybersec│  Iteration 7: Setup Linux
│ Task   │  Iteration 8: Binary analysis
│        │  Iteration 9: Find flag
Hour 22 ───  Iteration 10: Documentation
│        │
Hour 24 ═══════════════════════════════════════════════════
│ Submit │
═══════════════════════════════════════════════════════════

Each iteration: ~1.5 hours (1h work + 30min review/handoff)
```

---

## **Collaboration Matrix**

```
Who Does What?
════════════════════════════════════════════════════════════

┌──────────────┬─────────────┬─────────────┬──────────────┐
│   Activity   │ Claude Code │     Poe     │     You      │
├──────────────┼─────────────┼─────────────┼──────────────┤
│ Write code   │      ✅     │      ❌     │      ❌      │
│ Run tests    │      ✅     │      ❌     │      ❌      │
│ Git ops      │      ✅     │      ❌     │      ❌      │
│ Code review  │      ❌     │      ✅     │      ❌      │
│ Bug finding  │      ⚠️     │      ✅     │      ⚠️      │
│ Architecture │      ⚠️     │      ✅     │      ❌      │
│ Copy/paste   │      ❌     │      ❌     │      ✅      │
│ Decisions    │      ⚠️     │      ⚠️     │      ✅      │
│ Time mgmt    │      ❌     │      ❌     │      ✅      │
└──────────────┴─────────────┴─────────────┴──────────────┘

✅ = Primary responsibility
⚠️ = Supporting role
❌ = Not responsible
```

---

## **Error Handling Flow**

```
What Happens When Things Go Wrong?
═══════════════════════════════════════════════════════════

┌─────────────┐
│ Code breaks │
└──────┬──────┘
       │
       ▼
┌──────────────────────────┐      ┌────────────────┐
│ Claude Code tries fix    │─────►│ Fixed?         │
└──────────────────────────┘      └────┬───────┬───┘
                                       │ Yes   │ No
                                       ▼       ▼
                                  ┌─────┐  ┌──────────────┐
                                  │Done │  │Ask Poe       │
                                  └─────┘  └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌────────────────┐
                                          │ Poe diagnoses  │
                                          └────────┬───────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ Claude applies │
                                          └────────┬───────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │Still broken?   │
                                          └───┬────────┬───┘
                                              │ No     │ Yes
                                              ▼        ▼
                                          ┌─────┐  ┌──────┐
                                          │Done │  │ YOU  │
                                          └─────┘  │decide│
                                                   │skip/ │
                                                   │pivot │
                                                   └──────┘
```

---

## **Success Indicators**

```
How to Know It's Working
═══════════════════════════════════════════════════════════

Green Flags (Keep going!) 🟢
┌────────────────────────────────────────────┐
│ ✓ Each iteration completes in < 90 min    │
│ ✓ HANDOFF updates are clear               │
│ ✓ Poe provides actionable feedback        │
│ ✓ Code changes fix identified issues      │
│ ✓ Progress toward task completion         │
└────────────────────────────────────────────┘

Yellow Flags (Adjust!) 🟡
┌────────────────────────────────────────────┐
│ ⚠ Iterations taking > 2 hours              │
│ ⚠ Vague handoffs, unclear questions        │
│ ⚠ Poe asking for more context              │
│ ⚠ Same bug appearing multiple times        │
│ ⚠ No tests validating fixes                │
└────────────────────────────────────────────┘

Red Flags (Pivot!) 🔴
┌────────────────────────────────────────────┐
│ ✗ Stuck on same issue > 1 hour             │
│ ✗ No progress in 3 iterations              │
│ ✗ Behind schedule by > 3 hours             │
│ ✗ Code getting worse, not better           │
│ ✗ Less than 4 hours remaining              │
└────────────────────────────────────────────┘
   │
   └──► ACTION: Skip to minimum viable solution
```

---

**This system is designed for SPEED with QUALITY through collaboration!** 🚀
