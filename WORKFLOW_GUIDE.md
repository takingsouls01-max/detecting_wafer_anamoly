# 🔄 AI-to-AI Workflow Guide

## **How This System Works**

### **Phase Loop (Repeat for each task)**

```
┌──────────────────────────────────────────────┐
│ 1. Claude Code works on task                 │
│ 2. Claude Code updates HANDOFF.md            │
│ 3. YOU copy HANDOFF.md to Poe               │
│ 4. Poe reviews and writes feedback           │
│ 5. YOU paste Poe's feedback to HANDOFF.md   │
│ 6. Claude Code reads and implements          │
│ 7. REPEAT until task complete                │
└──────────────────────────────────────────────┘
```

---

## **Step-by-Step Instructions**

### **ITERATION 1: Initial Setup**

**In Claude Code (here):**
```
You: "Download the AI/ML materials from the drive and set up the notebook"
```

Claude Code will:
- Download files
- Create initial notebook
- Update HANDOFF.md with status

---

**In Your Browser:**
1. Open this file: `C:\Users\Takin\Downloads\gdg-wildcard-2024\HANDOFF.md`
2. Copy the ENTIRE content
3. Go to Poe.com
4. Paste and say:

```
I'm working on a 24-hour coding challenge. I'm using Claude Code locally, 
and I need you to review this handoff file and provide feedback.

Focus on:
- Bug identification
- Code correctness  
- Next steps to take

Here's the handoff:
[PASTE HANDOFF.MD CONTENT]

Please fill out the "TO: Claude Code" section with your review and 
specific implementation instructions.
```

5. Poe will analyze and provide feedback
6. Copy Poe's **entire response**
7. Come back here and say:

```
You (to Claude Code): "Here's the review from Poe: [PASTE POE'S RESPONSE]"
```

---

### **ITERATION 2-N: Working Loop**

**Repeat this cycle:**

1. **You:** "Implement Poe's suggestions and update HANDOFF.md"
2. **Claude Code:** Implements changes, updates HANDOFF.md
3. **You:** Copy HANDOFF.md → Paste to Poe → "Review these changes"
4. **Poe:** Provides next round of feedback
5. **You:** Copy Poe's feedback → Paste to Claude Code
6. **LOOP** until task complete

---

## **Templates for Quick Communication**

### **To Claude Code:**
```
Implement these changes and update HANDOFF.md:
[PASTE POE'S SUGGESTIONS]
```

### **To Poe:**
```
Here's my latest progress. Review and suggest next steps:
[PASTE HANDOFF.MD]
```

---

## **Emergency Shortcuts**

### **If Stuck:**
```
To Claude Code: "Skip this, move to next bug/task"
To Poe: "Given time constraints, what's the minimum viable fix?"
```

### **Speed Mode:**
```
To Claude Code: "Quick fix without perfect code, update HANDOFF"
To Poe: "30-second review, critical issues only"
```

---

## **Handoff File Sections Explained**

### **FROM: Claude Code**
This is what I (Claude Code) fill out:
- Current status
- Code changes made
- Questions/concerns
- What I need reviewed

### **TO: Claude Code**
This is what Poe fills out:
- Code review
- Bug findings
- Implementation instructions
- Priority actions

---

## **Example Workflow (AI/ML Task)**

**Iteration 1:**
- Claude Code: Downloads notebook, identifies bugs conceptually
- Poe: Validates bug identification, suggests starting with Bug #1

**Iteration 2:**
- Claude Code: Fixes data leakage bug
- Poe: Reviews code, spots edge case, suggests refinement

**Iteration 3:**
- Claude Code: Applies refinement, moves to Bug #2
- Poe: Confirms Bug #1 fixed, guides Bug #2 approach

**Iteration 4:**
- Claude Code: Fixes tensor shape bug
- Poe: Reviews, suggests testing strategy

**Continue until complete...**

---

## **Time Estimates Per Iteration**

- Claude Code work: 30-60 min
- HANDOFF.md update: 2 min
- Copy to Poe: 1 min
- Poe review: 2-3 min
- Copy back: 1 min
- **Total cycle: 35-65 min**

**For 16-hour AI/ML task: ~12-15 iterations possible**

---

## **Tips for Maximum Efficiency**

1. **Don't wait for perfect code** - Get feedback early
2. **Use specific questions** - "Is this train-test split correct?" vs "Review this"
3. **Keep handoffs focused** - One bug/feature per iteration
4. **Trust but verify** - Poe might miss context, Claude Code double-checks
5. **Parallel work** - While model trains, get next review ready

---

## **Ready to Start?**

Say to Claude Code:
```
"Start the first iteration: Download AI/ML materials and set up the workspace. 
Update HANDOFF.md when ready for review."
```

Then follow the workflow loop! 🚀
