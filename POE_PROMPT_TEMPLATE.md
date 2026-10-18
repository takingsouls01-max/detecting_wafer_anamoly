# 📋 Poe Prompt Templates

Copy-paste these into Poe for maximum effectiveness.

---

## **Initial Setup Prompt (Use Once)**

```
You are a senior code reviewer working with me on a 24-hour coding challenge 
for GDG Tech Wildcard 2024. I'm using Claude Code locally to write code, 
and you'll review via a HANDOFF.md file.

TASKS:
1. AI/ML: Fix 3 critical bugs in a PyTorch wafer anomaly detector
   - Bug 1: Data leakage (train-test contamination)
   - Bug 2: Tensor shape mismatches
   - Bug 3: Categorical features treated as continuous
   
2. Cybersecurity: Binary analysis and web app security

YOUR ROLE:
- Review code for correctness
- Identify bugs and edge cases
- Provide specific, actionable implementation instructions
- Keep me on track for 24-hour deadline

CONTEXT:
- I know Python but rely on AI to write code
- I understand concepts but need implementation guidance
- Priority: Working solution > Perfect code
- Time constraint: 24 hours total

Let's start! I'll share HANDOFF.md updates with you for review.
```

---

## **Standard Review Prompt (Use Every Iteration)**

```
Here's my latest progress on [AI/ML / Cybersecurity] task.

Please review the HANDOFF.md file below and:
1. Validate the code changes made
2. Identify any bugs or issues
3. Suggest the next immediate action
4. Update the "TO: Claude Code" section with specific instructions

Remember: 24-hour deadline, focus on critical issues only.

---
[PASTE ENTIRE HANDOFF.MD CONTENT HERE]
---

Provide your review in this format:

## Review Summary
[Quick overview of what was done]

## Issues Found
- [List critical issues]

## Next Steps
1. [Specific action 1]
2. [Specific action 2]

## Updated HANDOFF Section
```markdown
## 📥 TO: Claude Code (Implementation Instructions)
**From:** Poe/Claude Reviewer  
**Timestamp:** [Current time]

### Review Status:
[Your status]

### Suggestions:
[Your suggestions]

### Priority Actions:
[Your priority list]
```
```

---

## **Quick Check Prompt (For Fast Iterations)**

```
Quick 30-second review needed!

Task: [AI/ML Bug #X / Cybersec challenge]
Question: [Specific question - e.g., "Is this train-test split correct?"]

Code snippet:
```python
[PASTE RELEVANT CODE]
```

Critical issues only. What's wrong and how to fix?
```

---

## **Stuck Prompt (When Blocked)**

```
I'm stuck on [specific issue].

What I tried:
- [Attempt 1]
- [Attempt 2]

Error/Problem:
[Paste error or describe issue]

Time remaining: [X hours]

What's the fastest path forward? Skip this part or quick workaround?
```

---

## **Final Review Prompt (Before Submission)**

```
Final review before submission!

Task: [AI/ML / Cybersecurity]

Files to review:
1. [File 1 name and purpose]
2. [File 2 name and purpose]

---
[PASTE FILE CONTENTS]
---

Critical check:
- Does it meet all requirements?
- Any submission-blocking issues?
- Documentation complete?

Go/No-go for submission?
```

---

## **Bug-Specific Prompts**

### **For AI/ML Data Leakage:**
```
Review this preprocessing pipeline for data leakage:

```python
[PASTE PREPROCESSING CODE]
```

Specifically check:
- Is scaler fitted ONLY on training data?
- Is train-test split done BEFORE any data-dependent transformations?
- Are there any temporal causality violations?
```

### **For AI/ML Tensor Shapes:**
```
Check this PyTorch loss calculation for shape issues:

```python
[PASTE LOSS CALCULATION CODE]
```

Print statements show:
- output.shape = [X, Y]
- target.shape = [A, B]

Is there a broadcasting problem? How to fix?
```

### **For Cybersec Binary Analysis:**
```
I'm analyzing a Linux binary for a CTF flag (format: gdg{...}).

Tools used:
- strings output: [PASTE]
- file output: [PASTE]
- readelf output: [PASTE]

Where should I look next? What analysis technique should I try?
```

---

## **Response Format You Want from Poe**

Ask Poe to respond like this:

```
## ✅ What's Correct
[What works]

## ❌ What's Wrong
[Issues found]

## 🔧 How to Fix
[Specific code changes]

## ⏭️ Next Step
[Immediate next action]

## 📋 Updated HANDOFF
[Complete "TO: Claude Code" section ready to paste]
```

---

## **Pro Tips**

1. **Be specific:** "Review data leakage in lines 45-60" > "Review this code"
2. **Paste code, not descriptions:** Let Poe see actual code
3. **Include error messages:** Full stack traces help
4. **Time context:** Remind Poe of deadline pressure
5. **Request format:** Ask for structured responses

---

## **Save Time Tricks**

### **Multi-Question Batching:**
```
Quick batch review:
1. Is this train-test split correct? [CODE]
2. Are these tensor shapes compatible? [CODE]  
3. Should I use .fit() or .fit_transform() here? [CODE]

Answer all 3 briefly.
```

### **Code + Question Together:**
```
I implemented [feature]. Two questions:

```python
[CODE]
```

Q1: Is the logic correct?
Q2: Any edge cases I missed?
```

---

**Ready to start? Copy the "Initial Setup Prompt" to Poe now!** 🚀
