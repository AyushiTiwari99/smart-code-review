# 🔍 Smart Code Review — OpenEnv Environment

### What if every pull request had a senior engineer reviewing it — instantly?

> An AI-powered reinforcement learning environment for the **Meta OpenEnv Hackathon**.
> Agents detect bugs, classify issues, and generate verified fixes — all under time pressure.

👉 **[Live Demo — Hugging Face Spaces](https://huggingface.co/spaces/ayushiTiwarii/smart-code-review)**
🔗 **[GitHub Repository](https://github.com/AyushiTiwari99/smart-code-review)**

### 🏆 Why This Matters
This environment evaluates not just correctness — but **reasoning quality, robustness to adversarial fixes, and real-world debugging ability**, making it a practical benchmark for production-grade AI code review agents.

---

## 📌 Environment Description & Motivation

**Smart Code Review** turns code review into a structured RL task. An agent receives buggy Python code and must: locate the bug, classify it, and submit a working fix — within a step budget.

### The problem

Code review is the **biggest bottleneck** in software development. Developers spend 6+ hours per week reviewing code, and subtle logical bugs slip through constantly.

| Tool | Catches | Misses |
|------|---------|--------|
| Linters (pylint, flake8) | Style violations | Logic errors |
| Type checkers (mypy) | Type mismatches | Correct types, wrong logic |
| Static analysis (Semgrep) | Known patterns | Novel bugs, domain logic |
| **Smart Code Review** | **Runtime bugs, edge cases, semantic errors** | — |

### Why this environment matters for AI agents

- **Deterministic grading** — no LLM-as-judge, no ambiguity
- **Partial reward signal** — smooth gradient for RL training, not binary pass/fail
- **Execution-based verification** — fixes are actually run, not string-matched
- **Temporal realism** — step budget simulates real debugging pressure
- **Adversarial robustness** — agents cannot cheat with hardcoded or trivial fixes

---

## ⚙️ OpenEnv Compliance

Fully implements the OpenEnv standard with typed **Pydantic v2** models. Behavior is **deterministic and reproducible** — same task, same inputs, same reward every time.

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset()` | `reset(task_id?: str)` | Loads task, resets step counter, returns observation |
| `step()` | `step(action: dict \| str)` | Grades action, applies penalty, returns reward + done |
| `state()` | `state()` | Returns full environment state at any point in episode |

---

## 👁️ Observation Space

`reset()` returns a structured JSON observation:

```json
{
  "task_id": "hard_missing_validation",
  "difficulty": "hard",
  "buggy_code": "def transfer(self, target, amount):\n    self.balance -= amount\n    target.balance += amount\n    return True",
  "description": "This function transfers money between two bank accounts."
}
```

---

## 🕹️ Action Space

### Submit action

`step()` accepts a structured action dict:

```json
{
  "bug_line": 7,
  "issues": ["no check for negative amount", "no check for insufficient funds"],
  "fix": "if amount <= 0 or amount > self.balance:\n    raise ValueError('Invalid transfer')\nself.balance -= amount\ntarget.balance += amount\nreturn True"
}
```

| Field | Type | Constraint |
|-------|------|------------|
| `bug_line` | `int` | **1-indexed** line number; ±1 tolerance applied in grading |
| `issues` | `list[str]` | One or more plain-language issue descriptions |
| `fix` | `str` | **Complete** corrected function body — not a diff or snippet |

### Multi-turn protocol actions

`step()` also accepts string actions for iterative debugging:

| Action | Effect | Step Cost |
|--------|--------|-----------|
| `"hint"` | Returns the bug type as a hint | 1 step |
| `"run_test"` | Runs one test case against the current fix | 1 step |
| `"submit"` | Grades the full submission and ends the episode | 1 step |

---

## 📋 Task Descriptions

### 🟢 Easy — Off-by-one Error

```python
def sum_list(numbers):
    total = 0
    for i in range(1, len(numbers)):   # Bug: skips index 0
        total += numbers[i]
    return total
```

| Property | Value |
|----------|-------|
| Bug type | `off-by-one` |
| Bug line | 3 |
| Difficulty | Simple to spot; requires understanding loop indexing |
| Free steps | 2 |

---

### 🟡 Medium — Mutable Default Argument

```python
def add_item(item, item_list=[]):   # Bug: list shared across all calls
    item_list.append(item)
    return item_list
```

| Property | Value |
|----------|-------|
| Bug type | `mutable-default-argument` |
| Bug line | 1 |
| Difficulty | Syntactically valid — only revealed through behavioral testing |
| Free steps | 3 |

---

### 🔴 Hard — Missing Validation in Bank Transfer

```python
class BankAccount:
    def transfer(self, target, amount):
        self.balance -= amount   # Bug: no validation whatsoever
        target.balance += amount
        return True
```

| Property | Value |
|----------|-------|
| Bug type | `missing-validation` |
| Bug line | 7 |
| Difficulty | **Multiple missing checks** — negative amount, insufficient funds, self-transfer. Requires domain reasoning, not just syntax analysis. |
| Free steps | 4 |

---

## 🏆 Reward Function

```
base_score   =  0.3 × issue_score + 0.2 × line_score + 0.2 × compile_score + 0.3 × test_score
final_score  =  0.85 × base_score + 0.15 × reasoning_score
penalty      =  max(0, steps_taken − free_steps) × 0.05
final_reward =  clamp(final_score − penalty, 0.1, 1.0)
```

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Issue description | 30% | Correct bug type identified (fuzzy-matched) |
| Line identification | 20% | Correct line flagged (±1 tolerance) |
| Compile check | 20% | Fixed code executes without errors |
| Test cases | 30% | Fix produces correct output on all test inputs |
| Reasoning score | 15% | Clarity and correctness of the agent's explanation |

**Partial rewards** (0.25 / 0.50 / 0.75) give agents a real learning gradient — not just 0 or 1.
**Score floor** is clamped to `0.1` — agents always receive a non-zero signal.

---

## 📊 Baseline Scores

| Task | Expected Score | Why |
|------|---------------|-----|
| `easy_off_by_one` | 0.70 – 0.85 | Most agents recognize loop indexing errors |
| `medium_mutable_default` | 0.90 – 0.95 | Common Python pitfall; well-represented in training data |
| `hard_missing_validation` | 0.50 – 0.70 | Requires identifying **multiple** missing checks; partial credit is common |

The hard task is challenging because no single fix is complete — agents must reason about negative amounts, zero amounts, insufficient funds, and self-transfer independently.

---

## 🚀 Advanced Features

### Multi-Turn Debugging Protocol
Agents can use `"hint"` and `"run_test"` actions to probe the environment before committing. Each costs 1 step — forcing agents to **trade information for time**.

### Self-Reflection Loop
The LLM agent makes **two passes** per review:
1. **Initial pass** — generates `bug_line`, `issues`, and `fix`
2. **Reflection pass** — verifies fix logic, checks edge cases, revises if needed

If reflection fails or degrades the answer, the initial response is used as a safe fallback.

### Adversarial Fix Detection
The grader actively catches gaming attempts:
- **Static checks** — flags empty fixes, unchanged code, hardcoded returns (`return 0`, `return True`, `return []`)
- **Dynamic checks** — runs adversarial inputs to catch constant-output functions
- **Penalty** — suspicious fixes receive `−0.3` before score clamping

### Time Penalty
Decisive agents score higher. Every step beyond the free allowance costs `−0.05`:

| Difficulty | Free Steps | Penalty per Extra Step |
|-----------|-----------|----------------------|
| Easy | 2 | −0.05 |
| Medium | 3 | −0.05 |
| Hard | 4 | −0.05 |

### Leaderboard & Consistency Tracking
Every episode is recorded. `env.leaderboard()` prints:

```
TASK                      AVG     MIN     MAX     PENALTY     CONSISTENCY
--------------------------------------------------------------------------
easy_off_by_one           0.85    0.72    1.00    0.05        0.91
medium_mutable_default    0.95    0.95    0.95    0.00        1.00
hard_missing_validation   0.60    0.40    0.80    0.15        0.75
```

**Consistency** = `1 − std_dev(rewards)` — stable agents beat lucky ones.

---

## 🧪 Evaluation Philosophy

This environment prioritizes **execution correctness over surface-level reasoning**.
Agents are rewarded only when fixes:

- **Compile successfully** — code must run without errors
- **Pass all test cases** — correct output on every provided input
- **Handle edge cases** — adversarial inputs are tested, not just happy paths
- **Avoid shortcuts** — hardcoded returns and trivially unchanged code are penalized

This ensures models are evaluated on **true problem-solving ability**, not pattern memorization or stylistic plausibility.

---

## 🧪 OpenEnv Execution Format

All runs follow the OpenEnv structured logging protocol. Output is **single-line, machine-readable**, and strictly formatted.

### Format

```
[START] task=<task_id> env=code_review model=<model_name>
[STEP]  step=<n> action=<action> reward=<float> done=<bool> error=<null|msg>
[END]   success=<bool> steps=<n> score=<float> rewards=<float>
```

### Example — successful easy run

```
[START] task=easy_off_by_one env=code_review model=gpt-4o
[STEP]  step=1 action=submit reward=0.95 done=true error=null
[END]   success=true steps=1 score=0.95 rewards=0.95
```

### Example — multi-turn hard run

```
[START] task=hard_missing_validation env=code_review model=gpt-4o
[STEP]  step=1 action=hint reward=0.00 done=false error=null
[STEP]  step=2 action=run_test reward=0.00 done=false error=null
[STEP]  step=3 action=submit reward=0.65 done=true error=null
[END]   success=true steps=3 score=0.65 rewards=0.65
```

**Logging rules:**
- One `[STEP]` line per action — no multi-line entries
- `reward` on intermediate steps is `0.00` until `submit`
- `done=true` only on `submit` or terminal error
- `error=null` unless an exception occurred

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   Gradio UI (app.py)             │
│           Paste code → instant review            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              inference.py                        │
│   Initial LLM call → Self-reflection pass        │
│   Structured output: bug_line, issues, fix       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           OpenEnv Environment                    │
│  ┌────────────┐  ┌──────────────┐  ┌─────────┐ │
│  │  Grader    │  │   Code       │  │  Time   │ │
│  │  (4-axis)  │  │  Verifier    │  │ Penalty │ │
│  │            │  │ (subprocess) │  │         │ │
│  └────────────┘  └──────────────┘  └─────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
smart-code-review/
├── environment.py      # OpenEnv class: reset(), step(), state()
├── models.py           # Pydantic v2 models: Observation, Action, Reward
├── tasks.py            # Task definitions (easy / medium / hard) + test cases
├── grader.py           # 4-axis weighted scorer with fuzzy bug-type matching
├── codeverifier.py     # subprocess-based sandboxed code runner (5s timeout)
├── timepenalty.py      # Step-count penalty with per-difficulty thresholds
├── inference.py        # LLM agent with self-reflection loop
├── app.py              # Gradio web interface
├── openenv.yaml        # OpenEnv spec definition
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- A Hugging Face account (primary) or OpenAI API key (local use)

### Install

```bash
git clone https://github.com/your-username/smart-code-review
cd smart-code-review
pip install -r requirements.txt
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | **Yes** (primary) | — | Hugging Face token for inference API |
| `OPENAI_API_KEY` | Optional | — | OpenAI key for local / alternative use |
| `MODEL_NAME` | Optional | `gpt-4o` | Model name to use for the agent |
| `API_BASE_URL` | Optional | HF default | Custom base URL for self-hosted models |

```bash
# Primary setup (Hugging Face)
export HF_TOKEN=your-hf-token
export MODEL_NAME=gpt-4o
export API_BASE_URL=https://api-inference.huggingface.co/v1

# Alternative (local / OpenAI)
export OPENAI_API_KEY=your-openai-key
```

---

## ▶️ Usage

### Run the agent

```bash
python inference.py                          # random task
python inference.py easy_off_by_one          # specific task
python inference.py hard_missing_validation
```

### Run the Gradio UI

```bash
python app.py
# Open http://localhost:7860
```

---

## 🐳 Deployment

### Docker

```bash
docker build -t smart-code-review .
docker run -e HF_TOKEN=your-token -p 7860:7860 smart-code-review
```

### Hugging Face Spaces

1. Push this repo to your Hugging Face account
2. Set `HF_TOKEN` (and optionally `MODEL_NAME`, `API_BASE_URL`) in **Space Secrets**
3. Space auto-launches via `app.py`

---

## 📸 Screenshots

### Agent Inference Output
![Inference](assets/inference.png)

### Leaderboard & Consistency Tracking
![Leaderboard](assets/leaderboard.png)

### Hugging Face Spaces Demo
![HF Demo](assets/hf_demo.png)

---

## 🏆 Why This Project Stands Out

| Property | What it means |
|----------|--------------|
| **Deterministic grading** | No LLM-as-judge. Code either runs and passes tests, or it doesn't. Results are fully reproducible. |
| **Adversarial robustness** | Static + dynamic checks catch hardcoded, trivial, or gamed fixes. Agents must actually solve the problem. |
| **Real-world relevance** | Simulates genuine debugging workflows — bug identification, classification, fix generation, and verification. |
| **Explainability** | Every reward is decomposed across 4 axes. Agents (and judges) can see exactly why a score was given. |
| **Training-friendly** | Smooth partial rewards and a step penalty give RL agents a dense, well-shaped gradient to learn from. |
| **Production patterns** | Sandboxed subprocess execution, temp file cleanup, 5s timeout, safe score clamping — built like real software. |

---

## 🧪 Evaluation Philosophy

This environment prioritizes **execution correctness over surface-level reasoning**.
Agents are rewarded only when fixes:

- **Compile successfully** — code must run without errors
- **Pass all test cases** — correct output on every provided input
- **Handle edge cases** — adversarial inputs are tested, not just happy paths
- **Avoid shortcuts** — hardcoded returns and trivially unchanged code are penalized

This ensures models are evaluated on **true problem-solving ability**, not pattern memorization or stylistic plausibility.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| LLM Backend | OpenAI-compatible API (HF Inference or OpenAI) |
| Code Execution | `subprocess` + `tempfile` (sandboxed, 5s timeout) |
| Data Models | Pydantic v2 |
| Frontend | Gradio |
| Deployment | Docker + Hugging Face Spaces |

---

*Code review shouldn't be a bottleneck. It should be instant, accurate, and always available.*

*Built for the Meta OpenEnv Hackathon.*
