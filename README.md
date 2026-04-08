---
title: Smart Code Review
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# 🧠 Smart Code Review — OpenEnv

An AI-powered code review environment built for the **Meta OpenEnv Hackathon**.
Agents are scored not just on *correctness* — but on *speed*, *precision*, and *code quality*.

---

## Environment Overview

Code review is one of the most time-consuming tasks in software development.
**Smart Code Review** turns code review into a structured reinforcement learning task.
An AI agent is shown buggy Python code, must identify the bug, classify it, and submit
a working fix — all within a step budget.

---

## Observation Space

After `reset()`, the agent receives:

```python
{
    "task_id":     str,   # unique task identifier
    "difficulty":  str,   # "easy", "medium", or "hard"
    "title":       str,   # short task name
    "description": str,   # plain-text task instructions
    "buggy_code":  str,   # the Python code containing the bug
    "done":        bool,  # always False after reset
}
```

## Action Space

Agents submit a structured dict to `step()`:

```python
{
    "bug_line": int,        # line number of the bug (1-indexed)
    "issues":   list[str],  # list of bug descriptions
    "fix":      str,        # corrected Python code
}
```

## Reward

`step()` always returns:

```python
{
    "state":  dict,   # current task state
    "reward": float,  # final score in [0.10, 1.00]
    "done":   bool,   # True after submission
}
```

Reward formula:
```
final_score  =  0.3 × issue_score
             +  0.2 × line_score
             +  0.2 × compile_score
             +  0.3 × test_score

penalty      =  max(0, steps_taken - allowed) × 0.05
final_reward =  max(0.1, min(1.0, final_score - penalty))
```

---

## Tasks

### 🟢 Easy — Off-by-one Error
- **Bug type:** `off-by-one`
- **Bug line:** 3
- **Free steps:** 2

### 🟡 Medium — Mutable Default Argument
- **Bug type:** `mutable-default-argument`
- **Bug line:** 1
- **Free steps:** 3

### 🔴 Hard — Missing Validation in Bank Transfer
- **Bug type:** `missing-validation`
- **Bug line:** 7
- **Free steps:** 4

---

## Baseline Performance Scores

| Task | Baseline Score |
|---|---|
| `easy_off_by_one` | ~0.70 |
| `medium_mutable_default` | ~0.95 |
| `hard_missing_validation` | ~0.50 – 0.70 |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Load a task, returns observation |
| `/step` | POST | Submit action, returns reward |
| `/state` | GET | Returns current environment state |
| `/gradio` | GET | Web UI for human interaction |

---

## Setup & Usage

### Environment Variables (Secrets)

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for the AI agent |
| `HF_TOKEN` | Yes | Hugging Face API token |
| `API_BASE_URL` | Optional | Custom API base URL (default: OpenAI) |
| `MODEL_NAME` | Optional | Model to use (default: gpt-4o) |

### Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
python app.py
```

### Run with Docker

```bash
docker build -t smart-code-review .
docker run -e OPENAI_API_KEY=your-key -p 7860:7860 smart-code-review
```

### Run inference script

```bash
python inference.py                        # random task
python inference.py easy_off_by_one        # specific task
```

---

## Project Structure

```
smart-code-review/
├── app.py            # FastAPI + Gradio web UI
├── environment.py    # OpenEnv class: reset(), step(), state()
├── tasks.py          # 3 tasks (easy / medium / hard)
├── grader.py         # 4-axis weighted scorer
├── codeverifier.py   # subprocess-based code runner
├── timepenalty.py    # step-count penalty
├── inference.py      # LLM agent using OpenAI API
├── openenv.yaml      # OpenEnv metadata
├── pyproject.toml    # Python package config
├── requirements.txt  # Dependencies
└── Dockerfile        # Container definition
```

---

*Built for the Meta OpenEnv Hackathon.*
