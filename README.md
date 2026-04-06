---
title: Smart Code Review
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# 🧠 Smart Code Review — OpenEnv

An AI-powered code review environment built for the **Meta OpenEnv Hackathon**.

Agents are scored not just on *correctness* — but on *speed*, *precision*, and *code quality*.

## How to use

1. **Select a task** from the dropdown (Easy / Medium / Hard)
2. **Load Task** to see the buggy Python code
3. Either:
   - Click **Run AI Agent** to let GPT-4o automatically find and fix the bug
   - Switch to **Manual Submission** to submit your own fix

## Scoring (4 axes)

| Axis | Weight |
|---|---|
| Issue description | 30% |
| Line identification | 20% |
| Compile check | 20% |
| Test cases | 30% |

## Setup

Set `OPENAI_API_KEY` in Space Secrets to enable the AI agent.
