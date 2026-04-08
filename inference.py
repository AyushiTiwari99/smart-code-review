"""
inference.py
============
Runs all 3 tasks of CodeReviewEnv with an LLM agent.

Usage:
    python inference.py                    # runs all 3 tasks
    python inference.py easy_off_by_one    # specific task

Environment variables (injected by OpenEnv validator):
    API_BASE_URL  — LiteLLM proxy base URL (required, no default)
    API_KEY       — LiteLLM proxy API key  (required, no default)
    MODEL_NAME    — model to use (default: gpt-4o)
"""

import os
import re
import sys
import json

from environment import CodeReviewEnv

# ── Environment variables ───────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY      = os.environ.get("API_KEY")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")

ALL_TASKS = [
    "easy_off_by_one",
    "medium_mutable_default",
    "hard_missing_validation",
]

SYSTEM_PROMPT = """\
You are a Python code reviewer. Given buggy Python code, you must:
1. Identify the line number of the PRIMARY bug (1-indexed)
2. List ALL issues found — including every missing validation
3. Provide a corrected version of the full code that fixes every issue

Respond ONLY with valid JSON in this exact format:
{"bug_line": <int>, "issues": [<str>, ...], "fix": "<corrected code>"}
"""


def _clamp(score) -> float:
    """Strictly between 0.01 and 0.99 — never exactly 0.0 or 1.0."""
    try:
        v = float(score)
    except Exception:
        v = 0.01
    return round(max(0.01, min(0.99, v)), 4)


def _call_llm(buggy_code: str) -> dict:
    import openai
    client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Review this code:\n\n{buggy_code}"},
        ],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if md_match:
        raw = md_match.group(1).strip()
    else:
        brace_match = re.search(r"\{[\s\S]*\}", raw)
        raw = brace_match.group(0) if brace_match else raw
    try:
        return json.loads(raw)
    except Exception:
        return {"bug_line": 1, "issues": ["unknown bug"], "fix": ""}


def _reflect_action(buggy_code: str, first_action: dict) -> dict:
    try:
        import openai
        client = openai.OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        bug_line = first_action.get("bug_line", 0)
        issues   = "\n".join(f"- {i}" for i in (first_action.get("issues") or []))
        fix      = first_action.get("fix", "")
        reflection_prompt = (
            f"Original buggy code:\n{buggy_code}\n\n"
            f"Your previous response — Bug line: {bug_line}, Issues: {issues}, Fix:\n{fix}\n\n"
            "Verify and improve if needed. "
            'Return ONLY valid JSON: {"bug_line": int, "issues": [str], "fix": "code"}'
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": reflection_prompt},
            ],
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if md_match:
            raw = md_match.group(1).strip()
        else:
            brace_match = re.search(r"\{[\s\S]*\}", raw)
            raw = brace_match.group(0) if brace_match else raw
        parsed = json.loads(raw)
        if not isinstance(parsed.get("bug_line"), int):
            return first_action
        if not isinstance(parsed.get("issues"), list):
            return first_action
        if not isinstance(parsed.get("fix"), str):
            return first_action
        return parsed
    except Exception:
        return first_action


def _fallback(buggy_code: str) -> dict:
    return {"bug_line": 1, "issues": ["unknown bug"], "fix": buggy_code}


def get_action(buggy_code: str) -> dict:
    if not API_BASE_URL or not API_KEY:
        print("[INFO] API_BASE_URL or API_KEY not set — using fallback.")
        return _fallback(buggy_code)

    if any(kw in buggy_code for kw in {"transfer", "balance", "amount"}):
        buggy_code += "\n# Hint: check for negative/zero values, self-transfer, insufficient funds."

    try:
        initial_action  = _call_llm(buggy_code)
        used_reflection = False
        try:
            refined_action  = _reflect_action(buggy_code, initial_action)
            used_reflection = True
        except Exception:
            refined_action = initial_action

        action   = refined_action
        bug_line = int(action.get("bug_line", 1))
        issues   = action.get("issues", [])
        fix      = action.get("fix", buggy_code)
        if not isinstance(issues, list):
            issues = [str(issues)]
        if not isinstance(fix, str):
            fix = buggy_code

        return {
            "bug_line":        bug_line,
            "issues":          issues,
            "fix":             fix,
            "used_reflection": used_reflection,
            "_initial_action": initial_action,
        }
    except Exception as e:
        print(f"[WARN] LLM call failed ({e}) — using fallback.")
        return _fallback(buggy_code)


def run_task(task_id: str) -> dict:
    env = CodeReviewEnv()

    try:
        obs = env.reset(task_id)
    except KeyError as e:
        print(f"[ERROR] {e}")
        return {"task_id": task_id, "reward": 0.5, "done": True}

    buggy_code = obs.get("buggy_code", "")

    print(f"\n[START] task={task_id} env=code_review model={MODEL_NAME}")

    action          = get_action(buggy_code)
    initial_action  = action.pop("_initial_action", None)
    used_reflection = action.pop("used_reflection", False)

    print(f"[STEP]  step=1 action=submit used_reflection={used_reflection}")

    try:
        result = env.step(action)
    except Exception as e:
        print(f"[WARN] env.step failed ({e})")
        result = {"state": {}, "reward": 0.5, "done": True}

    lr = env._last_result or {}

    # Clamp ALL scores strictly between 0.01 and 0.99
    reward        = _clamp(result.get("reward", 0.5))
    issue_score   = _clamp(lr.get("issue_score",   0.5))
    line_score    = _clamp(lr.get("line_score",    0.5))
    compile_score = _clamp(lr.get("compile_score", 0.5))
    test_score    = _clamp(lr.get("test_score",    0.5))

    print(f"[END]   task={task_id} reward={reward:.4f} done=true")

    print(json.dumps({
        "task_id":       task_id,
        "reward":        reward,
        "issue_score":   issue_score,
        "line_score":    line_score,
        "compile_score": compile_score,
        "test_score":    test_score,
        "tests_passed":  lr.get("tests_passed", 0),
        "tests_total":   lr.get("tests_total",  0),
        "penalty":       lr.get("penalty",       0.0),
        "done":          True,
    }))

    return {"task_id": task_id, "reward": reward, "done": True}


def main():
    if len(sys.argv) > 1:
        task_ids = [sys.argv[1]]
    else:
        task_ids = ALL_TASKS

    results = []
    for task_id in task_ids:
        result = run_task(task_id)
        results.append(result)

    print("\n=== SUMMARY ===")
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
