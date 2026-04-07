"""
inference.py
============
Runs a single episode of CodeReviewEnv with an optional LLM agent.

Usage:
    python inference.py                        # random task
    python inference.py easy_off_by_one        # specific task
    OPENAI_API_KEY=sk-... python inference.py  # with LLM
"""

import os
import re
import sys
import json

from environment import CodeReviewEnv

# ── Environment variables ───────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

# ── LLM prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Python code reviewer. Given buggy Python code, you must:
1. Identify the line number of the PRIMARY bug (1-indexed)
2. List ALL issues found — including every missing validation:
   - negative or zero value checks
   - self-referential operation checks (e.g. self-transfer)
   - boundary conditions (empty input, overflow, underflow)
   - multiple bugs or missing guards within the same function
3. Provide a corrected version of the full code that fixes every issue

Respond ONLY with valid JSON in this exact format:
{"bug_line": <int>, "issues": [<str>, ...], "fix": "<corrected code>"}
"""


def _call_llm(buggy_code: str) -> dict:
    """Call OpenAI API and return a parsed action dict."""
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=API_BASE_URL,
    )

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
    except (json.JSONDecodeError, ValueError):
        return {"bug_line": 0, "issues": [], "fix": ""}


# ── Self-reflection ─────────────────────────────────────────────────

def _reflect_action(buggy_code: str, first_action: dict) -> dict:
    """Ask the LLM to review its own fix and return an improved action."""
    try:
        import openai

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=API_BASE_URL,
        )

        bug_line = first_action.get("bug_line", 0)
        issues   = "\n".join(f"- {i}" for i in (first_action.get("issues") or []))
        fix      = first_action.get("fix", "")

        reflection_prompt = (
            "You are reviewing your previous code fix.\n\n"
            f"Original buggy code:\n{buggy_code}\n\n"
            f"Your previous response:\n"
            f"Bug line: {bug_line}\n"
            f"Issues: {issues}\n"
            f"Fix:\n{fix}\n\n"
            "Now:\n"
            "- Verify the bug_line is correct (1-indexed)\n"
            "- Check if the fix fully resolves the bug\n"
            "- Check for missed edge cases (empty input, boundary values)\n"
            "- Improve the fix if necessary\n\n"
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


# ── Fallback action ─────────────────────────────────────────────────

def _fallback(buggy_code: str) -> dict:
    return {"bug_line": 0, "issues": [], "fix": buggy_code}


# ── Agent ───────────────────────────────────────────────────────────

def get_action(buggy_code: str) -> dict:
    """Return an action dict, using LLM if available, else fallback."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[INFO] No OPENAI_API_KEY found — using fallback action.")
        return _fallback(buggy_code)

    _VALIDATION_KEYWORDS = {"transfer", "balance", "amount"}
    if any(kw in buggy_code for kw in _VALIDATION_KEYWORDS):
        buggy_code += "\n# Hint: check for negative/zero values, self-transfer, and insufficient funds."

    try:
        initial_action  = _call_llm(buggy_code)
        refined_action  = initial_action
        used_reflection = False
        try:
            refined_action  = _reflect_action(buggy_code, initial_action)
            used_reflection = True
        except Exception as re_:
            print(f"[WARN] Reflection failed ({re_}) — using initial action.")

        action   = refined_action
        bug_line = int(action.get("bug_line", 0))
        issues   = action.get("issues", [])
        fix      = action.get("fix", buggy_code)
        if not isinstance(issues, list):
            issues = [str(issues)]
        if not isinstance(fix, str):
            fix = buggy_code
        return {
            "bug_line": bug_line,
            "issues": issues,
            "fix": fix,
            "used_reflection": used_reflection,
            "_initial_action": initial_action,
        }
    except Exception as e:
        print(f"[WARN] LLM call failed ({e}) — using fallback action.")
        return _fallback(buggy_code)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    task_id = sys.argv[1] if len(sys.argv) > 1 else None

    env = CodeReviewEnv()

    try:
        obs = env.reset(task_id)
    except KeyError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    buggy_code = obs.get("buggy_code", "")

    # ── [START] log ────────────────────────────────────────────────
    print("[START]")
    print(json.dumps({
        "task_id":    obs.get("task_id"),
        "difficulty": obs.get("difficulty"),
        "model":      MODEL_NAME,
        "api_base":   API_BASE_URL,
    }))

    action         = get_action(buggy_code)
    initial_action = action.pop("_initial_action", None)
    used_reflection = action.pop("used_reflection", False)

    # ── [STEP] log ────────────────────────────────────────────────
    print("[STEP]")
    print(json.dumps({
        "bug_line":       action.get("bug_line"),
        "issues":         action.get("issues"),
        "fix_length":     len(action.get("fix", "")),
        "used_reflection": used_reflection,
    }))

    try:
        result = env.step(action)
    except Exception as e:
        print(f"[WARN] env.step failed ({e}) — using default result.")
        result = {"state": {}, "reward": 0.0, "done": True}

    lr = env._last_result or {}

    # ── [END] log ─────────────────────────────────────────────────
    print("[END]")
    print(json.dumps({
        "task_id":       obs.get("task_id"),
        "reward":        result.get("reward", 0.0),
        "issue_score":   lr.get("issue_score", 0.0),
        "line_score":    lr.get("line_score", 0.0),
        "compile_score": lr.get("compile_score", 0.0),
        "test_score":    lr.get("test_score", 0.0),
        "tests_passed":  lr.get("tests_passed", 0),
        "tests_total":   lr.get("tests_total", 0),
        "penalty":       lr.get("penalty", 0.0),
        "done":          result.get("done", True),
    }))

    env.leaderboard()


if __name__ == "__main__":
    main()
