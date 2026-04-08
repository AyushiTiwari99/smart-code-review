"""
grader.py
=========
Grades an agent's code-review submission against a task definition.

Public API:
    grade(task, action) -> dict

Action schema:
    {
        "bug_line": int,        # line number the agent flagged
        "issues":   list[str],  # agent's description of the bug(s)
        "fix":      str,        # agent's corrected code
    }

Score breakdown:
    issue_score   0.30  — does the agent describe the right bug type?
    line_score    0.20  — did the agent flag the correct line (±1)?
    compile_score 0.20  — does the fixed code run without errors?
    test_score    0.30  — fraction of test cases passed

    base_score    = 0.3*issue + 0.2*line + 0.2*compile + 0.3*test

    reasoning_score — explanation quality (length, keyword coverage)

Final score = 0.85 * base_score + 0.15 * reasoning_score
Scores are always strictly between 0.01 and 0.99.
"""

from codeverifier import run_code, check_test_cases, run_adversarial_tests

# ── Weights ────────────────────────────────────────────────────────
WEIGHTS = {
    "issue":   0.30,
    "line":    0.20,
    "compile": 0.20,
    "test":    0.30,
}

# Scores must be STRICTLY between 0 and 1
SCORE_FLOOR = 0.01
SCORE_CEIL  = 0.99


def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return float(max(SCORE_FLOOR, min(SCORE_CEIL, score)))


# ── Public API ─────────────────────────────────────────────────────

def grade(task: dict, action: dict) -> dict:
    """
    Grade an agent submission.

    Args:
        task:   a task dict from tasks.py
        action: {"bug_line": int, "issues": list[str], "fix": str}

    Returns:
        {
            "final_reward":   float,   # strictly in (0, 1)
            "issue_score":    float,
            "line_score":     float,
            "compile_score":  float,
            "test_score":     float,
            "tests_passed":   int,
            "tests_total":    int,
            "breakdown":      dict,
        }
    """
    fix      = action.get("fix", "") or ""
    bug_line = action.get("bug_line", -1)

    raw_issues  = action.get("issues", "") or ""
    issues_list = raw_issues if isinstance(raw_issues, list) else ([raw_issues] if raw_issues else [])
    issues      = " ".join(issues_list) if issues_list else str(raw_issues)

    # ── Axis 1: issue description ───────────────────────────────────
    issue_score = _score_issues(
        agent_issues=issues,
        expected_type=task.get("bug_type", ""),
    )

    # ── Axis 2: line identification ─────────────────────────────────
    line_score = _score_line(
        agent_line=bug_line,
        correct_line=task.get("bug_line", -1),
    )

    # ── Axis 3: compile / runtime check ────────────────────────────
    compile_score, compile_output = _score_compile(fix)

    # ── Axis 4: test cases ──────────────────────────────────────────
    test_cases = task.get("test_cases", [])
    test_score, tests_passed, tests_total = _score_tests(fix, test_cases)

    reasoning_score = score_reasoning(action.get("issues", []))

    normal_passed = tests_passed == tests_total and tests_total > 0

    # ── Adversarial fix detection ───────────────────────────────────
    is_adversarial, adversarial_reason = _detect_adversarial(
        fix, task.get("code", "")
    )
    if is_adversarial:
        compile_score = 0.0
        test_score    = 0.0
        tests_passed  = 0

    # ── Final weighted sum ──────────────────────────────────────────
    final = (
        WEIGHTS["issue"]   * issue_score   +
        WEIGHTS["line"]    * line_score    +
        WEIGHTS["compile"] * compile_score +
        WEIGHTS["test"]    * test_score
    )
    final = 0.85 * final + 0.15 * reasoning_score

    try:
        _suspicious = is_suspicious_fix(task.get("code", ""), action.get("fix", ""))
        if not _suspicious and run_adversarial_tests(fix, normal_passed):
            _suspicious = True
    except Exception:
        _suspicious = False

    # ── Gaming penalty ──────────────────────────────────────────────
    if _suspicious:
        final -= 0.3

    # ── Clamp strictly between 0.01 and 0.99 ───────────────────────
    final = _clamp(final)

    return {
        "final_reward":  float(f"{final:.4f}"),
        "issue_score":   float(f"{float(issue_score):.4f}"),
        "line_score":    float(f"{float(line_score):.4f}"),
        "compile_score": float(f"{float(compile_score):.4f}"),
        "test_score":    float(f"{float(test_score):.4f}"),
        "tests_passed":     tests_passed,
        "tests_total":      tests_total,
        "reasoning_score":  float(f"{float(reasoning_score):.4f}"),
        "breakdown": {
            "issue_matched":        bool(issue_score),
            "line_correct":         bool(line_score),
            "code_compiles":        bool(compile_score),
            "compile_output":       compile_output,
            "tests_passed":         tests_passed,
            "tests_total":          tests_total,
            "adversarial":          is_adversarial,
            "adversarial_reason":   adversarial_reason,
            "suspicious_fix":       _suspicious,
            "adversarial_detected": _suspicious,
        },
    }


# ── Axis scorers ───────────────────────────────────────────────────

def _score_issues(agent_issues: str, expected_type: str) -> float:
    if not expected_type or not agent_issues:
        return 0.0

    def _norm(s: str) -> str:
        return s.lower().replace("-", " ").replace("_", " ").strip()

    agent_norm    = _norm(agent_issues)
    expected_norm = _norm(expected_type)

    if (
        expected_norm in agent_norm
        or agent_norm in expected_norm
        or agent_norm == expected_norm
    ):
        return 1.0

    return 0.0


def _score_line(agent_line: int, correct_line: int) -> float:
    try:
        agent_line   = int(agent_line)
        correct_line = int(correct_line)
    except (TypeError, ValueError):
        return 0.0

    return 1.0 if abs(agent_line - correct_line) <= 1 else 0.0


def _score_compile(fix: str) -> tuple[float, str]:
    if not fix.strip():
        return 0.0, "ERROR: empty fix"

    success, output = run_code(fix, "", raw_script=True)
    return (1.0 if success else 0.0), output


def score_reasoning(issues: list) -> float:
    if not issues:
        return 0.0

    score = 0.0
    for issue in issues:
        text = str(issue).lower()
        if len(issue) > 25:
            score += 0.4
        if any(word in text for word in [
            "error", "bug", "missing", "incorrect", "validation"
        ]):
            score += 0.3
        if any(word in text for word in [
            "index", "loop", "default", "condition", "check"
        ]):
            score += 0.3

    return min(score / len(issues), 1.0)


def _score_reasoning(issues_list: list, expected_type: str) -> float:
    if not issues_list:
        return 0.0

    non_empty = [str(i).strip() for i in issues_list if str(i).strip()]
    if not non_empty:
        return 0.0

    score = 0.4
    if len(non_empty) >= 2:
        score += 0.3
    if sum(len(i) for i in non_empty) >= 30:
        score += 0.3

    return round(score, 4)


def is_suspicious_fix(original_code: str, fixed_code: str) -> bool:
    if not fixed_code or not fixed_code.strip():
        return True
    if original_code.strip() == fixed_code.strip():
        return True
    if len(fixed_code.strip()) < 15:
        return True
    return False


def _detect_adversarial(fix: str, original_code: str) -> tuple[bool, str]:
    stripped = fix.strip()
    if not stripped:
        return True, "empty fix"
    if stripped == original_code.strip():
        return True, "fix is unchanged from original buggy code"
    if len(stripped) < 20:
        return True, "fix is suspiciously short"
    original_len = len(original_code.strip())
    if original_len > 0 and len(stripped) < original_len * 0.2:
        return True, "fix is too short relative to original code"
    return False, ""


def _score_tests(fix: str, test_cases: list) -> tuple[float, int, int]:
    total = len(test_cases)
    if not fix.strip() or not test_cases:
        return 0.0, 0, total

    score, passed, total = check_test_cases(fix, test_cases)
    ratio = (passed / total) if total > 0 else 0.0
    return round(ratio, 4), passed, total


# ── Self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from tasks import TASKS

    easy_task = next(t for t in TASKS if t["id"] == "easy_off_by_one")

    perfect = {
        "bug_line": 3,
        "issues": ["off-by-one", "range starts at 1 instead of 0"],
        "fix": (
            "def sum_list(numbers):\n"
            "    total = 0\n"
            "    for i in range(0, len(numbers)):\n"
            "        total += numbers[i]\n"
            "    return total\n"
        ),
    }

    print("=== Easy task — perfect agent ===")
    r = grade(easy_task, perfect)
    print(f"  final={r['final_reward']}  issue={r['issue_score']}  "
          f"line={r['line_score']}  compile={r['compile_score']}  "
          f"test={r['test_score']}  tests={r['tests_passed']}/{r['tests_total']}")
