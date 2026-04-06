"""
environment.py
==============
OpenEnv environment for code-review tasks.

How to use:
    env = CodeReviewEnv()

    obs    = env.reset()              # load a random task
    result = env.step(action)         # submit your answer
    info   = env.state()              # inspect current state at any time

Action format for step():
    {
        "bug_line": int,        # line number you think contains the bug
        "issues":   list[str],  # list of bug descriptions
        "fix":      str,        # your corrected version of the code
    }

step() always returns:
    {
        "state":  dict,   # full environment state (same as state())
        "reward": float,  # final score after time penalty (0.0 – 1.0)
        "done":   bool,   # always True after a submission
    }

Time penalty:
    Each difficulty has a free-step allowance:
        easy=2, medium=3, hard=4
    Every step beyond that costs 0.05, with a minimum reward of 0.1.
    (Exception: if the base reward is 0.0 it stays 0.0 — no inflation.)
"""

import random

from tasks import TASKS, TASK_INDEX
from grader import grade
from timepenalty import compute_time_penalty
from codeverifier import run_code


# ── Step allowances — mirrors time_penalty.py ──────────────────────
STEP_ALLOWANCES = {"easy": 2, "medium": 3, "hard": 4}


# ══════════════════════════════════════════════════════════════════════
class CodeReviewEnv:
    """
    OpenEnv-compatible environment for code-review evaluation.

    Attributes (read-only — do not set directly):
        _task         current task dict, or None before first reset()
        _steps_taken  how many times step() has been called this episode
        _done         True once the agent has submitted an answer
        _last_result  grading + penalty details from the last step()
    """

    def __init__(self):
        self._task: dict | None        = None
        self._steps_taken              = 0
        self._done                     = False
        self._last_result: dict | None = None
        self._history: list            = []

    # ── reset() ────────────────────────────────────────────────────
    def reset(self, task_id: str | None = None) -> dict:
        """
        Start a new episode.

        Args:
            task_id: a specific task id (see tasks.py), or None to pick
                     one at random.

        Returns:
            Observation dict — the buggy code and task metadata.
        """
        # Pick the task
        if task_id is not None:
            if task_id not in TASK_INDEX:
                raise KeyError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available ids: {list(TASK_INDEX.keys())}"
                )
            self._task = TASK_INDEX[task_id]
        else:
            self._task = random.choice(TASKS)   # random task each episode

        # Reset episode state
        self._steps_taken = 0
        self._done        = False
        self._last_result = None

        return self._build_observation()

    # ── step() ─────────────────────────────────────────────────────
    def step(self, action: dict) -> dict:
        """
        Submit an answer.  Every call counts as one step.

        Args:
            action: {"bug_line": int, "issues": list[str], "fix": str}
                    Missing or wrong-typed keys are handled safely.

        Returns:
            {"state": dict, "reward": float, "done": True}

        Raises:
            RuntimeError: if reset() has not been called yet.
        """
        # Guard: must call reset() first
        if self._task is None:
            raise RuntimeError(
                "No task loaded. Call reset() before step()."
            )

        # Guard: episode already finished — return cached result
        if self._done:
            last = getattr(self, "_last_result", None) or {}
            return {
                "state":  self._task,
                "reward": float(max(0.0, min(1.0, last.get("final_reward", 0.0)))),
                "done":   True,
            }

        # Count this step (all action types consume a step)
        self._steps_taken += 1

        # ── Action type dispatch ────────────────────────────────────
        action_type = action.get("type", "submit") if isinstance(action, dict) else "submit"

        if action_type == "hint":
            return {
                "state":  self._task,
                "reward": 0.0,
                "done":   False,
                "hint":   (self._task or {}).get("bug_type", "unknown"),
            }

        if action_type == "run_test":
            test_cases = (self._task or {}).get("tests", [])
            if not test_cases:
                return {"state": self._task, "reward": 0.0, "done": False}
            case   = test_cases[0]
            passed = run_code(action.get("fix", ""), case["input"])[0]
            return {
                "state":       self._task,
                "reward":      0.0,
                "done":        False,
                "test_result": "pass" if passed else "fail",
            }

        # default: "submit" — proceed with grading

        # Validate and normalise the action
        safe_action = _validate_action(action)

        # Grade the submission (returns scores 0.0 – 1.0 per axis)
        grade_result = grade(self._task, safe_action)
        base_reward  = grade_result.get("final_reward", 0.0)   # weighted sum

        # Apply time-pressure penalty
        task = self._task or {}
        penalty_result = compute_time_penalty(
            base_reward=base_reward,
            steps_taken=self._steps_taken,
            difficulty=task.get("difficulty", "medium"),
        )
        final_reward = penalty_result.get("final_reward", 0.0)
        final_reward = float(max(0.0, min(1.0, final_reward)))  # always in [0.0, 1.0]

        # Store everything for state() and future calls
        self._done = True
        self._last_result = {
            **grade_result,
            **penalty_result,
            "final_reward": final_reward,  # clamped penalty result takes priority
            "steps_taken":  self._steps_taken,
        }

        self._history.append({
            "task_id":     (self._task or {}).get("id"),
            "reward":      final_reward,
            "penalty":     penalty_result.get("penalty", 0.0),
            "base_reward": base_reward,
        })

        last = getattr(self, "_last_result", None) or {}
        return {
            "state":  self._task,
            "reward": float(max(0.0, min(1.0, last.get("final_reward", 0.0)))),
            "done":   True,
        }

    # ── state() ────────────────────────────────────────────────────
    def state(self) -> dict:
        """
        Return the current environment state.

        Safe to call at any time — before reset(), between reset() and
        step(), and after the episode ends.

        Returns:
            {
                "task_id":       str | None,
                "difficulty":    str | None,
                "steps_taken":   int,
                "steps_allowed": int | None,
                "done":          bool,
                "result":        dict | None,   # None until step() is called
            }
        """
        task          = getattr(self, "_task", None)
        difficulty    = task.get("difficulty") if task else None
        steps_allowed = STEP_ALLOWANCES.get(difficulty, 0) if difficulty else 0

        return {
            "task_id":       task.get("id") if task else None,
            "difficulty":    difficulty,
            "steps_taken":   getattr(self, "_steps_taken", 0),
            "steps_allowed": steps_allowed,
            "done":          getattr(self, "_done", False),
            "result":        getattr(self, "_last_result", {}) or {},
        }

    # ── leaderboard() ──────────────────────────────────────────────
    def leaderboard(self) -> list[dict]:
        """
        Summarise self._history grouped by task_id.

        Returns a list of dicts (one per task), sorted by mean_reward descending:
            task_id, mean_reward, min_reward, max_reward,
            avg_penalty, consistency_score
        """
        try:
            if not self._history:
                print("No runs yet")
                return []

            def _std(values: list[float]) -> float:
                n = len(values)
                if n < 2:
                    return 0.0
                mean = sum(values) / n
                variance = sum((v - mean) ** 2 for v in values) / n
                return variance ** 0.5 if variance > 0 else 0.0

            def _f(v: float) -> float:
                return float(f"{v:.4f}")

            # Group entries by task_id
            groups: dict[str, list] = {}
            for entry in self._history:
                tid = str(entry.get("task_id") or "unknown")
                groups.setdefault(tid, []).append(entry)

            rows = []
            for tid, entries in groups.items():
                rewards   = [float(e.get("reward", 0.0))  for e in entries] or [0.0]
                penalties = [float(e.get("penalty", 0.0)) for e in entries] or [0.0]

                n = len(rewards)
                mean_reward       = sum(rewards) / n if n > 0 else 0.0
                avg_penalty       = sum(penalties) / len(penalties) if penalties else 0.0
                consistency_score = max(0.0, 1.0 - _std(rewards))

                rows.append({
                    "task_id":           tid,
                    "mean_reward":       _f(mean_reward),
                    "min_reward":        _f(min(rewards)),
                    "max_reward":        _f(max(rewards)),
                    "avg_penalty":       _f(avg_penalty),
                    "consistency_score": _f(consistency_score),
                })

            rows.sort(key=lambda r: r["mean_reward"], reverse=True)

            # ── Print formatted table ───────────────────────────────
            header = (
                "TASK".ljust(26) +
                "AVG".ljust(8) +
                "MIN".ljust(8) +
                "MAX".ljust(8) +
                "PENALTY".ljust(12) +
                "CONSISTENCY".ljust(12)
            )
            print(header)
            print("-" * 74)
            for r in rows:
                print(
                    str(r["task_id"]).ljust(26) +
                    f"{float(r['mean_reward']):.2f}".ljust(8) +
                    f"{float(r['min_reward']):.2f}".ljust(8) +
                    f"{float(r['max_reward']):.2f}".ljust(8) +
                    f"{float(r['avg_penalty']):.2f}".ljust(12) +
                    f"{float(r['consistency_score']):.2f}".ljust(12)
                )

            return rows

        except Exception:
            print("No runs yet")
            return []

    # ── private helpers ────────────────────────────────────────────
    def _build_observation(self) -> dict:
        """Return what the agent sees at the start of an episode."""
        task = self._task
        assert task is not None
        return {
            "task_id":     task.get("id"),
            "difficulty":  task.get("difficulty", "medium"),
            "title":       task.get("title", ""),
            "description": task.get("description", ""),
            "buggy_code":  task.get("code", ""),
            "done":        False,
        }


# ══════════════════════════════════════════════════════════════════════
# Action validation
# ══════════════════════════════════════════════════════════════════════

def _validate_action(action: dict) -> dict:
    """
    Safely coerce the raw action into the schema grader.grade() expects.

    Always returns:
        {"bug_line": int, "issues": list, "fix": str}

    This function:
      - coerces non-dict input to {}
      - uses setdefault to fill missing keys
      - converts bug_line to int safely
      - keeps issues as a list, wrapping a bare str in one
      - never raises — bad values become safe defaults
    """
    # Guard: if action is not a dict, treat it as empty
    if not isinstance(action, dict):
        action = {}

    # Apply defaults for all expected keys
    action.setdefault("bug_line", -1)
    action.setdefault("issues", "")
    action.setdefault("fix", "")

    # ── bug_line: coerce to int ────────────────────────────────────
    try:
        bug_line = int(action["bug_line"])
    except (TypeError, ValueError):
        bug_line = -1

    # ── issues: always return a list ──────────────────────────────
    raw_issues = action.get("issues")
    if isinstance(raw_issues, list):
        issues = [str(i) for i in raw_issues if i is not None]
    elif isinstance(raw_issues, str) and raw_issues:
        issues = [raw_issues]
    else:
        issues = []

    # ── fix: must be str ──────────────────────────────────────────
    fix = action["fix"] if isinstance(action["fix"], str) else ""

    return {
        "bug_line": bug_line,
        "issues":   issues,
        "fix":      fix,
    }


# ══════════════════════════════════════════════════════════════════════
# Demo — run `python environment.py` to see everything in action
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import json

    env = CodeReviewEnv()

    # ── 1. Perfect agent on the easy task ─────────────────────────
    print("=" * 62)
    print("EASY — perfect answer, 1 step (allowance=2, no penalty)")
    print("=" * 62)

    obs = env.reset("easy_off_by_one")
    print(f"Task : {obs['title']}")
    print(f"Code :\n{obs['buggy_code']}")

    result = env.step({
        "bug_line": 3,
        "issues":   ["off-by-one", "range starts at 1 instead of 0"],
        "fix": (
            "def sum_list(numbers):\n"
            "    total = 0\n"
            "    for i in range(0, len(numbers)):\n"
            "        total += numbers[i]\n"
            "    return total\n"
        ),
    })
    r = getattr(env, "_last_result", {}) or {}
    print(f"reward={result.get('reward', 0)}  "
          f"base={r.get('base_reward', 0)}  penalty={r.get('penalty', 0)}")
    print(f"scores → issue={r.get('issue_score', 0)}  line={r.get('line_score', 0)}  "
          f"compile={r.get('compile_score', 0)}  test={r.get('test_score', 0)}")
    print(f"tests  → {r.get('tests_passed', 0)}/{r.get('tests_total', 0)} passed")

    # ── 2. Slow agent — 5 steps total on easy (3 extra) ───────────
    print()
    print("=" * 62)
    print("EASY — correct fix but 5 steps (3 extra → penalty 0.15)")
    print("=" * 62)

    obs = env.reset("easy_off_by_one")

    # Simulate 4 prior steps by setting the counter directly (demo only)
    env._steps_taken = 4

    result = env.step({
        "bug_line": 3,
        "issues":   ["off-by-one"],
        "fix": (
            "def sum_list(numbers):\n"
            "    total = 0\n"
            "    for i in range(0, len(numbers)):\n"
            "        total += numbers[i]\n"
            "    return total\n"
        ),
    })
    r = getattr(env, "_last_result", {}) or {}
    print(f"reward={result.get('reward', 0)}  "
          f"base={r.get('base_reward', 0)}  "
          f"steps={r.get('steps_taken', 0)}/{r.get('steps_allowed', 0)}  "
          f"penalty={r.get('penalty', 0)}")

    # ── 3. Blank action — all keys missing ────────────────────────
    print()
    print("=" * 62)
    print("MEDIUM — blank action {} (all keys missing → safe defaults)")
    print("=" * 62)

    obs = env.reset("medium_mutable_default")
    result = env.step({})
    r = getattr(env, "_last_result", {}) or {}
    print(f"reward={result.get('reward', 0)}  base={r.get('base_reward', 0)}")
    print(f"scores → issue={r.get('issue_score', 0)}  line={r.get('line_score', 0)}  "
          f"compile={r.get('compile_score', 0)}  test={r.get('test_score', 0)}")

    # ── 4. Full fix on the hard task ──────────────────────────────
    print()
    print("=" * 62)
    print("HARD — complete fix with all three validations")
    print("=" * 62)

    obs = env.reset("hard_missing_validation")
    result = env.step({
        "bug_line": 7,
        "issues": [
            "missing-validation",
            "no check for negative amount",
            "no check for insufficient funds",
            "no check for self-transfer",
        ],
        "fix": (
            "class BankAccount:\n"
            "    def __init__(self, owner, balance):\n"
            "        self.owner = owner\n"
            "        self.balance = balance\n"
            "\n"
            "    def transfer(self, target, amount):\n"
            "        if amount <= 0:\n"
            "            raise ValueError('Amount must be positive')\n"
            "        if self.balance < amount:\n"
            "            raise ValueError('Insufficient funds')\n"
            "        if self is target:\n"
            "            raise ValueError('Cannot transfer to self')\n"
            "        self.balance -= amount\n"
            "        target.balance += amount\n"
            "        return True\n"
            "\n"
            "    def get_balance(self):\n"
            "        return self.balance\n"
        ),
    })
    r = getattr(env, "_last_result", {}) or {}
    print(f"reward={result.get('reward', 0)}  "
          f"base={r.get('base_reward', 0)}  penalty={r.get('penalty', 0)}")
    print(f"scores → issue={r.get('issue_score', 0)}  line={r.get('line_score', 0)}  "
          f"compile={r.get('compile_score', 0)}  test={r.get('test_score', 0)}")
    print(f"tests  → {r.get('tests_passed', 0)}/{r.get('tests_total', 0)} passed")


    # ── 5. Random task (no task_id) ───────────────────────────────
    print()
    print("=" * 62)
    print("RANDOM — reset() with no task_id")
    print("=" * 62)
    obs = env.reset()
    print(f"Got task : '{obs['task_id']}'  [{obs['difficulty']}]")
    snapshot = {k: v for k, v in env.state().items() if k != "result"}
    print(f"state()  → {json.dumps(snapshot, indent=2)}")