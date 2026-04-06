"""
time_penalty.py
===============
Applies a time-pressure penalty to a base reward score.

Public API:
    compute_time_penalty(base_reward, steps_taken, difficulty) -> dict

Step allowances by difficulty:
    easy   -> 2 steps
    medium -> 3 steps
    hard   -> 4 steps

Penalty formula:
    penalty      = max(0, steps_taken - allowed) * 0.05
    final_reward = max(REWARD_FLOOR, min(1.0, base_reward - penalty))
"""

# ── Configuration ──────────────────────────────────────────────────
STEP_ALLOWANCES = {
    "easy":   2,
    "medium": 3,
    "hard":   4,
}

PENALTY_PER_EXTRA_STEP = 0.05
REWARD_FLOOR = 0.1


# ── Public API ─────────────────────────────────────────────────────

def compute_time_penalty(
    base_reward: float,
    steps_taken: int = 0,
    difficulty: str = "medium",
) -> dict:
    """
    Apply a time-pressure penalty to a base reward.

    Args:
        base_reward:  raw score from the grader (0.0 – 1.0)
        steps_taken:  number of step() calls the agent made (default 0)
        difficulty:   "easy", "medium", or "hard" (default "medium")

    Returns:
        {
            "final_reward":  float,  # penalised score
            "base_reward":   float,  # original score
            "steps_taken":   int,
            "steps_allowed": int,    # threshold for this difficulty
            "extra_steps":   int,    # steps over the allowance
            "penalty":       float,  # total deduction applied
        }
    """
    steps_allowed = STEP_ALLOWANCES.get(difficulty, STEP_ALLOWANCES["medium"])

    extra_steps = max(0, steps_taken - steps_allowed)
    penalty = extra_steps * PENALTY_PER_EXTRA_STEP

    final_reward = max(REWARD_FLOOR, min(1.0, base_reward - penalty))

    return {
        "final_reward":  round(float(final_reward), 4),  # type: ignore[arg-type]
        "base_reward":   round(float(base_reward), 4),   # type: ignore[arg-type]
        "steps_taken":   steps_taken,
        "steps_allowed": steps_allowed,
        "extra_steps":   extra_steps,
        "penalty":       round(float(penalty), 4),       # type: ignore[arg-type]
    }


# ── Self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (base_reward, steps_taken, difficulty, description)
        (0.80, 2, "easy",    "within allowance — no penalty"),
        (0.80, 4, "easy",    "2 extra steps on easy → -0.10"),
        (0.80, 3, "medium",  "within allowance — no penalty"),
        (0.80, 6, "medium",  "3 extra steps on medium → -0.15, floored"),
        (0.90, 4, "hard",    "within allowance — no penalty"),
        (0.90, 8, "hard",    "4 extra steps on hard → -0.20"),
        (0.50, 15, "easy",   "many extra steps — floored at REWARD_FLOOR"),
        (0.00, 10, "easy",   "zero base — floored at REWARD_FLOOR"),
        (0.80, 2, "unknown", "unknown difficulty falls back to medium=3"),
    ]

    print(f"{'Description':<45} base   steps  allowed  extra  penalty  final")
    print("-" * 95)
    for base, steps, diff, desc in cases:
        r = compute_time_penalty(base, steps, diff)
        print(
            f"{desc:<45} "
            f"{r['base_reward']:.2f}   "
            f"{r['steps_taken']:<6} "
            f"{r['steps_allowed']:<8} "
            f"{r['extra_steps']:<6} "
            f"{r['penalty']:.4f}   "
            f"{r['final_reward']:.4f}"
        )