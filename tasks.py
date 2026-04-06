"""
tasks.py
========
Three code-review tasks for the CodeReviewEnv.

Each task has:
  - id           : unique string key
  - difficulty   : "easy" | "medium" | "hard"
  - title        : short human-readable name
  - description  : instructions shown to the agent
  - code         : the buggy code the agent must review
  - bug_line     : correct line number of the bug (1-indexed)
  - bug_type     : canonical bug category string
  - test_cases   : list of {"input": ..., "output": ...} dicts
                   set raw_script=True when input is a multi-line script
"""

from typing import List, cast

TASKS = [

    # ──────────────────────────────────────────────────────────────────
    # EASY: Off-by-one error in range()
    # ──────────────────────────────────────────────────────────────────
    {
        "id": "easy_off_by_one",
        "difficulty": "easy",
        "title": "Off-by-one error in array sum",
        "description": (
            "Review the Python function below. "
            "Find the bug, classify it, explain what is wrong, "
            "and provide corrected code."
        ),
        "code": (
            "def sum_list(numbers):\n"
            "    total = 0\n"
            "    for i in range(1, len(numbers)):\n"
            "        total += numbers[i]\n"
            "    return total\n"
        ),
        # Bug: range(1, ...) skips index 0, so the first element is
        # always excluded from the sum.
        "bug_line": 3,
        "bug_type": "off-by-one",
        "test_cases": [
            {"input": "sum_list([1, 2, 3])", "output": "6"},
            {"input": "sum_list([])",         "output": "0"},
            {"input": "sum_list([10])",       "output": "10"},
            {"input": "sum_list([-1, 1])",    "output": "0"},
            {"input": "sum_list([5, 5, 5])",  "output": "15"},
        ],
    },

    # ──────────────────────────────────────────────────────────────────
    # MEDIUM: Mutable default argument
    # ──────────────────────────────────────────────────────────────────
    {
        "id": "medium_mutable_default",
        "difficulty": "medium",
        "title": "Mutable default argument bug",
        "description": (
            "Review the Python function below. "
            "It behaves unexpectedly when called multiple times. "
            "Find the bug, classify it, explain why it happens, "
            "and provide corrected code."
        ),
        "code": (
            "def add_item(item, item_list=[]):\n"
            "    item_list.append(item)\n"
            "    return item_list\n"
        ),
        # Bug: the default list [] is created once at function definition
        # time and shared across all calls that omit item_list.
        # Fix: use None as the default and create a new list inside the body.
        "bug_line": 1,
        "bug_type": "mutable-default-argument",
        "test_cases": [
            # Each test runs in its own subprocess — no state leaks between cases.
            {"input": "add_item('a')",               "output": "['a']"},
            {"input": "add_item('b')",               "output": "['b']"},
            {"input": "add_item('x', ['existing'])", "output": "['existing', 'x']"},
            {"input": "add_item('z', [])",           "output": "['z']"},
        ],
    },

    # ──────────────────────────────────────────────────────────────────
    # HARD: Missing validation in BankAccount.transfer()
    # ──────────────────────────────────────────────────────────────────
    {
        "id": "hard_missing_validation",
        "difficulty": "hard",
        "title": "Missing validation in bank transfer",
        "description": (
            "Review the BankAccount class below. "
            "The transfer() method is missing several safety checks. "
            "Find ALL missing validations, classify the bug type, "
            "explain each issue, and provide fully corrected code."
        ),
        "code": (
            "class BankAccount:\n"
            "    def __init__(self, owner, balance):\n"
            "        self.owner = owner\n"
            "        self.balance = balance\n"
            "\n"
            "    def transfer(self, target, amount):\n"
            "        self.balance -= amount\n"
            "        target.balance += amount\n"
            "        return True\n"
            "\n"
            "    def get_balance(self):\n"
            "        return self.balance\n"
        ),
        # Bugs in transfer():
        #   1. No check for negative or zero amount
        #   2. No check for insufficient funds
        #   3. No check for self-transfer
        "bug_line": 7,
        "bug_type": "missing-validation",
        "test_cases": [
            # raw_script=True → input is pasted directly after agent's fix;
            # must call print() itself.
            {
                "input": (
                    "a = BankAccount('Alice', 100)\n"
                    "b = BankAccount('Bob', 50)\n"
                    "try:\n"
                    "    a.transfer(b, -10)\n"
                    "    print('FAIL')\n"
                    "except ValueError:\n"
                    "    print('PASS')"
                ),
                "output": "PASS",
                "raw_script": True,
                "description": "Negative amount must raise ValueError",
            },
            {
                "input": (
                    "a = BankAccount('Alice', 100)\n"
                    "b = BankAccount('Bob', 50)\n"
                    "try:\n"
                    "    a.transfer(b, 200)\n"
                    "    print('FAIL')\n"
                    "except ValueError:\n"
                    "    print('PASS')"
                ),
                "output": "PASS",
                "raw_script": True,
                "description": "Insufficient funds must raise ValueError",
            },
            {
                "input": (
                    "a = BankAccount('Alice', 100)\n"
                    "try:\n"
                    "    a.transfer(a, 50)\n"
                    "    print('FAIL')\n"
                    "except ValueError:\n"
                    "    print('PASS')"
                ),
                "output": "PASS",
                "raw_script": True,
                "description": "Self-transfer must raise ValueError",
            },
            {
                "input": (
                    "a = BankAccount('Alice', 100)\n"
                    "b = BankAccount('Bob', 50)\n"
                    "a.transfer(b, 30)\n"
                    "print(f'{a.get_balance()},{b.get_balance()}')"
                ),
                "output": "70,80",
                "raw_script": True,
                "description": "Valid transfer must update both balances correctly",
            },
        ],
    },
]


# ── Convenience lookup ─────────────────────────────────────────────
TASK_INDEX = {task["id"]: task for task in TASKS}


def get_task(task_id: str) -> dict:
    """Return a task by id, or raise KeyError if not found."""
    if task_id not in TASK_INDEX:
        raise KeyError(
            f"Unknown task '{task_id}'. "
            f"Available: {list(TASK_INDEX.keys())}"
        )
    return TASK_INDEX[task_id]


def get_all_tasks() -> list:
    """Return all tasks."""
    return TASKS


# ── Quick self-check ───────────────────────────────────────────────
if __name__ == "__main__":
    for task in TASKS:
        print(f"[{str(task['difficulty']).upper()}] {task['id']}")
        print(f"  title     : {task['title']}")
        print(f"  bug_line  : {task['bug_line']}")
        print(f"  bug_type  : {task['bug_type']}")
        print(f"  test_cases: {len(cast(list, task['test_cases']))}")
        print()