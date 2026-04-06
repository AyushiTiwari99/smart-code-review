"""
code_verifier.py
================
Runs agent-submitted code in isolated subprocesses and checks test cases.

Public API:
    run_code(code, input, raw_script)   -> (success: bool, output: str)
    check_test_cases(code, tests)       -> (score: float, passed: int, total: int)
"""

import subprocess
import tempfile
import os


# ── Configuration ──────────────────────────────────────────────────
TIMEOUT = 5  # seconds per subprocess run


# ── run_code ───────────────────────────────────────────────────────

def run_code(code: str, input_data: str, raw_script: bool = False) -> tuple[bool, str]:
    """
    Execute code combined with an input expression or script in an isolated subprocess.

    Args:
        code:       Python source defining functions/classes to be executed.
        input_data: Either a single callable expression (e.g. "sum_list([1,2,3])")
                    that will be wrapped in print(), or a raw multi-line script
                    when raw_script=True (must call print() itself).
        raw_script: If False (default), wraps input_data in print() before running.
                    If True, appends input_data as-is to the code.

    Returns:
        tuple[bool, str]:
            success (bool): True if the subprocess exited with code 0, False otherwise.
            output  (str):  Stripped stdout on success, or a short error/traceback on failure.
                            Timeout produces "ERROR: timeout". Empty code produces "ERROR: empty code".
    """
    if not code.strip():
        return False, "ERROR: empty code"

    if raw_script:
        script = code.rstrip() + "\n\n" + input_data
    else:
        script = code.rstrip() + "\n\n" + f"print({input_data})"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(script)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        success = result.returncode == 0
        output = result.stdout.strip() if success else result.stderr.strip()
        return success, output

    except subprocess.TimeoutExpired:
        return False, "ERROR: timeout"

    except Exception as e:
        err_msg = "".join(c for i, c in enumerate(repr(e)) if i < 120)
        return False, f"ERROR: {err_msg}"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return False, "ERROR: unexpected failure"


# ── Adversarial fix detection ──────────────────────────────────────

# NOTE: This function is currently unused; main detection happens in grader.py
def detect_suspicious_fix(fix: str) -> bool:
    """
    Analyse a fix string for common adversarial / low-quality patterns.

    Returns True (suspicious) if any of the following are found:
        1. Hardcoded trivial returns  (return 0/1/True/False/None)
        2. Silent exception swallowing  (except: pass / except Exception: pass)
        3. Fix is too short or has no function/class body
    """
    if not fix or len(fix.strip()) < 15:
        return True

    stripped = fix.strip()
    lines    = [l.strip() for l in stripped.splitlines()]

    # 1. Hardcoded trivial return values
    _TRIVIAL_RETURNS = {
        "return 0", "return 1", "return -1",
        "return true", "return false", "return none",
        "return \"\"", "return ''",
    }
    for line in lines:
        if line.lower() in _TRIVIAL_RETURNS:
            return True

    # 2. Silent except blocks (except: pass  /  except ...: pass)
    for i, line in enumerate(lines):
        if line.startswith("except"):
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if next_line == "pass":
                return True

    # 3. No function or class body defined
    has_def = any(l.startswith("def ") or l.startswith("class ") for l in lines)
    if not has_def:
        return True

    return False


# ── Adversarial test runner ────────────────────────────────────────

def run_adversarial_tests(fix: str, normal_passed: bool) -> bool:
    """
    Run 2 generic adversarial test cases against the fix.

    Only executes if normal tests already passed (normal_passed=True).
    Returns True if the fix appears suspicious based on these tests.

    Test 1 — Side-effect check:
        Just loading/defining the fix should produce no output.
        Fixes that print hardcoded values at module level are flagged.

    Test 2 — Unexpected-value resilience:
        Feed a clearly wrong argument type (None) and ensure the fix
        does not silently return a hardcoded success value (True/1/"ok").
    """
    if not normal_passed or not fix or not fix.strip():
        return False

    try:
        # Test 1: importing/defining the fix must not produce output
        success, output = run_code(fix, "", raw_script=True)
        if success and output.strip():
            # Fix produces output at definition time — suspicious
            return True

        # Test 2: calling with None should not silently return a constant
        probe_script = (
            "try:\n"
            "    import inspect, types\n"
            "    _fns = [(n, f) for n, f in vars().items()\n"
            "            if isinstance(f, types.FunctionType)]\n"
            "    if _fns:\n"
            "        _name, _fn = _fns[0]\n"
            "        _sig = inspect.signature(_fn)\n"
            "        _nargs = len(_sig.parameters)\n"
            "        _result = _fn(*([None] * _nargs))\n"
            "        if _result in (True, 1, 0, 'ok', ''):\n"
            "            print('SUSPICIOUS')\n"
            "        else:\n"
            "            print('OK')\n"
            "    else:\n"
            "        print('OK')\n"
            "except Exception:\n"
            "    print('OK')\n"
        )
        success2, output2 = run_code(fix, probe_script, raw_script=True)
        if success2 and output2.strip() == "SUSPICIOUS":
            return True

    except Exception:
        pass

    return False


# ── check_test_cases ───────────────────────────────────────────────

def check_test_cases(code: str, tests: list[dict]) -> tuple[float, int, int]:
    """
    Run every test case and compare output to expected value.

    Args:
        code:  Python source defining the function/class under test.
        tests: List of dicts, each with:
                 "input"      - expression or raw script
                 "output"     - expected printed output (str or str-able)
                 "raw_script" - optional bool (default False)

    Returns:
        (score, passed, total)
            score:  float 0.0 – 1.0  (passed / total)
            passed: number of test cases that produced correct output
            total:  total number of test cases
    """
    total = len(tests)
    flags: list[bool] = []

    for test in tests:
        raw      = test.get("raw_script", False)
        # support both "output" (new) and "expected" (legacy) keys
        expected = str(test.get("output", test.get("expected", ""))).strip()

        success, output = run_code(code, test["input"], raw_script=raw)
        flags.append(bool(success and output.strip() == expected))

    passed: int = int(sum(flags))
    total: int  = int(total)
    score: float = float(f"{float(passed) / float(total):.4f}") if total > 0 else 0.0
    return score, passed, total


# ── Self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Easy: off-by-one ──────────────────────────────────────────
    correct_fix = (
        "def sum_list(numbers):\n"
        "    total = 0\n"
        "    for i in range(0, len(numbers)):\n"
        "        total += numbers[i]\n"
        "    return total\n"
    )

    still_buggy = (
        "def sum_list(numbers):\n"
        "    total = 0\n"
        "    for i in range(1, len(numbers)):\n"
        "        total += numbers[i]\n"
        "    return total\n"
    )

    easy_tests = [
        {"input": "sum_list([1, 2, 3])", "output": "6"},
        {"input": "sum_list([])",         "output": "0"},
        {"input": "sum_list([10])",       "output": "10"},
        {"input": "sum_list([-1, 1])",    "output": "0"},
        {"input": "sum_list([5, 5, 5])",  "output": "15"},
    ]

    print("=== run_code examples ===")
    ok, out = run_code(correct_fix, "sum_list([1, 2, 3])")
    print(f"correct fix → success={ok}, output={out!r}")

    ok, out = run_code(still_buggy, "sum_list([1, 2, 3])")
    print(f"buggy fix   → success={ok}, output={out!r}")

    ok, out = run_code("def f():\n    x = 1/0\nf()", "", raw_script=True)
    print(f"runtime err → success={ok}, output={out!r}")

    ok, out = run_code("import time\ntime.sleep(10)", "", raw_script=True)
    print(f"timeout     → success={ok}, output={out!r}")

    print("\n=== check_test_cases examples ===")
    score, passed, total = check_test_cases(correct_fix, easy_tests)
    print(f"correct fix: {passed}/{total}  score={score}")

    score, passed, total = check_test_cases(still_buggy, easy_tests)
    print(f"buggy fix:   {passed}/{total}  score={score}")

    # ── Hard: bank validation (raw_script) ────────────────────────
    bank_fix = (
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
    )

    bank_tests = [
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
        },
    ]

    score, passed, total = check_test_cases(bank_fix, bank_tests)
    print(f"bank fix:    {passed}/{total}  score={score}")