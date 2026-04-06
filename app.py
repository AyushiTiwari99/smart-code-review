"""
app.py — Smart Code Review · Hugging Face Space
================================================
A Gradio web UI that wraps the CodeReviewEnv so anyone can
interact with it via browser without touching the CLI.
"""

import os
import json
import gradio as gr

from environment import CodeReviewEnv
from tasks import TASKS

# ── Global env (one per session via gr.State) ──────────────────────
TASK_IDS   = [t["id"] for t in TASKS]
TASK_LABEL = {t["id"]: f"[{t['difficulty'].upper()}] {t['title']}" for t in TASKS}

# ── Helpers ────────────────────────────────────────────────────────

def load_task(task_id: str, state: dict) -> tuple:
    env = CodeReviewEnv()
    obs = env.reset(task_id)
    state["env"] = env
    state["obs"] = obs

    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(obs["difficulty"], "⚪")
    header = f"{diff_emoji} **{obs['title']}**  ·  `{obs['task_id']}`"
    desc   = obs["description"]
    code   = obs["buggy_code"]
    return header, desc, code, "", "", "", "", "", state


def run_inference(task_id: str, state: dict) -> tuple:
    """Run the LLM agent (needs OPENAI_API_KEY secret on HF)."""
    env: CodeReviewEnv = state.get("env")
    obs = state.get("obs", {})
    if env is None:
        return "⚠️ Load a task first.", "", "", "", state

    buggy_code = obs.get("buggy_code", "")

    from inference import get_action
    action = get_action(buggy_code)
    action.pop("used_reflection", None)
    action.pop("_initial_action", None)

    result = env.step(action)
    lr     = env._last_result or {}

    reward_bar = _reward_bar(result["reward"])
    scores_md  = _scores_table(lr)
    action_md  = f"```json\n{json.dumps(action, indent=2)}\n```"
    fix_code   = action.get("fix", "")

    state["last_result"] = result
    return reward_bar, scores_md, action_md, fix_code, state


def run_manual(bug_line: str, issues: str, fix: str, state: dict) -> tuple:
    env: CodeReviewEnv = state.get("env")
    if env is None:
        return "⚠️ Load a task first.", "", state

    try:
        bl = int(bug_line)
    except (ValueError, TypeError):
        bl = -1

    issues_list = [i.strip() for i in issues.split(",") if i.strip()]
    action = {"bug_line": bl, "issues": issues_list, "fix": fix}
    result = env.step(action)
    lr     = env._last_result or {}

    reward_bar = _reward_bar(result["reward"])
    scores_md  = _scores_table(lr)
    state["last_result"] = result
    return reward_bar, scores_md, state


# ── Formatting helpers ─────────────────────────────────────────────

def _reward_bar(reward: float) -> str:
    pct   = int(reward * 100)
    color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 40 else "#ef4444"
    bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return (
        f"### Reward: **{reward:.2f}** / 1.00\n\n"
        f"`{bar}` {pct}%"
    )


def _scores_table(lr: dict) -> str:
    if not lr:
        return ""
    rows = [
        ("Issue description", lr.get("issue_score", 0),   30),
        ("Line identification", lr.get("line_score", 0),  20),
        ("Compile check",      lr.get("compile_score", 0), 20),
        ("Test cases",         lr.get("test_score", 0),   30),
    ]
    tp = lr.get("tests_passed", 0)
    tt = lr.get("tests_total", 0)
    md = "| Axis | Score | Weight |\n|---|---|---|\n"
    for label, score, weight in rows:
        emoji = "✅" if score >= 0.99 else "🟡" if score > 0 else "❌"
        extra = f" ({tp}/{tt} tests)" if label == "Test cases" else ""
        md += f"| {emoji} {label}{extra} | `{score:.2f}` | {weight}% |\n"

    penalty = lr.get("penalty", 0)
    steps   = lr.get("steps_taken", 0)
    allowed = lr.get("steps_allowed", 0)
    md += f"\n**Steps:** {steps} / {allowed} allowed"
    if penalty:
        md += f"  ·  **Penalty:** −{penalty:.2f}"
    return md


# ── Gradio UI ──────────────────────────────────────────────────────

CSS = """
#header { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.03em; }
#subheader { color: #6b7280; margin-top: -0.5rem; }
.code-block { font-family: monospace !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="🧠 Smart Code Review",
    theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
    css=CSS,
) as demo:

    state = gr.State({})

    # ── Header ─────────────────────────────────────────────────────
    gr.Markdown("# 🧠 Smart Code Review", elem_id="header")
    gr.Markdown(
        "An AI-powered code review environment. "
        "Pick a buggy Python task, run the AI agent or submit your own fix, "
        "and see how it scores across 4 dimensions.",
        elem_id="subheader",
    )

    # ── Task picker ────────────────────────────────────────────────
    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[(TASK_LABEL[tid], tid) for tid in TASK_IDS],
            value=TASK_IDS[0],
            label="Select Task",
            scale=4,
        )
        load_btn = gr.Button("Load Task ▶", variant="primary", scale=1)

    task_header = gr.Markdown("*Load a task to begin.*")

    with gr.Row():
        with gr.Column(scale=1):
            task_desc = gr.Textbox(label="Task Description", lines=3, interactive=False)
        with gr.Column(scale=2):
            buggy_code = gr.Code(label="Buggy Code", language="python", lines=14, interactive=False)

    gr.Markdown("---")

    # ── AI agent tab / Manual tab ───────────────────────────────────
    with gr.Tabs():

        with gr.TabItem("🤖  AI Agent"):
            gr.Markdown(
                "> Runs `inference.py` using **GPT-4o** with self-reflection.  \n"
                "> Requires `OPENAI_API_KEY` to be set in Space Secrets."
            )
            agent_btn = gr.Button("Run AI Agent ⚡", variant="primary")

            with gr.Row():
                agent_reward = gr.Markdown("*Run the agent to see results.*")

            agent_scores = gr.Markdown()

            with gr.Accordion("Agent Action (JSON)", open=False):
                agent_action_md = gr.Markdown()

            with gr.Accordion("Agent's Fixed Code", open=False):
                agent_fix = gr.Code(language="python", lines=16, interactive=False)

        with gr.TabItem("✍️  Manual Submission"):
            gr.Markdown(
                "> Enter your own bug analysis and fix, then click **Submit**."
            )
            with gr.Row():
                manual_line   = gr.Textbox(label="Bug Line Number", placeholder="e.g. 3", scale=1)
                manual_issues = gr.Textbox(
                    label="Issues (comma-separated)",
                    placeholder="e.g. off-by-one, range starts at 1",
                    scale=3,
                )
            manual_fix = gr.Code(
                label="Your Fixed Code",
                language="python",
                lines=14,
                value="# Paste your corrected code here",
            )
            manual_btn    = gr.Button("Submit Fix ✅", variant="primary")
            manual_reward = gr.Markdown()
            manual_scores = gr.Markdown()

    # ── Leaderboard info ────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "### Scoring breakdown\n"
        "| Axis | Weight | What it checks |\n"
        "|---|---|---|\n"
        "| Issue description | 30% | Correct bug type keyword |\n"
        "| Line identification | 20% | Correct line ±1 |\n"
        "| Compile check | 20% | Code runs without errors |\n"
        "| Test cases | 30% | Fraction of tests passed |\n\n"
        "A time penalty of **−0.05 per extra step** applies beyond the free-step allowance "
        "(easy=2, medium=3, hard=4). Final reward is clamped to [0.10, 1.00]."
    )

    # ── Wiring ─────────────────────────────────────────────────────
    load_btn.click(
        load_task,
        inputs=[task_dropdown, state],
        outputs=[task_header, task_desc, buggy_code,
                 agent_reward, agent_scores, agent_action_md, agent_fix,
                 manual_reward, state],
    )

    agent_btn.click(
        run_inference,
        inputs=[task_dropdown, state],
        outputs=[agent_reward, agent_scores, agent_action_md, agent_fix, state],
    )

    manual_btn.click(
        run_manual,
        inputs=[manual_line, manual_issues, manual_fix, state],
        outputs=[manual_reward, manual_scores, state],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
