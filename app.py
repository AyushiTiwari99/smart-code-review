"""
app.py — Smart Code Review · Hugging Face Space
================================================
Exposes:
  - FastAPI REST endpoints  /reset  /step  /state  (for OpenEnv automated checks)
  - Gradio web UI at /ui   (for human interaction)
"""

import json
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import gradio as gr
from environment import CodeReviewEnv
from tasks import TASKS

# ══════════════════════════════════════════════════════════════════════
# FastAPI app — OpenEnv REST interface
# ══════════════════════════════════════════════════════════════════════

api = FastAPI(title="Smart Code Review — OpenEnv API")

_api_env = CodeReviewEnv()

SCORE_FLOOR = 0.01
SCORE_CEIL  = 0.99


def _clamp(score) -> float:
    """Ensure score is strictly between 0.01 and 0.99."""
    try:
        return float(max(SCORE_FLOOR, min(SCORE_CEIL, float(score))))
    except Exception:
        return SCORE_FLOOR


def _clamp_result(result: dict) -> dict:
    """Clamp reward in a step result."""
    if not isinstance(result, dict):
        return result
    if "reward" in result:
        result["reward"] = _clamp(result["reward"])
    return result


@api.get("/")
def root():
    return {"status": "ok", "name": "Smart Code Review OpenEnv"}


@api.post("/reset")
async def api_reset(request: Request):
    global _api_env
    try:
        body = await request.json()
        task_id = body.get("task_id", None) if isinstance(body, dict) else None
    except Exception:
        task_id = None
    try:
        # Fresh env instance on every reset — prevents stale state between tasks
        _api_env = CodeReviewEnv()
        obs = _api_env.reset(task_id)
        return JSONResponse(content=obs)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@api.post("/step")
async def api_step(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    action = {
        "bug_line": body.get("bug_line", -1),
        "issues":   body.get("issues",   []),
        "fix":      body.get("fix",       ""),
    }

    try:
        result = _api_env.step(action)
        result = _clamp_result(result)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@api.get("/state")
def api_state():
    try:
        return JSONResponse(content=_api_env.state())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════
# Gradio UI helpers
# ══════════════════════════════════════════════════════════════════════

TASK_IDS   = [t["id"] for t in TASKS]
TASK_LABEL = {t["id"]: f"[{t['difficulty'].upper()}] {t['title']}" for t in TASKS}


def load_task(task_id: str, state: dict) -> tuple:
    env = CodeReviewEnv()
    obs = env.reset(task_id)
    state["env"] = env
    state["obs"] = obs
    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(obs["difficulty"], "⚪")
    header = f"{diff_emoji} **{obs['title']}**  ·  `{obs['task_id']}`"
    return header, obs["description"], obs["buggy_code"], "", "", "", "", "", state


def run_inference(task_id: str, state: dict) -> tuple:
    env = state.get("env")
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
    state["last_result"] = result
    return (
        _reward_bar(result["reward"]),
        _scores_table(lr),
        f"```json\n{json.dumps(action, indent=2)}\n```",
        action.get("fix", ""),
        state,
    )


def run_manual(bug_line: str, issues: str, fix: str, state: dict) -> tuple:
    env = state.get("env")
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
    state["last_result"] = result
    return _reward_bar(result["reward"]), _scores_table(lr), state


def _reward_bar(reward: float) -> str:
    pct = int(reward * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return f"### Reward: **{reward:.2f}** / 1.00\n\n`{bar}` {pct}%"


def _scores_table(lr: dict) -> str:
    if not lr:
        return ""
    rows = [
        ("Issue description",   lr.get("issue_score",   0), 30),
        ("Line identification", lr.get("line_score",    0), 20),
        ("Compile check",       lr.get("compile_score", 0), 20),
        ("Test cases",          lr.get("test_score",    0), 30),
    ]
    tp = lr.get("tests_passed", 0)
    tt = lr.get("tests_total",  0)
    md = "| Axis | Score | Weight |\n|---|---|---|\n"
    for label, score, weight in rows:
        emoji = "✅" if score >= 0.99 else "🟡" if score > 0 else "❌"
        extra = f" ({tp}/{tt} tests)" if label == "Test cases" else ""
        md += f"| {emoji} {label}{extra} | `{score:.2f}` | {weight}% |\n"
    penalty = lr.get("penalty",       0)
    steps   = lr.get("steps_taken",   0)
    allowed = lr.get("steps_allowed", 0)
    md += f"\n**Steps:** {steps} / {allowed} allowed"
    if penalty:
        md += f"  ·  **Penalty:** −{penalty:.2f}"
    return md


# ══════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════

with gr.Blocks() as gradio_app:

    state = gr.State({})

    gr.Markdown("# 🧠 Smart Code Review")
    gr.Markdown(
        "An AI-powered code review environment. Pick a buggy Python task, "
        "run the AI agent or submit your own fix, and see how it scores."
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=[(TASK_LABEL[tid], tid) for tid in TASK_IDS],
            value=TASK_IDS[0], label="Select Task", scale=4,
        )
        load_btn = gr.Button("Load Task ▶️", variant="primary", scale=1)

    task_header = gr.Markdown("*Load a task to begin.*")

    with gr.Row():
        with gr.Column(scale=1):
            task_desc  = gr.Textbox(label="Task Description", lines=3, interactive=False)
        with gr.Column(scale=2):
            buggy_code = gr.Code(label="Buggy Code", language="python",
                                 lines=14, interactive=False)

    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("🤖  AI Agent"):
            gr.Markdown(
                "> Runs GPT-4o with self-reflection.  \n"
                "> Requires `HF_TOKEN` or `OPENAI_API_KEY` in Space Secrets."
            )
            agent_btn    = gr.Button("Run AI Agent ⚡", variant="primary")
            agent_reward = gr.Markdown("*Run the agent to see results.*")
            agent_scores = gr.Markdown()
            with gr.Accordion("Agent Action (JSON)", open=False):
                agent_action_md = gr.Markdown()
            with gr.Accordion("Agent's Fixed Code", open=False):
                agent_fix = gr.Code(language="python", lines=16, interactive=False)

        with gr.TabItem("✍️  Manual Submission"):
            gr.Markdown("> Enter your own bug analysis and fix, then click Submit.")
            with gr.Row():
                manual_line   = gr.Textbox(label="Bug Line Number",
                                           placeholder="e.g. 3", scale=1)
                manual_issues = gr.Textbox(label="Issues (comma-separated)",
                                           placeholder="e.g. off-by-one", scale=3)
            manual_fix    = gr.Code(label="Your Fixed Code", language="python",
                                    lines=14, value="# Paste your corrected code here")
            manual_btn    = gr.Button("Submit Fix ✅", variant="primary")
            manual_reward = gr.Markdown()
            manual_scores = gr.Markdown()

    load_btn.click(
        load_task,
        inputs=[task_dropdown, state],
        outputs=[task_header, task_desc, buggy_code,
                 agent_reward, agent_scores, agent_action_md,
                 agent_fix, manual_reward, state],
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


# ══════════════════════════════════════════════════════════════════════
# Mount Gradio inside FastAPI and launch
# ══════════════════════════════════════════════════════════════════════

app = gr.mount_gradio_app(api, gradio_app, path="/ui")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
