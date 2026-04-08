"""
FastAPI server — Emergency Dispatch OpenEnv
FINAL VERSION (passes OpenEnv checker)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Optional

from app.env import EmergencyDispatchEnv
from app.models import Action, StepResult
from app.tasks import TASKS

app = FastAPI(
    title="Emergency Dispatch OpenEnv",
    description="OpenEnv-compliant RL environment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[EmergencyDispatchEnv] = None


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────
@app.get("/tasks")
def tasks():
    return {
        name: {
            "name": cfg.name,
            "description": cfg.description,
            "difficulty": cfg.difficulty,
            "max_steps": cfg.max_steps,
            "success_threshold": cfg.success_threshold,
        }
        for name, cfg in TASKS.items()
    }


# ─────────────────────────────────────────────
# RESET (FINAL FIX — NO BODY REQUIRED)
# ─────────────────────────────────────────────
@app.post("/reset", include_in_schema=False)
async def reset(request: Request):
    global _env

    # Try to read body safely
    try:
        body = await request.json()
    except:
        body = {}

    task_name = body.get("task_name", "standard_dispatch")
    seed = body.get("seed", 42)

    if task_name not in TASKS:
        raise HTTPException(400, f"Unknown task '{task_name}'")

    _env = EmergencyDispatchEnv(task_name=task_name, seed=seed)

    return {
        "observation": _env.reset()
    }


# ─────────────────────────────────────────────
# STEP
# ─────────────────────────────────────────────
@app.post("/step")
def step(action: Action):
    global _env

    if _env is None:
        raise HTTPException(400, "Call /reset first")

    result = _env.step(action)

    return {
        "observation": result.observation,
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
@app.get("/state")
def state():
    global _env

    if _env is None:
        raise HTTPException(400, "Call /reset first")

    return _env.state()


# ─────────────────────────────────────────────
# GRADE
# ─────────────────────────────────────────────
@app.get("/grade")
def grade():
    global _env

    if _env is None:
        raise HTTPException(400, "Call /reset first")

    score, breakdown = _env.grade()

    return {
        "score": score,
        "breakdown": breakdown,
    }


# ─────────────────────────────────────────────
# ROOT UI
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head>
        <title>Emergency Dispatch OpenEnv</title>
        <style>
            body{font-family:monospace;background:#060a12;color:#d4daf0;padding:40px;}
            a{color:#3b82f6;}
            h1{color:#06b6d4;}
            table{border-collapse:collapse;margin:16px 0;}
            td,th{padding:8px 16px;border:1px solid #1e2d45;text-align:left;}
        </style>
    </head>
    <body>
        <h1>⚡ Emergency Dispatch OpenEnv</h1>
        <p>Live RL environment for emergency dispatch.</p>

        <table>
            <tr><th>Endpoint</th><th>Method</th></tr>
            <tr><td>/docs</td><td>GET</td></tr>
            <tr><td>/health</td><td>GET</td></tr>
            <tr><td>/tasks</td><td>GET</td></tr>
            <tr><td>/reset</td><td>POST</td></tr>
            <tr><td>/step</td><td>POST</td></tr>
            <tr><td>/state</td><td>GET</td></tr>
            <tr><td>/grade</td><td>GET</td></tr>
        </table>
    </body>
    </html>
    """