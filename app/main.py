"""
FastAPI server — Emergency Dispatch OpenEnv
Endpoints: POST /reset, POST /step, GET /state, GET /grade, GET /tasks, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from app.env import EmergencyDispatchEnv
from app.models import Action, StepResult, Observation
from app.tasks import TASKS

app = FastAPI(
    title="Emergency Dispatch OpenEnv",
    description="OpenEnv-compliant RL environment for emergency dispatch coordination.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance
_env: Optional[EmergencyDispatchEnv] = None


# ─────────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_name: str = "standard_dispatch"
    seed: int = 42


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "env": "emergency-dispatch", "version": "1.0.0"}


# ─────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────
@app.get("/tasks")
def list_tasks():
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
# RESET (FIXED ✅)
# ─────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    global _env

    # Default values if no body is provided
    if req is None:
        task_name = "standard_dispatch"
        seed = 42
    else:
        task_name = req.task_name
        seed = req.seed

    if task_name not in TASKS:
        raise HTTPException(400, f"Unknown task '{task_name}'. Valid: {list(TASKS.keys())}")

    _env = EmergencyDispatchEnv(task_name=task_name, seed=seed)
    return _env.reset()


# ─────────────────────────────────────────────
# STEP
# ─────────────────────────────────────────────
@app.post("/step", response_model=StepResult)
def step(action: Action):
    global _env

    if _env is None:
        raise HTTPException(400, "Environment not initialized. Call /reset first.")

    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
@app.get("/state")
def state():
    global _env

    if _env is None:
        raise HTTPException(400, "Environment not initialized. Call /reset first.")

    return _env.state()


# ─────────────────────────────────────────────
# GRADE
# ─────────────────────────────────────────────
@app.get("/grade")
def grade():
    global _env

    if _env is None:
        raise HTTPException(400, "Environment not initialized. Call /reset first.")

    score, breakdown = _env.grade()

    return {
        "task": _env.task_name,
        "score": score,
        "breakdown": breakdown,
        "passed": score >= TASKS[_env.task_name].success_threshold,
        "success_threshold": TASKS[_env.task_name].success_threshold,
    }


# ─────────────────────────────────────────────
# ROOT (Dashboard)
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
            code{background:#1a2035;padding:2px 6px;border-radius:3px;}
            table{border-collapse:collapse;margin:16px 0;}
            td,th{padding:8px 16px;border:1px solid #1e2d45;text-align:left;}
        </style>
    </head>
    <body>
        <h1>⚡ Emergency Dispatch OpenEnv</h1>
        <p>OpenEnv-compliant environment for training and evaluating LLM dispatch agents.</p>

        <table>
            <tr><th>Endpoint</th><th>Method</th><th>Description</th></tr>
            <tr><td><a href="/docs">/docs</a></td><td>GET</td><td>Interactive API documentation</td></tr>
            <tr><td>/health</td><td>GET</td><td>Health check</td></tr>
            <tr><td>/tasks</td><td>GET</td><td>List all tasks</td></tr>
            <tr><td>/reset</td><td>POST</td><td>Reset environment</td></tr>
            <tr><td>/step</td><td>POST</td><td>Take a step</td></tr>
            <tr><td>/state</td><td>GET</td><td>Current state snapshot</td></tr>
            <tr><td>/grade</td><td>GET</td><td>Grade current episode</td></tr>
        </table>

        <p>Tasks: <code>standard_dispatch</code> · <code>mass_casualty</code> · <code>resource_scarcity</code></p>
    </body>
    </html>
    """