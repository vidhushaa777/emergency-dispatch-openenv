"""
FINAL VERSION — OpenEnv Checker Safe
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import Optional

from app.env import EmergencyDispatchEnv
from app.models import Action
from app.tasks import TASKS

app = FastAPI(
    title="Emergency Dispatch OpenEnv",
    version="1.0.0"
)

# Allow all (needed for HF)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
            "difficulty": cfg.difficulty,
            "max_steps": cfg.max_steps,
        }
        for name, cfg in TASKS.items()
    }


# ─────────────────────────────────────────────
# 🔥 RESET (FINAL FIX — NO BODY)
# ─────────────────────────────────────────────
@app.post("/reset")
def reset():
    global _env

    _env = EmergencyDispatchEnv(
        task_name="standard_dispatch",
        seed=42
    )

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
            body {font-family: monospace; background:#060a12; color:#d4daf0; padding:40px;}
            h1 {color:#06b6d4;}
            table {border-collapse: collapse; margin-top:20px;}
            td, th {border:1px solid #1e2d45; padding:10px;}
        </style>
    </head>
    <body>
        <h1>⚡ Emergency Dispatch OpenEnv</h1>
        <p>API is running successfully.</p>

        <table>
            <tr><th>Endpoint</th><th>Method</th></tr>
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