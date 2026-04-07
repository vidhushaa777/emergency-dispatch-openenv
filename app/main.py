"""
FastAPI server — Emergency Dispatch OpenEnv
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os

from app.env import EmergencyDispatchEnv
from app.models import Action, StepResult, Observation
from app.tasks import TASKS

app = FastAPI(
    title="Emergency Dispatch OpenEnv",
    description="OpenEnv-compliant RL environment for emergency dispatch coordination.",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# STATIC FILES (VERY IMPORTANT 🔥)
# ─────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ─────────────────────────────────────────────
# ENV INSTANCE
# ─────────────────────────────────────────────
_env: Optional[EmergencyDispatchEnv] = None


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
# RESET
# ─────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    global _env
    if req.task_name not in TASKS:
        raise HTTPException(400, f"Unknown task '{req.task_name}'. Valid: {list(TASKS.keys())}")
    _env = EmergencyDispatchEnv(task_name=req.task_name, seed=req.seed)
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
# ROOT — SERVE YOUR HTML UI 🔥
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    file_path = os.path.join("app", "static", "index.html")

    if not os.path.exists(file_path):
        return HTMLResponse("<h2>index.html not found in app/static</h2>")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()