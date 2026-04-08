from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional

from app.env import EmergencyDispatchEnv
from app.models import Action
from app.tasks import TASKS

app = FastAPI(
    title="Emergency Dispatch OpenEnv",
    version="1.0.0"
)

# CORS (important for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: Optional[EmergencyDispatchEnv] = None


# ─────────────────────────────────────────────
# ROOT (UI — SAFE CHANGE)
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("app/static/index.html")


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
# RESET (CRITICAL — DO NOT CHANGE)
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