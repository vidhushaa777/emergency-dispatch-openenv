from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from app.env import EmergencyDispatchEnv
from app.tasks import TASKS

app = FastAPI(title="Emergency Dispatch OpenEnv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EmergencyDispatchEnv()

class DispatchAction(BaseModel):
    unit_id: str
    incident_id: Optional[str] = None
    reasoning: Optional[str] = ""

class ActionRequest(BaseModel):
    dispatches: List[DispatchAction] = []

class ResetRequest(BaseModel):
    task_name: Optional[str] = "standard_dispatch"
    seed: Optional[int] = 42

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def get_tasks():
    return {"tasks": list(TASKS.keys())}

@app.post("/reset")
def reset(body: Optional[ResetRequest] = None):
    task_name = body.task_name if body else "standard_dispatch"
    seed = body.seed if body else 42
    obs = env.reset(task_name=task_name, seed=seed)
    return obs

@app.post("/step")
def step(action: ActionRequest):
    result = env.step(action.dict())
    return result

@app.get("/state")
def state():
    return env.get_state()

@app.get("/grade")
def grade():
    score = env.get_grade()
    return {"score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
