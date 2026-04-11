from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

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
    env.__init__(task_name=task_name, seed=seed)
    obs = env.reset()
    return obs

@app.post("/step")
def step(action: ActionRequest):
    from app.models import Action, DispatchDecision
    act = Action(dispatches=[
        DispatchDecision(
            unit_id=d.unit_id,
            incident_id=d.incident_id,
            reasoning=d.reasoning
        ) for d in action.dispatches
    ])
    result = env.step(act)
    return result

@app.get("/state")
def state():
    return env.state()

@app.get("/grade")
def grade():
    score, details = env.grade()
    score = max(0.001, min(0.999, float(score)))
    return {"score": score, "details": details}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
