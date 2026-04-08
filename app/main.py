"""
app/main.py — Emergency Dispatch OpenEnv Server
"""
import random
import time
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Emergency Dispatch OpenEnv")

state = {
    "task_name": None,
    "seed": 42,
    "step_count": 0,
    "resolved": [],
    "active_incidents": [],
    "units": [],
    "scores": [],
    "max_steps": 20,
}

UNIT_TYPES = ["ambulance", "fire_truck", "police"]
INCIDENT_TYPES = ["ambulance", "fire_truck", "police"]
PRIORITIES = ["high", "medium", "low"]

TASKS = {
    "standard_dispatch": {"max_steps": 20, "num_incidents": 5, "num_units": 6},
    "mass_casualty":     {"max_steps": 30, "num_incidents": 12, "num_units": 8},
    "resource_scarcity": {"max_steps": 25, "num_incidents": 8, "num_units": 3},
}


def make_incidents(n, rng):
    return [
        {
            "id": f"inc_{i}",
            "type": rng.choice(INCIDENT_TYPES),
            "priority": rng.choice(PRIORITIES),
            "location": [round(rng.uniform(0, 10), 2), round(rng.uniform(0, 10), 2)],
            "resolved": False,
        }
        for i in range(n)
    ]


def make_units(n, rng):
    return [
        {
            "id": f"unit_{i}",
            "type": UNIT_TYPES[i % len(UNIT_TYPES)],
            "status": "idle",
            "fuel": rng.randint(50, 100),
        }
        for i in range(n)
    ]


def get_obs():
    return {
        "active_incidents": [inc for inc in state["active_incidents"] if not inc["resolved"]],
        "units": state["units"],
        "resolved_count": len(state["resolved"]),
        "step": state["step_count"],
    }


class Dispatch(BaseModel):
    unit_id: str
    incident_id: str
    reasoning: Optional[str] = ""


class StepRequest(BaseModel):
    dispatches: Optional[List[Dispatch]] = []


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/")
def root():
    return {"status": "ok", "service": "Emergency Dispatch OpenEnv"}


@app.post("/reset")
async def reset(request: Request):
    # Accept empty body OR json body
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_name = body.get("task_name", "standard_dispatch") if body else "standard_dispatch"
    seed = body.get("seed", 42) if body else 42

    rng = random.Random(seed)
    cfg = TASKS.get(task_name, TASKS["standard_dispatch"])

    state.update({
        "task_name": task_name,
        "seed": seed,
        "step_count": 0,
        "resolved": [],
        "active_incidents": make_incidents(cfg["num_incidents"], rng),
        "units": make_units(cfg["num_units"], rng),
        "scores": [],
        "max_steps": cfg["max_steps"],
    })
    return get_obs()


@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    dispatches = body.get("dispatches", []) if body else []

    state["step_count"] += 1
    reward = 0.0
    priority_reward = {"high": 1.0, "medium": 0.5, "low": 0.2}

    unit_map = {u["id"]: u for u in state["units"]}
    inc_map  = {i["id"]: i for i in state["active_incidents"]}

    for d in dispatches:
        unit_id = d.get("unit_id") if isinstance(d, dict) else d.unit_id
        inc_id  = d.get("incident_id") if isinstance(d, dict) else d.incident_id
        unit = unit_map.get(unit_id)
        inc  = inc_map.get(inc_id)
        if unit and inc and not inc["resolved"]:
            if unit["type"] == inc["type"] and unit["status"] == "idle" and unit["fuel"] > 10:
                inc["resolved"] = True
                state["resolved"].append(inc)
                unit["status"] = "busy"
                unit["fuel"] = max(0, unit["fuel"] - 10)
                reward += priority_reward.get(inc["priority"], 0.2)
            else:
                reward -= 0.1

    for u in state["units"]:
        if u["status"] == "busy":
            u["status"] = "idle"

    max_steps = state.get("max_steps", 20)
    done = (state["step_count"] >= max_steps or
            all(i["resolved"] for i in state["active_incidents"]))

    state["scores"].append(reward)
    return {
        "observation": get_obs(),
        "reward": reward,
        "done": done,
        "info": {"step": state["step_count"]},
    }


@app.get("/grade")
def grade():
    total_incidents = len(state["active_incidents"])
    resolved = len(state["resolved"])
    ratio = resolved / total_incidents if total_incidents else 0.0

    high_res = sum(1 for i in state["resolved"] if i["priority"] == "high")
    high_tot = sum(1 for i in state["active_incidents"] if i["priority"] == "high")
    high_ratio = high_res / high_tot if high_tot else 1.0

    score = round(0.6 * ratio + 0.4 * high_ratio, 4)
    passed = score >= 0.5

    return {
        "score": score,
        "passed": passed,
        "breakdown": {
            "resolved_ratio": round(ratio, 4),
            "high_priority_ratio": round(high_ratio, 4),
            "total_incidents": total_incidents,
            "resolved_count": resolved,
        },
    }
