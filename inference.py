"""
inference.py — Submission-safe version for Emergency Dispatch OpenEnv
"""

import os
import json
import sys
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# ENV VARIABLES (REQUIRED FORMAT)
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

ENV_URL = API_BASE_URL
SEED = 42
TASK_NAME = "standard_dispatch"   # change if needed

client = OpenAI()

# ─────────────────────────────────────────────
# ENV FUNCTIONS
# ─────────────────────────────────────────────
def env_reset():
    r = requests.post(f"{ENV_URL}/reset", json={
        "task_name": TASK_NAME,
        "seed": SEED
    })
    r.raise_for_status()
    return r.json()

def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action)
    r.raise_for_status()
    return r.json()

def env_grade():
    r = requests.get(f"{ENV_URL}/grade")
    r.raise_for_status()
    return r.json()

# ─────────────────────────────────────────────
# SIMPLE AGENT (SAFE BASELINE)
# ─────────────────────────────────────────────
def choose_action(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    dispatches = []

    priority_map = {"high": 3, "medium": 2, "low": 1}
    incidents = sorted(incidents, key=lambda x: priority_map[x["priority"]], reverse=True)

    used_units = set()

    for inc in incidents:
        for unit in units:
            if (
                unit["type"] == inc["type"]
                and unit["status"] == "idle"
                and unit["fuel"] > 10
                and unit["id"] not in used_units
            ):
                dispatches.append({
                    "unit_id": unit["id"],
                    "incident_id": inc["id"],
                    "reasoning": f"{inc['priority']} {inc['type']} handled by {unit['id']}"
                })
                used_units.add(unit["id"])
                break

    return {"dispatches": dispatches}

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    print("START")
    sys.stdout.flush()

    obs_data = env_reset()
    obs = obs_data["observation"]

    total_reward = 0

    for step in range(30):
        action = choose_action(obs)

        result = env_step(action)

        reward = result.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("total", 0)

        total_reward += reward
        obs = result["observation"]

        print(f"STEP {step} | reward={reward} | actions={action}")
        sys.stdout.flush()

        if result.get("done"):
            break

    grade = env_grade()

    print("END")
    print(f"FINAL_GRADE: {grade}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()