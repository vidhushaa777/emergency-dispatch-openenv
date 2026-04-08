"""
inference.py — Emergency Dispatch OpenEnv
Baseline agent with correct [START][STEP][END] log format.
"""
import os
import json
import sys
import time
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
ENV_URL      = os.getenv("ENV_URL", API_BASE_URL)
SEED         = 42
TASKS        = ["standard_dispatch", "mass_casualty", "resource_scarcity"]


def env_reset(task_name):
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "seed": SEED})
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


def choose_action(obs):
    incidents = obs.get("active_incidents", [])
    units     = obs.get("units", [])
    priority_map = {"high": 3, "medium": 2, "low": 1}
    sorted_inc = sorted(incidents,
                        key=lambda x: priority_map.get(x["priority"], 0),
                        reverse=True)
    dispatches = []
    used_units = set()
    for inc in sorted_inc:
        for unit in units:
            if (unit["type"] == inc["type"]
                    and unit["status"] == "idle"
                    and unit["fuel"] > 10
                    and unit["id"] not in used_units):
                dispatches.append({
                    "unit_id": unit["id"],
                    "incident_id": inc["id"],
                    "reasoning": f"{inc['priority']} {inc['type']} to {unit['id']}"
                })
                used_units.add(unit["id"])
                break
    return {"dispatches": dispatches}


def run_task(task_name):
    result_data = env_reset(task_name)
    obs = result_data if "active_incidents" in result_data else result_data.get("observation", result_data)

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        action = choose_action(obs)
        result = env_step(action)

        reward = result.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("total", 0)

        total_reward += reward
        done = result.get("done", False)
        obs = result.get("observation", obs)
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        print(json.dumps({
            "type": "STEP",
            "task": task_name,
            "step": step,
            "action": action,
            "reward": round(reward, 4),
            "cumulative_reward": round(total_reward, 4),
            "done": done,
            "resolved_count": obs.get("resolved_count", 0),
            "active_incidents": len(obs.get("active_incidents", [])),
        }))
        sys.stdout.flush()

        if done:
            break

    grade = env_grade()
    return {
        "task": task_name,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "grade_score": grade.get("score", 0.0),
        "grade_breakdown": grade.get("breakdown", {}),
        "passed": grade.get("passed", False),
    }


def main():
    start_time = time.time()

    print(json.dumps({
        "type": "START",
        "model": MODEL_NAME,
        "tasks": TASKS,
        "seed": SEED,
        "env_url": ENV_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }))
    sys.stdout.flush()

    results = {}

    for task_name in TASKS:
        print(json.dumps({"type": "STEP", "event": "task_start", "task": task_name}))
        sys.stdout.flush()

        try:
            result = run_task(task_name)
            results[task_name] = result
        except Exception as e:
            results[task_name] = {"task": task_name, "error": str(e), "grade_score": 0.0, "passed": False}
            print(json.dumps({"type": "STEP", "event": "task_error", "task": task_name, "error": str(e)}))
            sys.stdout.flush()

    elapsed = round(time.time() - start_time, 2)
    avg_score = round(sum(r.get("grade_score", 0) for r in results.values()) / len(results), 4)

    print(json.dumps({
        "type": "END",
        "model": MODEL_NAME,
        "seed": SEED,
        "elapsed_seconds": elapsed,
        "average_grade_score": avg_score,
        "results": {
            task: {
                "grade_score": r.get("grade_score", 0.0),
                "total_reward": r.get("total_reward", 0.0),
                "passed": r.get("passed", False),
                "steps": r.get("steps", 0),
                "grade_breakdown": r.get("grade_breakdown", {}),
            }
            for task, r in results.items()
        },
        "passed_all": all(r.get("passed", False) for r in results.values()),
    }))
    sys.stdout.flush()
    sys.exit(0 if all(r.get("passed", False) for r in results.values()) else 1)


if __name__ == "__main__":
    main()