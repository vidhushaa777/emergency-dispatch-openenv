"""
inference.py — Emergency Dispatch OpenEnv
Correct [START][STEP][END] log format.
"""
import os
import sys
import time
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
ENV_URL      = os.getenv("ENV_URL", API_BASE_URL)
SEED         = 42
TASKS        = ["standard_dispatch", "mass_casualty", "resource_scarcity"]

# ── Print [START] immediately so validator always sees it ──────────────────────
print(f"[START] model={MODEL_NAME} tasks={','.join(TASKS)} seed={SEED} env_url={ENV_URL}", flush=True)


def env_reset(task_name):
    r = requests.post(f"{ENV_URL}/reset",
                      json={"task_name": task_name, "seed": SEED},
                      timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action):
    r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_grade():
    r = requests.get(f"{ENV_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()


def clamp_score(score):
    """Ensure score is strictly between 0 and 1 (not 0.0 and not 1.0)."""
    return round(min(0.999, max(0.001, float(score))), 4)


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
    print(f"[STEP] event=task_start task={task_name}", flush=True)

    try:
        result_data = env_reset(task_name)
    except Exception as e:
        print(f"[STEP] event=reset_error task={task_name} error={e}", flush=True)
        print(f"[END] task={task_name} score=0.001 steps=0", flush=True)
        return {"task": task_name, "grade_score": 0.001, "passed": False, "steps": 0, "total_reward": 0.0}

    obs = result_data if "active_incidents" in result_data else result_data.get("observation", result_data)

    total_reward = 0.0
    step = 0
    done = False

    while not done:
        step += 1
        action = choose_action(obs)

        try:
            result = env_step(action)
        except Exception as e:
            print(f"[STEP] task={task_name} step={step} error={e}", flush=True)
            break

        reward = result.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("total", 0)

        total_reward += reward
        done = result.get("done", False)
        obs = result.get("observation", obs)
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        print(
            f"[STEP] task={task_name} step={step} reward={round(reward, 4)} "
            f"cumulative_reward={round(total_reward, 4)} done={done} "
            f"resolved_count={obs.get('resolved_count', 0)} "
            f"active_incidents={len(obs.get('active_incidents', []))}",
            flush=True
        )

        if done:
            break

    try:
        grade = env_grade()
        raw_score = grade.get("score", 0.5)
    except Exception as e:
        print(f"[STEP] event=grade_error task={task_name} error={e}", flush=True)
        raw_score = 0.5

    clamped_score = clamp_score(raw_score)
    passed = grade.get("passed", False) if "grade" in dir() else False

    print(f"[END] task={task_name} score={clamped_score} steps={step}", flush=True)

    return {
        "task": task_name,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "grade_score": clamped_score,
        "passed": passed,
    }


def main():
    start_time = time.time()
    results = {}

    for task_name in TASKS:
        try:
            result = run_task(task_name)
            results[task_name] = result
        except Exception as e:
            print(f"[STEP] event=task_error task={task_name} error={e}", flush=True)
            print(f"[END] task={task_name} score=0.001 steps=0", flush=True)
            results[task_name] = {"grade_score": 0.001, "passed": False, "steps": 0}

    elapsed = round(time.time() - start_time, 2)
    all_scores = [r.get("grade_score", 0.001) for r in results.values()]
    avg_score = clamp_score(sum(all_scores) / len(all_scores))
    passed_all = all(r.get("passed", False) for r in results.values())

    print(
        f"[END] model={MODEL_NAME} seed={SEED} elapsed_seconds={elapsed} "
        f"average_grade_score={avg_score} passed_all={passed_all}",
        flush=True
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
