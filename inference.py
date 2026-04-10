"""
inference.py — Baseline LLM agent for Emergency Dispatch OpenEnv
Uses OpenAI client. Emits [START], [STEP], [END] structured logs.
Run: python inference.py
"""
import os
import json
import sys
import time
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# CONFIG — read from environment variables
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")
TASKS        = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED         = 42

client = OpenAI(
    api_key=OPENAI_KEY,
    base_url=API_BASE_URL if "openai" not in API_BASE_URL else None,
)


# ─────────────────────────────────────────────
# ENV CLIENT
# ─────────────────────────────────────────────
def env_reset(task_name: str, seed: int = SEED) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": seed})
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json=action)
    r.raise_for_status()
    return r.json()

def env_grade() -> dict:
    r = requests.get(f"{ENV_URL}/grade")
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Each step you receive a JSON observation containing:
- active_incidents: list of incidents needing response (type, priority, location, description)
- units: list of vehicles with their status, fuel level, and type

Your job: decide which units to dispatch to which incidents.

Rules:
1. Only dispatch the CORRECT unit type (fire→fire, medical→medical, police→police)
2. Prioritize HIGH priority incidents before MEDIUM or LOW
3. Don't dispatch units with fuel < 10
4. Don't assign already-responding units

Respond ONLY with valid JSON matching this schema:
{
  "dispatches": [
    {
      "unit_id": "F1",
      "incident_id": "F001",
      "reasoning": "High priority fire, F1 is idle and closest"
    }
  ]
}

If no action is needed, respond with: {"dispatches": []}
"""

def build_user_prompt(obs: dict) -> str:
    incidents = obs.get("active_incidents", [])
    units     = obs.get("units", [])
    unresolved = [i for i in incidents]

    prompt = f"Step {obs['step']} — {obs['message']}\n\n"

    if unresolved:
        prompt += "ACTIVE INCIDENTS:\n"
        for inc in unresolved:
            prompt += (
                f"  [{inc['priority'].upper()}] {inc['id']} | {inc['type']} | "
                f"{inc['location']} | {inc['description']}\n"
            )
    else:
        prompt += "No active incidents.\n"

    prompt += "\nUNITS:\n"
    for u in units:
        prompt += (
            f"  {u['id']} ({u['type']}) | status: {u['status']} | "
            f"fuel: {u['fuel']:.0f}% | assigned: {u.get('assigned_incident') or 'none'}\n"
        )

    prompt += f"\nEpisode reward so far: {obs['episode_reward_so_far']}"
    return prompt


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def get_action(obs: dict, history: list) -> tuple[dict, str]:
    user_msg = build_user_prompt(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # keep last 3 turns for context
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action = json.loads(raw)
        history.append({"role": "assistant", "content": raw})
        return action, raw
    except Exception as e:
        # Fallback: no-op action
        fallback = {"dispatches": []}
        history.append({"role": "assistant", "content": json.dumps(fallback)})
        return fallback, f"ERROR: {e}"


# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────
def run_task(task_name: str) -> dict:
    history = []
    obs = env_reset(task_name, seed=SEED)
    done = False
    step = 0
    total_reward = 0.0
    step_logs = []

    while not done:
        step += 1
        action, raw_response = get_action(obs, history)

        result = env_step(action)
        reward      = result["reward"]["total"]
        done        = result["done"]
        obs         = result["observation"]
        total_reward += reward

        step_log = {
            "step": step,
            "action": action,
            "reward": reward,
            "done": done,
            "active_incidents": len(obs["active_incidents"]),
            "resolved_count": obs["resolved_count"],
        }
        step_logs.append(step_log)

        # ── [STEP] log ──
        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

        if done:
            break

    grade = env_grade()
    return {
        "task": task_name,
        "steps": step,
        "total_reward": round(total_reward, 4),
        "grade_score": grade["score"],
        "grade_breakdown": grade["breakdown"],
        "passed": grade["passed"],
        "step_logs": step_logs,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    start_time = time.time()
    results = {}

    # ── [START] log ──
    for task_name in TASKS:
        print(f"[START] task={task_name}", flush=True)
    sys.stdout.flush()

    for task_name in TASKS:
        try:
            result = run_task(task_name)
            results[task_name] = result
            # ── [END] log per task ──
            print(f"[END] task={task_name} score={result['grade_score']} steps={result['steps']}", flush=True)
        except Exception as e:
            results[task_name] = {
                "task": task_name,
                "error": str(e),
                "grade_score": 0.0,
                "passed": False,
            }
            print(f"[END] task={task_name} score=0.0 steps=0", flush=True)

    elapsed = round(time.time() - start_time, 2)
    avg_score = round(
        sum(r.get("grade_score", 0) for r in results.values()) / len(results), 4
    )

    # Exit code: 0 if all passed, 1 otherwise
    sys.exit(0 if all(r.get("passed", False) for r in results.values()) else 1)


if __name__ == "__main__":
    main()
