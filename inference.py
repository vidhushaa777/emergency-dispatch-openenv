"""
inference.py — Baseline LLM agent for Emergency Dispatch OpenEnv
Emits [START], [STEP], [END] structured logs to stdout.
"""
import os
import json
import sys
import time
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY", HF_TOKEN) or "dummy-key"
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")
TASKS        = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED         = 42

# Lazy client — avoids crash at module load if key is missing
_client = None
def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        base = API_BASE_URL if "openai" not in API_BASE_URL else None
        _client = OpenAI(api_key=OPENAI_KEY, base_url=base)
    return _client


# ─────────────────────────────────────────────
# ENV CLIENT
# ─────────────────────────────────────────────
def env_reset(task_name, seed=SEED):
    r = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": seed}, timeout=30)
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


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Each step you receive a JSON observation containing:
- active_incidents: list of incidents needing response
- units: list of vehicles with status, fuel level, and type

Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "F001", "reasoning": "..."}]}
If no action needed: {"dispatches": []}
"""

def build_user_prompt(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])
    prompt = f"Step {obs['step']} -- {obs['message']}\n\n"
    if incidents:
        prompt += "ACTIVE INCIDENTS:\n"
        for inc in incidents:
            prompt += f"  [{inc['priority'].upper()}] {inc['id']} | {inc['type']} | {inc['location']}\n"
    else:
        prompt += "No active incidents.\n"
    prompt += "\nUNITS:\n"
    for u in units:
        prompt += f"  {u['id']} ({u['type']}) | status: {u['status']} | fuel: {u['fuel']:.0f}%\n"
    prompt += f"\nReward so far: {obs['episode_reward_so_far']}"
    return prompt


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────
def get_action(obs, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]
    messages.append({"role": "user", "content": build_user_prompt(obs)})
    try:
        response = get_client().chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.2, max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        action = json.loads(raw)
        history.append({"role": "assistant", "content": raw})
        return action
    except Exception:
        fallback = {"dispatches": []}
        history.append({"role": "assistant", "content": json.dumps(fallback)})
        return fallback


# ─────────────────────────────────────────────
# RUN ONE TASK
# ─────────────────────────────────────────────
def run_task(task_name):
    history = []
    obs = env_reset(task_name, seed=SEED)
    step = 0
    total_reward = 0.0

    while True:
        step += 1
        action = get_action(obs, history)
        result = env_step(action)
        reward = result["reward"]["total"]
        done   = result["done"]
        obs    = result["observation"]
        total_reward += reward

        # ── REQUIRED OUTPUT ──
        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

        if done:
            break

    grade = env_grade()
    return {
        "steps": step,
        "grade_score": grade["score"],
        "passed": grade["passed"],
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    for task_name in TASKS:
        # ── REQUIRED OUTPUT ──
        print(f"[START] task={task_name}", flush=True)

        result = {"grade_score": 0.0, "steps": 0, "passed": False}
        try:
            result = run_task(task_name)
        except Exception as e:
            print(f"[STEP] step=0 reward=0.0", flush=True)

        # ── REQUIRED OUTPUT ──
        print(f"[END] task={task_name} score={result['grade_score']} steps={result['steps']}", flush=True)

    sys.exit(0)

if __name__ == "__main__":
    main()


