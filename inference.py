import os
import sys
import json
import requests

try:
    from openai import OpenAI
except Exception as e:
    print(f"[DEBUG] OpenAI import failed: {e}", file=sys.stderr, flush=True)
    OpenAI = None

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_BASE   = os.environ.get("API_BASE_URL", "")
API_KEY    = os.environ.get("API_KEY", "")

TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED  = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""

# ✅ Debug: print what credentials we actually received
print(f"[DEBUG] API_BASE_URL={API_BASE}", file=sys.stderr, flush=True)
print(f"[DEBUG] API_KEY prefix={API_KEY[:8] if API_KEY else 'MISSING'}", file=sys.stderr, flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr, flush=True)
print(f"[DEBUG] ENV_URL={ENV_URL}", file=sys.stderr, flush=True)

# ✅ Initialize client
client = None
if OpenAI and API_BASE and API_KEY:
    try:
        client = OpenAI(
            base_url=API_BASE,
            api_key=API_KEY
        )
        print(f"[DEBUG] OpenAI client initialized OK", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[DEBUG] OpenAI client init failed: {e}", file=sys.stderr, flush=True)
else:
    print(f"[DEBUG] Skipping client init: OpenAI={OpenAI is not None} API_BASE={bool(API_BASE)} API_KEY={bool(API_KEY)}", file=sys.stderr, flush=True)


def get_llm_action(obs):
    if client is None:
        print(f"[DEBUG] client is None, skipping LLM call", file=sys.stderr, flush=True)
        return {"dispatches": []}
    try:
        incidents = obs.get("active_incidents", [])
        units     = obs.get("units", [])
        prompt    = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"

        print(f"[DEBUG] Calling LLM...", file=sys.stderr, flush=True)
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        print(f"[DEBUG] LLM call success", file=sys.stderr, flush=True)
        raw = res.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as e:
        # ✅ Now we can SEE what error is occurring
        print(f"[DEBUG] LLM call FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return {"dispatches": []}


def run_task(task_name):
    print(f"[START] task={task_name}", flush=True)
    step  = 0
    score = 0

    try:
        obs = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "seed": SEED},
            timeout=30
        ).json()

        while True:
            step  += 1
            action = get_llm_action(obs)
            result = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()

            reward = result.get("reward", {})
            if isinstance(reward, dict):
                reward = reward.get("total", 0.0)
            reward = float(reward)

            print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

            obs = result.get("observation", {})
            if result.get("done", False):
                break

        grade = requests.get(f"{ENV_URL}/grade", timeout=30).json()
        score = grade.get("score", 0)

    except Exception as e:
        print(f"[DEBUG] run_task error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if step == 0:
            print(f"[STEP] step=0 reward=0.0", flush=True)

    print(f"[END] task={task_name} score={score} steps={step}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)


