import os
import json
import requests
from openai import OpenAI

# ✅ FIXED: Docker-safe URL
ENV_URL = os.environ.get("ENV_URL", "http://host.docker.internal:8000")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""

# ✅ OpenAI client
client = OpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get("API_BASE_URL")
)

# ✅ SAFE REQUEST FUNCTION (prevents crashes)
def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"[ERROR] POST {url} -> {e}", flush=True)
        return {}

def safe_get(url):
    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"[ERROR] GET {url} -> {e}", flush=True)
        return {}

# ✅ LLM CALL
def call_llm(messages):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        print("[ERROR] LLM call failed:", e, flush=True)
        return '{"dispatches": []}'


# ✅ ACTION GENERATION
def get_action(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg},
    ]

    raw = call_llm(messages)

    # clean markdown if exists
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw.strip())
    except Exception as e:
        print("[ERROR] JSON parse failed:", e, flush=True)
        return {"dispatches": []}


# ✅ MAIN TASK LOOP
def run_task(task_name):
    print("Using ENV_URL:", ENV_URL, flush=True)

    obs = safe_post(
        f"{ENV_URL}/reset",
        {"task_name": task_name, "seed": SEED}
    )

    step = 0
    total_reward = 0.0

    print(f"[START] task={task_name}", flush=True)

    while True:
        step += 1

        action = get_action(obs)

        result = safe_post(f"{ENV_URL}/step", action)

        reward = result.get("reward", {})
        if isinstance(reward, dict):
            reward = reward.get("total", 0.0)

        try:
            reward = float(reward)
        except:
            reward = 0.0

        total_reward += reward

        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

        obs = result.get("observation", {})

        if result.get("done", False):
            break

    grade = safe_get(f"{ENV_URL}/grade")
    score = grade.get("score", 0)

    print(f"[END] task={task_name} score={score} steps={step}", flush=True)

    return score, step


# ✅ ENTRY POINT
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
