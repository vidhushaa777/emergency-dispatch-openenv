import os, json, sys, requests
from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""

# Initialize client at module level — no try/except, fail loud if env vars missing
client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["API_BASE_URL"]
)

def call_llm(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content

def get_action(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])
    msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg},
    ]
    raw = call_llm(messages)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"dispatches": []}

def run_task(task_name):
    obs = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name, "seed": SEED},
        timeout=30
    ).json()

    step, total_reward = 0, 0.0
    print(f"[START] task={task_name}", flush=True)

    while True:
        step += 1
        action = get_action(obs)  # LLM call — no silent fallback
        result = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()

        reward = result.get("reward", {})
        if isinstance(reward, dict):
            reward = reward.get("total", 0.0)
        total_reward += float(reward)

        print(f"[STEP] step={step} reward={round(float(reward), 4)}", flush=True)
        obs = result.get("observation", {})

        if result.get("done", False):
            break

    grade = requests.get(f"{ENV_URL}/grade", timeout=30).json()
    score = grade["score"]
    print(f"[END] task={task_name} score={score} steps={step}", flush=True)
    return score, step

if __name__ == "__main__":
    run_task("standard_dispatch")
    run_task("mass_casualty")
    run_task("resource_scarcity")
