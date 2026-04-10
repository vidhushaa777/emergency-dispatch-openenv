import os, json, sys, requests

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS        = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED         = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON: {"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""

def call_llm(api_key, api_base_url, messages):
    """Call LiteLLM proxy directly via HTTP — no OpenAI library needed."""
    url = api_base_url.rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"model": MODEL_NAME, "messages": messages, "temperature": 0.2, "max_tokens": 512},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def get_action(api_key, api_base_url, obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])
    msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg},
    ]
    try:
        raw = call_llm(api_key, api_base_url, messages)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"dispatches": []}

def run_task(api_key, api_base_url, task_name):
    obs = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": SEED}, timeout=30).json()
    step, total_reward = 0, 0.0
    while True:
        step += 1
        action = get_action(api_key, api_base_url, obs)
        result = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()
        reward = result["reward"]["total"]
        total_reward += reward
        print(f"[STEP] step={step} reward={round(reward,4)}", flush=True)
        obs = result["observation"]
        if result["done"]:
            break
    grade = requests.get(f"{ENV_URL}/grade", timeout=30).json()
    return grade["score"], step

def main():
    api_key      = os.environ["API_KEY"]
    api_base_url = os.environ["API_BASE_URL"]

    for task in TASKS:
        print(f"[START] task={task}", flush=True)
        score, steps = 0.0, 0
        try:
            score, steps = run_task(api_key, api_base_url, task)
        except Exception:
            print(f"[STEP] step=0 reward=0.0", flush=True)
        print(f"[END] task={task} score={score} steps={steps}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()




