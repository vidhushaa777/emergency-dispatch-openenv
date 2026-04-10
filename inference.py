import os, json, sys, requests

ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_KEY   = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", HF_TOKEN))
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED  = 42

_client = None
def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_KEY, base_url=API_BASE_URL)
    return _client

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON: {"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""

def get_action(obs):
    try:
        incidents = obs.get("active_incidents", [])
        units = obs.get("units", [])
        msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"
        response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
            temperature=0.2, max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"dispatches": []}

def run_task(task_name):
    obs = requests.post(f"{ENV_URL}/reset", json={"task_name": task_name, "seed": SEED}, timeout=30).json()
    step, total_reward = 0, 0.0
    while True:
        step += 1
        action = get_action(obs)
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
    for task in TASKS:
        print(f"[START] task={task}", flush=True)
        score, steps = 0.0, 0
        try:
            score, steps = run_task(task)
        except Exception:
            print(f"[STEP] step=0 reward=0.0", flush=True)
        print(f"[END] task={task} score={score} steps={steps}", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()



