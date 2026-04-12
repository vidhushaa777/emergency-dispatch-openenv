import os, json, requests

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:8000")
API_BASE   = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY    = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED  = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""


def get_llm_action(obs):
    try:
        incidents = obs.get("active_incidents", [])
        units     = obs.get("units", [])
        prompt    = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"

        # ✅ Direct HTTP call to LiteLLM proxy — no OpenAI SDK
        res = requests.post(
            f"{API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 512
            },
            timeout=30
        )
        raw = res.json()["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())
    except:
        return {"dispatches": []}


def run_task(task_name):
    print(f"[START] task={task_name}", flush=True)
    step = 0
    score = 0

    try:
        obs = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "seed": SEED},
            timeout=30
        ).json()

        while True:
            step += 1
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

    except:
        if step == 0:
            print(f"[STEP] step=0 reward=0.0", flush=True)

    print(f"[END] task={task_name} score={score} steps={step}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)


