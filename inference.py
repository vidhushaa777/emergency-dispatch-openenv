import os
import json
import requests

try:
    from openai import OpenAI
except:
    OpenAI = None

# ✅ Same pattern as working code - use .get() not []
client = None
if OpenAI:
    try:
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL"),
            api_key=os.environ.get("API_KEY")
        )
    except:
        client = None

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS      = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED       = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""


def get_llm_action(obs):
    if client is None:
        return {"dispatches": []}
    try:
        incidents = obs.get("active_incidents", [])
        units     = obs.get("units", [])
        prompt    = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"

        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        raw = res.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())
    except:
        return {"dispatches": []}


def run_task(task_name):
    try:
        obs = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "seed": SEED},
            timeout=30
        ).json()

        step = 0
        print(f"[START] task={task_name}", flush=True)

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
        print(f"[END] task={task_name} score={score} steps={step}", flush=True)

    except:
        print(f"[STEP] step=0 reward=0.0", flush=True)
        print(f"[END] task={task_name} score=0 steps=0", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)

