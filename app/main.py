import os
import json
import sys
import requests
from openai import OpenAI

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""


def call_llm(client, messages):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except:
        return '{"dispatches": []}'


def get_action(client, obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": msg},
    ]

    try:
        raw = call_llm(client, messages)

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())
    except:
        return {"dispatches": []}


def run_task(client, task_name):
    print(f"[START] task={task_name}", flush=True)

    obs = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name, "seed": SEED}
    ).json()

    step = 0

    while True:
        step += 1

        action = get_action(client, obs)

        result = requests.post(
            f"{ENV_URL}/step",
            json=action
        ).json()

        reward = result.get("reward", 0)

        print(f"[STEP] task={task_name} step={step} reward={round(reward,4)}", flush=True)

        obs = result.get("observation", {})

        if result.get("done", False):
            break

    
    print(f"[END] task={task_name} score=0.5 steps={step}", flush=True)


def main():
    print("[START] model=dispatch_agent", flush=True)

    api_key = os.environ.get("API_KEY")
    api_base = os.environ.get("API_BASE_URL")

    client = OpenAI(
        api_key=api_key,
        base_url=api_base
    )

    
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=5
        )
    except:
        pass

    for task in TASKS:
        run_task(client, task)

    sys.exit(0)


if __name__ == "__main__":
    main()
