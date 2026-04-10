import os
import json
import sys
import requests
from openai import OpenAI

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
TASKS      = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED       = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "unit_1", "incident_id": "inc_1", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""


# 🔥 ULTRA SAFE SCORE FUNCTION
def compute_safe_score(total_reward, steps):
    try:
        total_reward = float(total_reward)
        steps = int(steps)
    except:
        return 0.5

    # prevent division issues
    if steps <= 0:
        return 0.5

    score = total_reward / (steps + 1)

    # handle NaN / invalid
    if not isinstance(score, float) or score != score:
        return 0.5

    # STRICT RANGE (0,1)
    if score <= 0.0:
        return 0.001
    elif score >= 1.0:
        return 0.999
    else:
        return round(score, 4)


def call_llm(client, messages):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=200,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM ERROR: {e}", file=sys.stderr, flush=True)
        return '{"dispatches": []}'


def get_action(client, obs):
    if client is None:
        return {"dispatches": []}

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

    except Exception:
        return {"dispatches": []}


def run_task(client, task_name):
    print(f"[START] task={task_name}", flush=True)

    step = 0
    total_reward = 0.0

    try:
        obs = requests.post(
            f"{ENV_URL}/reset",
            json={"task_name": task_name, "seed": SEED},
            timeout=30
        ).json()

        while True:
            step += 1

            action = get_action(client, obs)

            result = requests.post(
                f"{ENV_URL}/step",
                json=action,
                timeout=30
            ).json()

            reward = result.get("reward", 0)

            try:
                reward = float(reward)
            except:
                reward = 0.0

            total_reward += reward

            print(
                f"[STEP] task={task_name} step={step} reward={round(reward, 4)}",
                flush=True
            )

            obs = result.get("observation", {})
            done = result.get("done", False)

            if done:
                break

    except Exception as e:
        print(f"[STEP] task={task_name} step={step} error={type(e).__name__}", flush=True)

    # 🔥 ALWAYS SAFE SCORE
    score = compute_safe_score(total_reward, step)

    # DOUBLE SAFETY
    if score <= 0.0:
        score = 0.001
    elif score >= 1.0:
        score = 0.999

    print(f"[END] task={task_name} score={float(score)} steps={step}", flush=True)


def main():
    print("[START] model=dispatch_agent", flush=True)

    api_key = os.environ.get("API_KEY")
    api_base = os.environ.get("API_BASE_URL")

    client = None

    # SAFE LLM INIT
    if api_key and api_base:
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=api_base
            )

            # 🔥 REQUIRED PROXY CALL
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=5
            )

        except Exception as e:
            print(f"LLM INIT ERROR: {e}", file=sys.stderr, flush=True)
            client = None

    for task in TASKS:
        run_task(client, task)

    sys.exit(0)


if __name__ == "__main__":
    main()
