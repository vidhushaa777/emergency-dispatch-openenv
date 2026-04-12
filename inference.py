import os, json, sys, requests
from openai import OpenAI

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

TASKS = ["standard_dispatch", "mass_casualty", "resource_scarcity"]
SEED  = 42

SYSTEM_PROMPT = """You are an emergency dispatch coordinator AI.
Respond ONLY with valid JSON:
{"dispatches": [{"unit_id": "F1", "incident_id": "INC001", "reasoning": "reason"}]}
If no action needed: {"dispatches": []}"""


def call_llm(client, messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content


def get_action(client, obs):
    incidents = obs.get("active_incidents", [])
    units     = obs.get("units", [])
    msg = f"Incidents: {json.dumps(incidents)}\nUnits: {json.dumps(units)}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": msg},
    ]
    # ✅ FIX: wrap call_llm in its own try/except so LLM errors
    #         don't propagate up and kill the episode silently
    try:
        raw = call_llm(client, messages)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return {"dispatches": []}


def run_task(client, task_name):
    obs = requests.post(
        f"{ENV_URL}/reset",
        json={"task_name": task_name, "seed": SEED},
        timeout=30
    ).json()

    step, total_reward = 0, 0.0

    while True:
        step += 1
        action = get_action(client, obs)
        result = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()

        reward = result.get("reward", {})
        if isinstance(reward, dict):
            reward = reward.get("total", 0.0)
        reward = float(reward)
        total_reward += reward

        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

        obs = result.get("observation", {})
        if result.get("done", False):
            break

    grade = requests.get(f"{ENV_URL}/grade", timeout=30).json()
    return grade.get("score", 0), step


def main():
    api_key      = os.environ["API_KEY"]
    api_base_url = os.environ["API_BASE_URL"]

    print(f"DEBUG base_url={api_base_url}", file=sys.stderr, flush=True)

    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url,
    )

    # ✅ FIX: pre-flight LLM check — confirms proxy is reachable
    #         before entering the task loop
    try:
        test = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Reply with: ok"}],
            max_tokens=5,
        )
        print(f"DEBUG LLM preflight OK: {test.choices[0].message.content}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"FATAL: LLM preflight failed — {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        sys.exit(1)  # fail fast and visibly instead of silently completing with 0 calls

    for task in TASKS:
        print(f"[START] task={task}", flush=True)
        score, steps = 0.0, 0
        try:
            score, steps = run_task(client, task)
        except Exception as e:
            print(f"[STEP] step=0 reward=0.0", flush=True)
            print(f"TASK ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"[END] task={task} score={score} steps={steps}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
