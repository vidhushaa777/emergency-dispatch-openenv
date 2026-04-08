import requests
import random

API_URL = "http://localhost:8000"
TASK = "standard_dispatch"


# ─────────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────────
def run_episode(agent_func):
    res = requests.post(f"{API_URL}/reset", json={"task_name": TASK})
    res.raise_for_status()
    obs = res.json()

    total_reward = 0
    steps = 0

    while True:
        action = agent_func(obs)

        res = requests.post(f"{API_URL}/step", json=action)
        res.raise_for_status()
        data = res.json()

        reward = data.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("total", 0)

        total_reward += reward
        steps += 1

        obs = data["observation"]

        if data.get("done"):
            break

    grade = requests.get(f"{API_URL}/grade").json()

    return {
        "reward": round(total_reward, 2),
        "steps": steps,
        "score": round(grade["score"], 4)
    }


# ─────────────────────────────────────────────
# RANDOM AGENT
# ─────────────────────────────────────────────
def random_agent(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    dispatches = []

    for unit in units:
        if unit["status"] == "idle" and unit["fuel"] > 10:
            if incidents:
                inc = random.choice(incidents)
                dispatches.append({
                    "unit_id": unit["id"],
                    "incident_id": inc["id"],
                    "reasoning": "Random assignment"
                })

    return {"dispatches": dispatches}


# ─────────────────────────────────────────────
# GREEDY AGENT
# ─────────────────────────────────────────────
def greedy_agent(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    dispatches = []

    priority_map = {"high": 3, "medium": 2, "low": 1}
    incidents = sorted(incidents, key=lambda x: priority_map[x["priority"]], reverse=True)

    used_units = set()

    for inc in incidents:
        for unit in units:
            if (
                unit["type"] == inc["type"]
                and unit["status"] == "idle"
                and unit["fuel"] > 10
                and unit["id"] not in used_units
            ):
                dispatches.append({
                    "unit_id": unit["id"],
                    "incident_id": inc["id"],
                    "reasoning": "Greedy dispatch"
                })
                used_units.add(unit["id"])
                break

    return {"dispatches": dispatches}


# ─────────────────────────────────────────────
# SMART AGENT (LLM SUBSTITUTE 🔥)
# ─────────────────────────────────────────────
def smart_agent(obs):
    incidents = obs.get("active_incidents", [])
    units = obs.get("units", [])

    dispatches = []

    priority_map = {"high": 3, "medium": 2, "low": 1}
    incidents = sorted(incidents, key=lambda x: priority_map[x["priority"]], reverse=True)

    used_units = set()

    for inc in incidents:
        best_unit = None
        best_fuel = -1

        for unit in units:
            if (
                unit["type"] == inc["type"]
                and unit["status"] == "idle"
                and unit["fuel"] > 10
                and unit["id"] not in used_units
            ):
                if unit["fuel"] > best_fuel:
                    best_unit = unit
                    best_fuel = unit["fuel"]

        if best_unit:
            dispatches.append({
                "unit_id": best_unit["id"],
                "incident_id": inc["id"],
                "reasoning": "Smart fuel-optimized dispatch"
            })
            used_units.add(best_unit["id"])

    return {"dispatches": dispatches}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("Running Random Agent...")
    r1 = run_episode(random_agent)

    print("Running Greedy Agent...")
    r2 = run_episode(greedy_agent)

    print("Running Smart Agent...")
    r3 = run_episode(smart_agent)

    print("\n📊 RESULTS COMPARISON")
    print("-" * 50)
    print(f"{'Method':<10} {'Score':<10} {'Steps':<10} {'Reward':<10}")
    print("-" * 50)
    print(f"{'Random':<10} {r1['score']:<10} {r1['steps']:<10} {r1['reward']:<10}")
    print(f"{'Greedy':<10} {r2['score']:<10} {r2['steps']:<10} {r2['reward']:<10}")
    print(f"{'Smart':<10} {r3['score']:<10} {r3['steps']:<10} {r3['reward']:<10}")
    print("-" * 50)


if __name__ == "__main__":
    main()