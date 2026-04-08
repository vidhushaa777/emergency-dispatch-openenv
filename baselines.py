import random

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
                    "reasoning": "Greedy priority dispatch"
                })
                used_units.add(unit["id"])
                break

    return {"dispatches": dispatches}