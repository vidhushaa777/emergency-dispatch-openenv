"""
Task definitions + deterministic graders.
Each grader returns a score in [0.0, 1.0].
"""
from dataclasses import dataclass, field
from typing import List, Dict
from app.models import IncidentType, Priority


@dataclass
class TaskConfig:
    name: str
    description: str
    difficulty: str                        # easy | medium | hard
    max_steps: int
    max_incidents: int
    spawn_prob: float
    priority_weights: Dict[str, float]     # high/medium/low
    forced_type: str | None                # None = mixed
    fuel_cost_multiplier: float
    initial_fuel: float
    success_threshold: float               # min grader score to "pass"
    resource_scarcity: bool = False
    movement_penalty: bool = False


TASKS: Dict[str, TaskConfig] = {

    # ─────────────────────────────────────────────
    # EASY: Standard mixed dispatch
    # Agent must assign correct unit types and resolve
    # incidents before they time out. Generous fuel,
    # slow spawn rate, balanced priorities.
    # ─────────────────────────────────────────────
    "standard_dispatch": TaskConfig(
        name="Standard Dispatch",
        description=(
            "Coordinate fire, medical, and police units to respond to incoming "
            "incidents across the city. Assign the right unit type to each incident "
            "and resolve them before they escalate. Fuel is plentiful, incidents "
            "arrive at a manageable rate."
        ),
        difficulty="easy",
        max_steps=40,
        max_incidents=5,
        spawn_prob=0.25,
        priority_weights={"high": 0.25, "medium": 0.50, "low": 0.25},
        forced_type=None,
        fuel_cost_multiplier=1.0,
        initial_fuel=100.0,
        success_threshold=0.55,
    ),

    # ─────────────────────────────────────────────
    # MEDIUM: Mass casualty event
    # Mostly high-priority medical incidents flood in.
    # Agent must triage: serve high-priority first,
    # manage limited ambulance availability, don't
    # send fire trucks to medical calls.
    # ─────────────────────────────────────────────
    "mass_casualty": TaskConfig(
        name="Mass Casualty Event",
        description=(
            "A major accident has caused a surge of medical emergencies across the city. "
            "Most incidents are high-priority. You have only 2 ambulances — triage carefully. "
            "Sending the wrong unit type wastes time and costs lives. "
            "Prioritize critical patients over stable ones."
        ),
        difficulty="medium",
        max_steps=50,
        max_incidents=9,
        spawn_prob=0.55,
        priority_weights={"high": 0.65, "medium": 0.30, "low": 0.05},
        forced_type="medical",
        fuel_cost_multiplier=1.5,
        initial_fuel=100.0,
        success_threshold=0.65,
        resource_scarcity=False,
    ),

    # ─────────────────────────────────────────────
    # HARD: Resource scarcity multi-crisis
    # All three incident types flood in simultaneously.
    # Units start with 35% fuel, refueling costs 1 step.
    # Agent must balance: dispatch vs. conserve fuel,
    # prioritize high-severity incidents, avoid sending
    # depleted units on long routes. Frontier-model hard.
    # ─────────────────────────────────────────────
    "resource_scarcity": TaskConfig(
        name="Resource Scarcity Crisis",
        description=(
            "A city-wide emergency has overwhelmed all services simultaneously. "
            "Fire, medical, and police incidents are all surging. "
            "Units start at 35% fuel — every move costs more. "
            "You must decide when to dispatch vs. hold back to refuel. "
            "High-priority incidents left unresolved escalate after 8 steps and "
            "incur heavy penalties. Wrong-type dispatches are penalized."
        ),
        difficulty="hard",
        max_steps=60,
        max_incidents=12,
        spawn_prob=0.65,
        priority_weights={"high": 0.55, "medium": 0.30, "low": 0.15},
        forced_type=None,
        fuel_cost_multiplier=3.0,
        initial_fuel=35.0,
        success_threshold=0.70,
        resource_scarcity=True,
        movement_penalty=True,
    ),
}


# ─────────────────────────────────────────────────────────
# GRADERS — deterministic, reproducible, 0.0–1.0 output
# ─────────────────────────────────────────────────────────

def grade_standard_dispatch(episode_log: dict) -> tuple[float, dict]:
    resolved        = episode_log.get("resolved_count", 0)
    spawned         = episode_log.get("total_spawned", 1)
    wrong_type      = episode_log.get("wrong_type_dispatches", 0)
    total_dispatches= episode_log.get("total_dispatches", 1)
    resp_times      = episode_log.get("response_times", [])

    resolution_rate = min(resolved / max(spawned, 1), 0.999)
    resolution_rate = max(resolution_rate, 0.001)

    correct_rate = 1.0 - (wrong_type / max(total_dispatches, 1))
    correct_rate = max(0.001, min(0.999, correct_rate))

    if resp_times:
        avg_rt = sum(resp_times) / len(resp_times)
        rt_score = max(0.001, min(0.999, 1.0 - (avg_rt - 5) / 15))
    else:
        rt_score = 0.001

    total = 0.40 * resolution_rate + 0.30 * correct_rate + 0.30 * rt_score
    total = round(min(max(total, 0.001), 0.999), 4)

    breakdown = {
        "resolution_rate": round(resolution_rate, 4),
        "correct_type_rate": round(correct_rate, 4),
        "response_time_score": round(rt_score, 4),
        "weighted_total": total,
    }
    return total, breakdown


def grade_mass_casualty(episode_log: dict) -> tuple[float, dict]:
    high_resolved   = episode_log.get("high_priority_resolved", 0)
    high_spawned    = episode_log.get("high_priority_spawned", 1)
    triage_score    = episode_log.get("triage_order_score", 0.0)
    wrong_type      = episode_log.get("wrong_type_dispatches", 0)
    total_dispatches= episode_log.get("total_dispatches", 1)
    fuel_used       = episode_log.get("total_fuel_used", 1)
    resolved        = episode_log.get("resolved_count", 0)

    hi_rate = max(0.001, min(0.999, high_resolved / max(high_spawned, 1)))
    correct_rate = max(0.001, min(0.999, 1.0 - (wrong_type / max(total_dispatches, 1))))
    fuel_eff = max(0.001, min(0.999, resolved / max(fuel_used / 20.0, 1)))
    triage_score = max(0.001, min(0.999, triage_score))

    total = (
        0.35 * hi_rate +
        0.25 * triage_score +
        0.25 * correct_rate +
        0.15 * fuel_eff
    )
    total = round(min(max(total, 0.001), 0.999), 4)

    breakdown = {
        "high_priority_resolution": round(hi_rate, 4),
        "triage_order_score": round(triage_score, 4),
        "correct_type_rate": round(correct_rate, 4),
        "fuel_efficiency": round(fuel_eff, 4),
        "weighted_total": total,
    }
    return total, breakdown


def grade_resource_scarcity(episode_log: dict) -> tuple[float, dict]:
    hi_res  = episode_log.get("high_priority_resolved", 0)
    med_res = episode_log.get("medium_priority_resolved", 0)
    lo_res  = episode_log.get("low_priority_resolved", 0)
    hi_sp   = episode_log.get("high_priority_spawned", 1)
    med_sp  = episode_log.get("medium_priority_spawned", 1)
    lo_sp   = episode_log.get("low_priority_spawned", 1)

    pw_score = (
        0.6 * (hi_res / max(hi_sp, 1)) +
        0.3 * (med_res / max(med_sp, 1)) +
        0.1 * (lo_res / max(lo_sp, 1))
    )
    pw_score = max(0.001, min(0.999, pw_score))

    escalated  = episode_log.get("escalated_incidents", 0)
    total_high = max(hi_sp, 1)
    esc_score  = max(0.001, min(0.999, 1.0 - min(escalated / total_high, 1.0)))

    units_depleted = episode_log.get("units_depleted", 0)
    total_units    = episode_log.get("total_units", 6)
    fuel_score     = max(0.001, min(0.999, 1.0 - (units_depleted / max(total_units, 1))))

    types_served = episode_log.get("incident_types_served", [])
    type_score   = max(0.001, min(0.999, len(types_served) / 3.0))

    total = (
        0.30 * pw_score +
        0.25 * esc_score +
        0.25 * fuel_score +
        0.20 * type_score
    )
    total = round(min(max(total, 0.001), 0.999), 4)

    breakdown = {
        "priority_weighted_resolution": round(pw_score, 4),
        "escalation_avoidance": round(esc_score, 4),
        "fuel_conservation": round(fuel_score, 4),
        "multi_type_coordination": round(type_score, 4),
        "weighted_total": total,
    }
    return total, breakdown


GRADERS = {
    "standard_dispatch": grade_standard_dispatch,
    "mass_casualty": grade_mass_casualty,
    "resource_scarcity": grade_resource_scarcity,
}

