"""
EmergencyDispatchEnv — core environment logic.
Implements the OpenEnv interface: reset(), step(), state()
"""
import random
import uuid
from typing import Optional
from app.models import (
    Incident, Unit, Observation, Action, Reward, StepResult,
    IncidentType, Priority, UnitStatus
)
from app.tasks import TaskConfig, TASKS, GRADERS


GRID_SIZE = 10
LOCATIONS = [f"Grid [{r},{c}]" for r in range(GRID_SIZE) for c in range(GRID_SIZE)]

INCIDENT_DESCRIPTIONS = {
    ("fire", "high"):    "Structure fire with reported occupants trapped. Immediate response required.",
    ("fire", "medium"):  "Vehicle fire on main road. Risk of fuel explosion.",
    ("fire", "low"):     "Grass fire near park boundary. Contained but growing.",
    ("medical", "high"): "Cardiac arrest reported. Patient unresponsive, bystander CPR in progress.",
    ("medical", "medium"): "Road accident with injuries. Two patients, ambulatory.",
    ("medical", "low"):  "Elderly fall at home. Patient stable, possible fracture.",
    ("police", "high"):  "Armed robbery in progress. Shots fired reported.",
    ("police", "medium"): "Domestic disturbance. Neighbors report shouting and breaking sounds.",
    ("police", "low"):   "Noise complaint — repeated violation at same address.",
}

BASE_LOCATIONS = {
    "fire":    "Fire HQ [0,0]",
    "medical": "Med Center [9,9]",
    "police":  "Police HQ [9,0]",
}


def manhattan(a: str, b: str) -> int:
    """Distance between two 'Grid [r,c]' strings."""
    def parse(loc):
        if "HQ" in loc or "Center" in loc:
            coords = loc.split("[")[1].rstrip("]").split(",")
        else:
            coords = loc.replace("Grid [", "").rstrip("]").split(",")
        return int(coords[0]), int(coords[1])
    ar, ac = parse(a)
    br, bc = parse(b)
    return abs(ar - br) + abs(ac - bc)


class EmergencyDispatchEnv:
    def __init__(self, task_name: str = "standard_dispatch", seed: int = 42):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task_name  = task_name
        self.task       = TASKS[task_name]
        self.seed       = seed
        self._rng       = random.Random(seed)
        self._state     = {}
        self.reset()

    # ─────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────
    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)  # reproducible
        fuel = self.task.initial_fuel

        self._units = {
            "F1": Unit(id="F1", type=IncidentType.FIRE,    status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["fire"]),
            "F2": Unit(id="F2", type=IncidentType.FIRE,    status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["fire"]),
            "A1": Unit(id="A1", type=IncidentType.MEDICAL, status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["medical"]),
            "A2": Unit(id="A2", type=IncidentType.MEDICAL, status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["medical"]),
            "P1": Unit(id="P1", type=IncidentType.POLICE,  status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["police"]),
            "P2": Unit(id="P2", type=IncidentType.POLICE,  status=UnitStatus.IDLE, fuel=fuel, location=BASE_LOCATIONS["police"]),
        }

        self._incidents: dict[str, Incident] = {}
        self._step = 0
        self._done = False
        self._episode_reward = 0.0

        # Episode log for graders
        self._log = {
            "resolved_count": 0,
            "total_spawned": 0,
            "response_times": [],
            "wrong_type_dispatches": 0,
            "total_dispatches": 0,
            "high_priority_resolved": 0,
            "medium_priority_resolved": 0,
            "low_priority_resolved": 0,
            "high_priority_spawned": 0,
            "medium_priority_spawned": 0,
            "low_priority_spawned": 0,
            "triage_order_score": 1.0,
            "escalated_incidents": 0,
            "total_fuel_used": 0.0,
            "units_depleted": 0,
            "total_units": len(self._units),
            "incident_types_served": set(),
            "resolution_order": [],     # list of priorities in order resolved
        }

        # Spawn 2 initial incidents so agent has something to act on immediately
        for _ in range(2):
            self._spawn_incident()

        return self._build_observation("Episode reset. Dispatch units to active incidents.")

    # ─────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1
        reward_components = {}
        reward_total = 0.0

        # 1. Process agent dispatches
        for dispatch in action.dispatches:
            self._log["total_dispatches"] += 1
            unit = self._units.get(dispatch.unit_id)
            if not unit:
                reward_components["invalid_unit"] = reward_components.get("invalid_unit", 0) - 0.02
                continue

            if dispatch.incident_id is None:
                # Return to base
                unit.status = UnitStatus.RETURNING
                unit.assigned_incident = None
                unit.location = BASE_LOCATIONS[unit.type.value]
                reward_components["return_to_base"] = 0.0
                continue

            incident = self._incidents.get(dispatch.incident_id)
            if not incident or incident.resolved:
                reward_components["stale_incident"] = reward_components.get("stale_incident", 0) - 0.01
                continue

            # Wrong type dispatch penalty
            if unit.type != incident.type:
                self._log["wrong_type_dispatches"] += 1
                reward_components["wrong_type"] = reward_components.get("wrong_type", 0) - 0.08
                continue

            # Valid dispatch
            unit.status = UnitStatus.RESPONDING
            unit.assigned_incident = incident.id
            incident.assigned_unit = unit.id

            # Compute travel steps needed
            dist = manhattan(unit.location, incident.location)
            travel_steps = max(1, dist // 2)

            # Fuel cost
            fuel_cost = travel_steps * self.task.fuel_cost_multiplier
            unit.fuel = max(0.0, unit.fuel - fuel_cost)
            self._log["total_fuel_used"] += fuel_cost

            if unit.fuel <= 0:
                unit.status = UnitStatus.OUT_OF_FUEL
                self._log["units_depleted"] += 1
                reward_components["fuel_depleted"] = reward_components.get("fuel_depleted", 0) - 0.05
                continue

            # Immediate dispatch reward (partial signal)
            pri_bonus = {"high": 0.08, "medium": 0.04, "low": 0.01}[incident.priority.value]
            reward_components["dispatch_bonus"] = reward_components.get("dispatch_bonus", 0) + pri_bonus

            # Resolve incident
            incident.resolved = True
            unit.location = incident.location
            unit.status = UnitStatus.RETURNING
            unit.assigned_incident = None

            resp_time = self._step - incident.spawned_step
            self._log["response_times"].append(resp_time)
            self._log["resolved_count"] += 1
            self._log["incident_types_served"].add(incident.type.value)
            self._log[f"{incident.priority.value}_priority_resolved"] += 1
            self._log["resolution_order"].append(incident.priority.value)

            # Resolution reward: priority × speed
            resolution_reward = {"high": 0.20, "medium": 0.12, "low": 0.05}[incident.priority.value]
            if resp_time <= 3:
                resolution_reward *= 1.5   # fast bonus
            elif resp_time > 10:
                resolution_reward *= 0.6   # slow penalty
            reward_components["resolution"] = reward_components.get("resolution", 0) + resolution_reward

        # 2. Spawn new incidents
        if (self._rng.random() < self.task.spawn_prob
                and len([i for i in self._incidents.values() if not i.resolved]) < self.task.max_incidents):
            self._spawn_incident()

        # 3. Escalation check — high-priority unresolved > 8 steps
        unresolved = [i for i in self._incidents.values() if not i.resolved]
        for inc in unresolved:
            age = self._step - inc.spawned_step
            if inc.priority == Priority.HIGH and age > 8:
                self._log["escalated_incidents"] += 1
                reward_components["escalation"] = reward_components.get("escalation", 0) - 0.06

        # 4. Step penalty (encourage efficiency)
        step_pen = -0.01
        if self.task.movement_penalty:
            step_pen = -0.02
        reward_components["step_penalty"] = step_pen

        # 5. Triage order scoring
        order = self._log["resolution_order"]
        if len(order) >= 2:
            score = self._compute_triage_score(order)
            self._log["triage_order_score"] = score

        # 6. Normalize total reward to [-1, 1]
        raw = sum(reward_components.values())
        normalized = max(-1.0, min(1.0, raw))
        self._episode_reward += normalized

        reward = Reward(
            total=round(normalized, 4),
            components={k: round(v, 4) for k, v in reward_components.items()},
            explanation=self._reward_explanation(reward_components)
        )

        # Episode done?
        self._done = (
            self._step >= self.task.max_steps or
            all(u.fuel <= 0 for u in self._units.values())
        )

        obs = self._build_observation(
            f"Step {self._step}/{self.task.max_steps}. "
            f"{len(unresolved)} active incidents. "
            f"Reward this step: {normalized:.3f}."
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "step": self._step,
                "episode_reward": round(self._episode_reward, 4),
                "episode_log": self._serializable_log(),
            }
        )

    # ─────────────────────────────────────────────
    # STATE
    # ─────────────────────────────────────────────
    def state(self) -> dict:
        return {
            "task": self.task_name,
            "step": self._step,
            "done": self._done,
            "episode_reward": round(self._episode_reward, 4),
            "units": {k: v.model_dump() for k, v in self._units.items()},
            "incidents": {k: v.model_dump() for k, v in self._incidents.items()},
            "log": self._serializable_log(),
        }

    # ─────────────────────────────────────────────
    # GRADE
    # ─────────────────────────────────────────────
    def grade(self) -> tuple[float, dict]:
        grader = GRADERS[self.task_name]
        return grader(self._serializable_log())

    # ─────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────
    def _spawn_incident(self):
        pw = self.task.priority_weights
        r = self._rng.random()
        if r < pw["high"]:
            priority = Priority.HIGH
        elif r < pw["high"] + pw["medium"]:
            priority = Priority.MEDIUM
        else:
            priority = Priority.LOW

        if self.task.forced_type:
            inc_type = IncidentType(self.task.forced_type)
        else:
            inc_type = self._rng.choice(list(IncidentType))

        location = self._rng.choice(LOCATIONS)
        desc = INCIDENT_DESCRIPTIONS.get(
            (inc_type.value, priority.value),
            "Emergency reported. Respond immediately."
        )
        inc_id = f"{inc_type.value[0].upper()}{self._log['total_spawned'] + 1:03d}"

        incident = Incident(
            id=inc_id,
            type=inc_type,
            priority=priority,
            location=location,
            description=desc,
            spawned_step=self._step,
        )
        self._incidents[inc_id] = incident
        self._log["total_spawned"] += 1
        self._log[f"{priority.value}_priority_spawned"] += 1

    def _build_observation(self, message: str) -> Observation:
        active = [i for i in self._incidents.values() if not i.resolved]
        rt = self._log["response_times"]
        avg_rt = round(sum(rt) / len(rt), 2) if rt else None

        return Observation(
            step=self._step,
            task_name=self.task_name,
            active_incidents=active,
            units=list(self._units.values()),
            resolved_count=self._log["resolved_count"],
            total_spawned=self._log["total_spawned"],
            average_response_time=avg_rt,
            episode_reward_so_far=round(self._episode_reward, 4),
            message=message,
        )

    def _compute_triage_score(self, order: list) -> float:
        """Score whether the agent resolved high-priority incidents before low ones."""
        pri_val = {"high": 3, "medium": 2, "low": 1}
        inversions = 0
        n = len(order)
        for i in range(n):
            for j in range(i + 1, n):
                if pri_val[order[i]] < pri_val[order[j]]:
                    inversions += 1
        max_inv = n * (n - 1) / 2
        return round(1.0 - inversions / max(max_inv, 1), 4)

    def _reward_explanation(self, components: dict) -> str:
        parts = []
        if components.get("resolution", 0) > 0:
            parts.append(f"resolved incident (+{components['resolution']:.2f})")
        if components.get("wrong_type", 0) < 0:
            parts.append(f"wrong unit type ({components['wrong_type']:.2f})")
        if components.get("escalation", 0) < 0:
            parts.append(f"incident escalated ({components['escalation']:.2f})")
        if components.get("fuel_depleted", 0) < 0:
            parts.append("unit ran out of fuel (-0.05)")
        parts.append(f"step penalty ({components.get('step_penalty', -0.01):.2f})")
        return "; ".join(parts) if parts else "no significant events this step"

    def _serializable_log(self) -> dict:
        log = dict(self._log)
        log["incident_types_served"] = list(log.get("incident_types_served", set()))
        return log
