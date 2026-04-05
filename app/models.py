from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class IncidentType(str, Enum):
    FIRE = "fire"
    MEDICAL = "medical"
    POLICE = "police"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UnitStatus(str, Enum):
    IDLE = "idle"
    RESPONDING = "responding"
    RETURNING = "returning"
    OUT_OF_FUEL = "out_of_fuel"


class Incident(BaseModel):
    id: str
    type: IncidentType
    priority: Priority
    location: str                  # human-readable: "Grid [3,7]"
    description: str               # natural language for LLM
    spawned_step: int
    resolved: bool = False
    assigned_unit: Optional[str] = None


class Unit(BaseModel):
    id: str                        # e.g. "F1", "A2", "P1"
    type: IncidentType
    status: UnitStatus
    fuel: float = Field(ge=0.0, le=100.0)
    location: str
    assigned_incident: Optional[str] = None  # incident id


class Observation(BaseModel):
    """What the agent sees each step."""
    step: int
    task_name: str
    active_incidents: List[Incident]
    units: List[Unit]
    resolved_count: int
    total_spawned: int
    average_response_time: Optional[float]
    episode_reward_so_far: float
    message: str                   # natural language summary for LLM


class DispatchAction(BaseModel):
    """Single dispatch decision."""
    unit_id: str                   # which unit to move
    incident_id: Optional[str]     # which incident to assign (None = return to base)
    reasoning: Optional[str]       # LLM can explain its decision


class Action(BaseModel):
    """Full action: one or more dispatch decisions per step."""
    dispatches: List[DispatchAction]


class Reward(BaseModel):
    """Structured reward with partial signals."""
    total: float = Field(ge=-1.0, le=1.0)   # normalized 0.0–1.0 (can go negative briefly)
    components: Dict[str, float]             # breakdown for transparency
    explanation: str                          # human-readable


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    max_steps: int
    success_threshold: float
