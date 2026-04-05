---
title: Emergency Dispatch OpenEnv
emoji: ??
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---
# Emergency Dispatch OpenEnv

An OpenEnv-compliant RL environment for training and evaluating LLM agents on **real-world emergency dispatch coordination**.

The agent acts as a dispatch coordinator: it reads incoming incident reports (fires, medical emergencies, police calls) and unit statuses, then decides which units to send where. This mirrors what human dispatchers do under time pressure with limited resources.

---

## Why This Domain

Emergency dispatch is a genuine real-world task with:
- **Clear objectives** (resolve incidents before they escalate)
- **Hard constraints** (only fire trucks handle fires, fuel is finite)
- **Meaningful trade-offs** (high priority vs. resource conservation)
- **Measurable outcomes** (response time, resolution rate, triage order)

It is challenging for frontier models because it requires multi-step planning, constraint satisfaction, and prioritization under uncertainty.

---

## Observation Space

Each step the agent receives a JSON `Observation`:

| Field | Type | Description |
|---|---|---|
| `step` | int | Current step number |
| `task_name` | str | Active task identifier |
| `active_incidents` | list | All unresolved incidents |
| `units` | list | All vehicle statuses |
| `resolved_count` | int | Incidents resolved this episode |
| `total_spawned` | int | Total incidents seen |
| `average_response_time` | float | Steps to resolve, averaged |
| `episode_reward_so_far` | float | Cumulative reward |
| `message` | str | Natural language step summary |

Each `Incident` contains: `id`, `type` (fire/medical/police), `priority` (high/medium/low), `location`, `description`, `spawned_step`.

Each `Unit` contains: `id`, `type`, `status` (idle/responding/returning/out_of_fuel), `fuel` (0–100), `location`, `assigned_incident`.

---

## Action Space

The agent submits a JSON `Action` with a list of dispatch decisions:

```json
{
  "dispatches": [
    {
      "unit_id": "F1",
      "incident_id": "F003",
      "reasoning": "High priority fire, F1 is idle and nearest"
    },
    {
      "unit_id": "A1",
      "incident_id": "M001",
      "reasoning": "Cardiac arrest is critical, A1 has fuel"
    }
  ]
}
```

- `unit_id`: one of F1, F2, A1, A2, P1, P2
- `incident_id`: incident to assign, or `null` to return to base
- `reasoning`: optional explanation (used for LLM scoring)

---

## Reward Function

Reward is shaped across the full trajectory (not sparse end-of-episode):

| Event | Reward |
|---|---|
| Resolve HIGH priority incident | +0.20 (×1.5 if resolved in ≤3 steps) |
| Resolve MEDIUM priority incident | +0.12 |
| Resolve LOW priority incident | +0.05 |
| Correct dispatch (matching type) | +0.04–0.08 |
| Wrong unit type dispatched | −0.08 |
| High-priority incident escalates (>8 steps unresolved) | −0.06 |
| Unit runs out of fuel | −0.05 |
| Step penalty | −0.01 to −0.02 |

All values normalized to `[-1.0, 1.0]`. Final episode score from grader is `[0.0, 1.0]`.

---

## Tasks

### Easy — Standard Dispatch
Mixed incident types (fire/medical/police), balanced priorities, full fuel (100%), slow spawn rate. Agent must assign correct types and resolve incidents efficiently.

- Max steps: 40
- Success threshold: 0.55
- Grader weights: resolution rate (40%), correct type rate (30%), response time (30%)

### Medium — Mass Casualty Event
Surge of high-priority medical incidents (65% high priority). Only 2 ambulances available. Agent must triage: serve critical patients first, avoid sending wrong units.

- Max steps: 50
- Success threshold: 0.65
- Grader weights: high-priority resolution (35%), triage order (25%), correct type (25%), fuel efficiency (15%)

### Hard — Resource Scarcity Crisis
All three incident types surge simultaneously. Units start at 35% fuel, each move costs 3× fuel. Agent must balance dispatch vs. conservation, resolve high-priority incidents before 8-step escalation window.

- Max steps: 60
- Success threshold: 0.70
- Grader weights: priority-weighted resolution (30%), escalation avoidance (25%), fuel conservation (25%), multi-type coordination (20%)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks with metadata |
| POST | `/reset` | Reset environment (body: `task_name`, `seed`) |
| POST | `/step` | Submit action, get observation + reward |
| GET | `/state` | Current full state snapshot |
| GET | `/grade` | Grade current episode (0.0–1.0) |

Interactive docs: `http://localhost:7860/docs`

---

## Setup & Usage

### Local

```bash
git clone <your-repo>
cd emergency-dispatch-env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t emergency-dispatch-env .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=your_key \
  -e MODEL_NAME=gpt-4o-mini \
  emergency-dispatch-env
```

### Run Baseline Inference

```bash
export OPENAI_API_KEY=your_key
export MODEL_NAME=gpt-4o-mini
export ENV_URL=http://localhost:7860
export API_BASE_URL=https://api.openai.com/v1

python inference.py
```

Outputs structured `[START]`, `[STEP]`, `[END]` logs to stdout.

---

## Baseline Scores (gpt-4o-mini, seed=42)

| Task | Score | Passed |
|---|---|---|
| standard_dispatch | 0.61 | ✅ |
| mass_casualty | 0.54 | ❌ |
| resource_scarcity | 0.41 | ❌ |
| **Average** | **0.52** | |

The easy task is passable. Medium and hard tasks remain genuinely challenging — this is by design. A well-tuned agent should be able to reach 0.70+ on all three.

---

## Project Structure

```
emergency-dispatch-env/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI server
│   ├── env.py         # Core environment logic
│   ├── models.py      # Pydantic typed models
│   └── tasks.py       # Task configs + graders
├── inference.py       # Baseline inference script
├── openenv.yaml       # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI (or compatible) API key |
| `API_BASE_URL` | LLM API base URL (default: OpenAI) |
| `MODEL_NAME` | Model identifier (default: gpt-4o-mini) |
| `HF_TOKEN` | HuggingFace token (used as fallback API key) |
| `ENV_URL` | URL of the running environment server |

