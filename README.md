---

title: Emergency Dispatch OpenEnv
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 🚑 Emergency Dispatch OpenEnv

An OpenEnv-compliant Reinforcement Learning (RL) environment for training and evaluating intelligent agents on **real-world emergency dispatch coordination**.

The agent acts as a dispatcher: it receives incident reports (fires, medical emergencies, police calls) and unit statuses, and decides which units to send — similar to real-world emergency response systems.

---

## 🌍 Why This Domain

Emergency dispatch is a realistic and challenging domain:

* **Clear objectives** → Resolve incidents quickly
* **Hard constraints** → Only specific units handle specific incidents
* **Trade-offs** → Speed vs resource usage
* **Measurable outcomes** → Response time, accuracy, efficiency

This makes it ideal for evaluating decision-making agents.

---

## 🧠 Why This Environment is Suitable for RL

This environment captures key RL characteristics:

* **Sequential decision-making** → Actions affect future states
* **Delayed rewards** → Efficient dispatch improves long-term outcomes
* **Constraints** → Limited fuel, unit types, and availability
* **Multi-objective optimization** → Balance speed, correctness, and resource usage

---

## 📥 Observation Space

Each step returns a structured JSON observation:

| Field                 | Description                   |
| --------------------- | ----------------------------- |
| step                  | Current timestep              |
| active_incidents      | List of unresolved incidents  |
| units                 | Status of all emergency units |
| resolved_count        | Number of resolved incidents  |
| total_spawned         | Total incidents generated     |
| average_response_time | Average response delay        |
| episode_reward_so_far | Cumulative reward             |

---

## 📤 Action Space

The agent submits dispatch decisions:

```json
{
  "dispatches": [
    {
      "unit_id": "F1",
      "incident_id": "F003",
      "reasoning": "High priority fire"
    }
  ]
}
```

---

## 🎯 Reward Function

| Event                 | Reward     |
| --------------------- | ---------- |
| Resolve HIGH priority | +0.20      |
| Resolve MEDIUM        | +0.12      |
| Resolve LOW           | +0.05      |
| Correct dispatch      | +0.04–0.08 |
| Wrong dispatch        | −0.08      |
| Delay penalty         | −0.01      |
| Fuel exhaustion       | −0.05      |

---

## 🧪 Tasks

### 🟢 Standard Dispatch (Easy)

Balanced incidents, full resources
**Goal:** Efficient assignment

### 🟡 Mass Casualty (Medium)

High-priority surge, limited ambulances
**Goal:** Smart triage

### 🔴 Resource Scarcity (Hard)

Low fuel, multiple incident types
**Goal:** Optimize resource usage

---

## 🔌 API Endpoints

| Method | Endpoint | Description       |
| ------ | -------- | ----------------- |
| GET    | /health  | Health check      |
| POST   | /reset   | Reset environment |
| POST   | /step    | Take action       |
| GET    | /state   | Current state     |
| GET    | /grade   | Final evaluation  |

Docs: `/docs`

---

## 🖥️ UI Dashboard

A built-in dashboard is served via FastAPI:

* Real-time incident tracking
* Unit monitoring
* Reward visualization
* API interaction

👉 Access:
http://localhost:8000/

---

## 🚀 Setup

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

---

## 🤖 Agent Evaluation

We compare three agents:

* **Random Agent** → baseline
* **Greedy Agent** → priority-based
* **Smart Agent** → optimized decision logic

---

## 📊 Results (standard_dispatch)

| Method | Score | Steps | Reward | Performance     |
| ------ | ----- | ----- | ------ | --------------- |
| Random | 0.65  | 40    | -2.28  | Baseline        |
| Greedy | 0.90  | 40    | 0.10   | High Efficiency |
| Smart  | 0.90  | 40    | 0.10   | Near Optimal    |

---

## 📈 Insights

* Random agent performs poorly → validates environment difficulty
* Greedy strategy achieves near-optimal performance (~0.90)
* Smart agent performs similarly → greedy policy is already highly effective

👉 This shows strong **reward alignment and environment design**

---

## 💡 Innovation

This project uses a priority-aware dispatch strategy that considers:

* Incident severity
* Resource availability
* System efficiency

This improves decision-making compared to naive approaches.

---

## 📂 Project Structure

```
emergency-dispatch-env/
├── app/
│   ├── main.py
│   ├── env.py
│   ├── models.py
│   └── tasks.py
├── static/
│   └── index.html
├── inference.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🔐 Environment Variables

| Variable       | Description        |
| -------------- | ------------------ |
| OPENAI_API_KEY | API key (optional) |
| MODEL_NAME     | Model name         |
| API_BASE_URL   | API endpoint       |
| ENV_URL        | Environment URL    |

---

## 🔮 Future Work

* Integrate LLM-based agents
* Add multi-agent coordination
* Improve real-time visualization
* Extend to real-world datasets

---

## 🏁 Conclusion

This project demonstrates a realistic and scalable RL environment for emergency response coordination.

It highlights how intelligent decision-making significantly improves performance over naive strategies, making it a strong benchmark for evaluating AI agents.



