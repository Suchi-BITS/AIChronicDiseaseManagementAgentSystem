# AI Chronic Disease Management System v2

A planner-executor AI agent system with human-in-the-loop oversight and in-memory
session memory, built with LangGraph and LangChain. The system monitors multi-modal
patient data (vitals, glucose, blood pressure, medications, symptoms, labs) across
multiple monitoring passes within a session, produces a structured care plan, filters
all clinical interventions through a human approval gate, and updates trend memory
to inform each subsequent pass.

The system runs fully end-to-end in demo mode with no external dependencies, and
upgrades automatically to live GPT-4o reasoning when an OpenAI API key is present.

---

## Table of Contents

1. [Use Case and Problem Statement](#use-case-and-problem-statement)
2. [System Architecture](#system-architecture)
3. [Architectural Pattern — Planner-Executor with HITL](#architectural-pattern)
4. [In-Memory Session Memory](#in-memory-session-memory)
5. [Human-in-the-Loop Gate](#human-in-the-loop-gate)
6. [Agent Graph Topology](#agent-graph-topology)
7. [Step-by-Step Workflow](#step-by-step-workflow)
8. [File-by-File Explanation](#file-by-file-explanation)
9. [Data Layer and Simulation Design](#data-layer-and-simulation-design)
10. [Patient Profiles](#patient-profiles)
11. [Clinical Thresholds Reference](#clinical-thresholds-reference)
12. [Production Deployment Guide](#production-deployment-guide)
13. [Run Modes and Quick Start](#run-modes-and-quick-start)
14. [Sample Output](#sample-output)
15. [Disclaimer](#disclaimer)

---

## Use Case and Problem Statement

### The Domain

Chronic disease management requires continuous monitoring of patients between clinic
visits. A patient with type 2 diabetes, hypertension, and chronic kidney disease
generates glucose readings every few hours from a continuous glucose monitor, blood
pressure readings once or twice daily from a connected BP cuff, daily weight and
SpO2 from a wearable, weekly lab reports from the health system, and symptom reports
via a mobile app. Their care team — a physician, a care coordinator nurse, and a
specialist — cannot manually review all of this data for hundreds of patients.

### The Problem

Four specific gaps make this an appropriate AI agent problem.

First, the volume of monitoring data per patient exceeds what any care team can review
manually. A single patient generates 30 to 50 data points per day across glucose,
vitals, and medication logs. Across a panel of 200 patients this is 6,000 to 10,000
data points per day that require triage before a clinician can act on them.

Second, the urgency classification requires clinical judgment across multiple domains
simultaneously. A blood glucose of 280 mg/dL is concerning on its own. Combined with
a heart rate of 125 bpm, SpO2 of 89 percent, and a symptom report of shortness of
breath, the same reading becomes an emergency. No single threshold rule handles this
correctly — multi-domain reasoning is required.

Third, safety requires human oversight of clinical decisions. An AI system that
directly sends medication change instructions to a patient, or directly calls 911
based on algorithm output alone, creates serious clinical and legal liability. Human
approval must be part of the execution path for any intervention above the level of
a routine reminder.

Fourth, context accumulates over a monitoring session. If a patient's blood pressure
was 158/92 on the first monitoring pass and is now 162/95, the correct interpretation
is "worsening despite prior monitoring" — a different urgency than a first-time
reading of 162/95 with no prior context. Memory of prior passes within a session
changes the clinical assessment.

### What This System Delivers

A monitoring pass produces: a risk level assessment (low/moderate/high/critical),
a list of clinical alerts and warnings with specific values, a set of actions
executed immediately (patient reminders, care coordinator tasks), a set of actions
queued for physician approval (medication reviews, urgent notifications), and a
structured clinical progress note. Multiple passes demonstrate memory-driven trend
detection and intervention deduplication.

---

## System Architecture

```
+------------------------------------------------------------------+
|  DATA LAYER                                                      |
|                                                                  |
|  get_vitals()        Apple HealthKit / Fitbit API (production)   |
|  get_glucose()       Dexcom Share API / Abbott LibreLink         |
|  get_bp_readings()   Withings Health API / Omron Connect         |
|  get_medications()   Medisafe API / SureScripts                  |
|  get_symptoms()      Patient app / Twilio IVR                    |
|  get_labs()          HL7 FHIR R4 — Epic/Cerner                  |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  IN-MEMORY SESSION MEMORY (memory/session_memory.py)            |
|                                                                  |
|  Per-patient SessionMemory instance:                             |
|    VitalSnapshot per pass (BP trend, glucose trend, adherence)   |
|    IssuedIntervention registry (deduplication by type + urgency) |
|    Pending approval queue (HITL pending items)                   |
|    Consecutive high-BP counter (escalation trigger)              |
|                                                                  |
|  context_summary() -> text block injected into planner prompt    |
|  can_issue_intervention() -> deduplication gate                  |
|  get_trend() -> trend dict read by planner and reporter          |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  LANGGRAPH PLANNER-EXECUTOR GRAPH (graph/care_graph.py)         |
|                                                                  |
|  START                                                           |
|    -> PLANNER AGENT                                              |
|         reads: all 6 data streams + memory context              |
|         computes: clinical signals (threshold logic)             |
|         produces: care plan with hitl_required tags             |
|    -> EXECUTOR AGENT                                             |
|         routes: hitl_required=False -> execute immediately       |
|                 hitl_required=True  -> queue for approval        |
|                 emergency           -> execute bypassing HITL    |
|    -> HUMAN REVIEW NODE (HITL gate)                             |
|         surfaces: pending approval queue to console/dashboard    |
|         auto-approves: routine items (MVP)                       |
|         holds: urgent items for physician sign-off              |
|    -> REPORTER AGENT                                             |
|         builds: clinical progress note                          |
|         updates: session memory for next pass                    |
|    -> END                                                        |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  OUTPUT LAYER                                                    |
|  Risk level (low / moderate / high / critical)                   |
|  Clinical alerts and warnings with specific values               |
|  Executed actions (dispatched via app/SMS/EHR in production)     |
|  Pending approvals (held for physician/nurse sign-off)           |
|  Clinical progress note                                          |
|  Session memory summary (trend context for next pass)           |
+------------------------------------------------------------------+
```

---

## Architectural Pattern

### Planner-Executor with Human-in-the-Loop

The planner-executor pattern separates the reasoning about what to do (planner) from
the execution of those decisions (executor). The human-in-the-loop gate sits between
planning and execution for all non-routine interventions.

**Why Planner-Executor for Clinical Care**

The alternative to this pattern — a single agent that both reasons about patient
status and directly issues interventions — creates a dangerous shortcut: the same
LLM call that assesses risk also triggers actions. A hallucination in the reasoning
step could trigger a wrong clinical action before any human has a chance to review it.

The planner-executor separation ensures that the clinical assessment (planner) is
always a separate artefact from the execution decision (executor). The care plan is a
structured JSON document that can be inspected, logged, and reviewed independently of
its execution. The executor reads the plan and applies the HITL gate before any
clinical action takes place.

**Why Human-in-the-Loop Is Non-Negotiable**

Clinical AI systems that issue medication change instructions or emergency alerts
autonomously create regulatory, legal, and patient safety problems. FDA guidance on
clinical decision support software and HL7 clinical documentation standards both
require that AI-generated recommendations be reviewed by a qualified healthcare
provider before being acted upon.

In this system every action is tagged at plan time with `hitl_required: true` or
`hitl_required: false`. The executor enforces these tags: routine patient education
messages and care coordinator task assignments are executed immediately; physician
notifications, medication reviews, and care plan modifications are always held for
human approval. Emergency alerts bypass HITL only because the legal and ethical
standard in genuine emergencies (e.g., glucose below 54 mg/dL, hypertensive crisis
at SBP 183 mmHg) is that delay is itself a patient safety harm.

**Why In-Memory (Not Persistent Database) for MVP**

For a same-session demonstration the in-memory `SessionMemory` object correctly
captures trend data across multiple monitoring passes and prevents duplicate alert
dispatches within a session. The clinical argument for persistent storage across
sessions (e.g., PostgreSQL with FHIR R4 patient records) is strong — a clinician
reviewing a patient needs to see weeks of trend data, not just the current session.
The code is architected so that `get_memory(patient_id)` can be replaced with a
database-backed `SessionMemory` subclass without any changes to the planner, executor,
or reporter agents.

---

## In-Memory Session Memory

The `SessionMemory` class (`memory/session_memory.py`) is per-patient, keyed by
patient ID. A shared dictionary `_MEMORIES` holds one instance per patient.

### What Is Stored

**VitalSnapshot per pass**: After each reporter call, a snapshot is appended containing
the maximum glucose, average systolic BP, minimum SpO2, average medication adherence,
active symptoms, and risk level from that pass. The snapshot list is trimmed to the
last 10 entries.

**Issued intervention registry**: Every intervention dispatched by the executor is
recorded as an `IssuedIntervention` with its type, urgency, patient ID, and timestamp.
This enables the deduplication gate.

**Pending approval queue**: All actions tagged `hitl_required=True` that have not yet
been reviewed are held in `pending_approvals`. The `approve_pending_actions()` function
drains this queue.

### Deduplication Logic

`can_issue_intervention(type, urgency, patient_id)` prevents the executor from
dispatching the same intervention twice at the same or lower urgency level within a
session. The same intervention type at a strictly higher urgency is always allowed
(escalation). For example, if a `physician_notification` at `routine` urgency was
issued in pass 1, pass 2 will suppress another `routine` physician notification for
the same patient but will allow an `urgent` one.

### Trend Computation

`get_trend()` computes the BP trend (worsening if average systolic increased more than
5 mmHg since last pass, improving if it decreased more than 5 mmHg, stable otherwise),
glucose trend (same logic with a 20 mg/dL threshold), and the consecutive high BP
counter (number of consecutive passes where average systolic exceeded 140 mmHg). These
values are injected into the planner's LLM context via `context_summary()`.

---

## Human-in-the-Loop Gate

The HITL gate is implemented as a dedicated LangGraph node (`human_review_node`) that
runs between the executor and the reporter.

### How It Works

The executor processes each planned action and applies one of three dispositions:

**Execute immediately** (`hitl_required=False`): Patient education messages, routine
check-in SMS, care coordinator task assignments. These are dispatched in the executor
node. No human review required.

**Queue for approval** (`hitl_required=True`): Physician notifications, care plan
modifications, medication reviews. These are placed in `memory.pending_approvals`.
The executor does not execute them.

**Emergency bypass**: Actions with `urgency=emergency` skip the HITL gate entirely
because delay constitutes a patient safety harm. Emergency alerts are dispatched
immediately by the executor regardless of `hitl_required`.

The `human_review_node` receives the queued actions, prints them to the console
(in production: surfaces them to a clinician dashboard), and calls
`approve_pending_actions()`. In the MVP, routine-urgency queued items are auto-approved.
Urgent items are printed and auto-approved for demo purposes only — in production
they wait for physician sign-off via the EHR portal.

### Conditional Edge Around HITL

When `emergency_triggered=True`, the LangGraph conditional edge routes directly from
executor to reporter, bypassing the human review node. This ensures emergency
situations are not delayed by the HITL queue.

---

## Agent Graph Topology

```
START
  |
  v
planner_agent
  | Reads: all 6 data streams + memory.context_summary()
  | Computes: clinical signals (deterministic threshold logic)
  | Produces: care_plan — list of action dicts with hitl_required tags
  | Writes: state["care_plan"], state["signals"], state["risk_level"]
  |
  v
executor_agent
  | Routes each action:
  |   urgency=emergency -> execute immediately (bypass HITL)
  |   hitl_required=False -> execute immediately
  |   hitl_required=True  -> queue in memory.pending_approvals
  | Writes: state["executed_actions"], state["queued_for_approval"]
  |
  v
should_skip_hitl()
  | emergency_triggered=True  -> reporter (skip HITL)
  | emergency_triggered=False -> human_review
  |
  +-- human_review_node (HITL gate)
  |     Surfaces queued actions
  |     approve_pending_actions() -> drains queue
  |     Writes: state["executed_actions"] += newly approved
  |
  v
reporter_agent
  | Builds: clinical progress note
  | Updates: memory.record_pass(state) — appends VitalSnapshot
  | Writes: state["progress_summary"]
  |
  v
END
```

---

## Step-by-Step Workflow

### Step 1: Planner — Data Collection and Signal Computation

The planner reads the session memory context summary first, printing the trend line
to the console so the monitoring pass begins with full awareness of prior results.

It then loads all six data streams for the patient: 7 days of wearable vitals, 14
glucose readings (last 7 days twice daily), 7 BP readings, medication adherence
records for all active medications, recent symptom reports, and the latest lab results.

The `_compute_signals()` function applies all clinical threshold checks deterministically
without an LLM call. This is deliberate — threshold logic should not be probabilistic.
The checks cover: critical hypoglycaemia (glucose below 54 mg/dL triggers emergency),
very high glucose (above 350 mg/dL alerts), hypertensive crisis (systolic above 180 mmHg
triggers emergency), consistently elevated BP (average systolic above 140 mmHg raises
an alert), low SpO2 (below 92 percent alerts, below 88 percent triggers emergency),
abnormal heart rate (below 45 or above 130 bpm triggers emergency), low medication
adherence (below 60 percent for any medication raises a warning), high HbA1c (above
10 percent raises alert), and severely reduced eGFR (below 30 mL/min raises alert).

If any emergency condition is detected, the planner creates a single emergency action
and routes directly to the executor, bypassing the LLM care plan generation.

For non-emergency passes, the LLM call receives the signal dict, lab results, and
trend context from memory. The LLM produces a JSON list of action objects, each
with type, urgency, recipient, instructions, and `hitl_required`. The demo path
builds this list directly from the signals without an LLM call.

### Step 2: Executor — HITL Routing

The executor iterates through each planned action in priority order and applies three
checks before dispatch.

Emergency check: if urgency is emergency, dispatch immediately regardless of
`hitl_required` or memory deduplication state.

Memory deduplication check: `memory.can_issue_intervention(type, urgency, patient_id)`
is called. If the same intervention type was already issued at the same or higher
urgency in this session, the action is skipped and logged as a duplicate.

HITL gate: if `hitl_required=True`, the action is added to `memory.pending_approvals`
and the executor moves on. If `hitl_required=False`, the action is dispatched.

### Step 3: Human Review Node

The node calls `approve_pending_actions()` which prints each pending action to the
console with its urgency level and title. Routine items are auto-approved. Approved
items are dispatched and added to `executed_actions`. Items held for physician
sign-off remain in `pending_approvals` (visible in the session memory summary).

### Step 4: Reporter — Summary and Memory Update

The reporter builds the clinical progress note covering: risk level, clinical alerts
and warnings with specific values, actions taken in this pass, items pending human
approval, and the trend context from session memory.

It then calls `memory.record_pass(state)`, which appends a `VitalSnapshot` to the
rolling history. This snapshot is what the planner reads on the next pass via
`get_trend()` to detect worsening or improving trends.

---

## File-by-File Explanation

### main.py

Entry point. Accepts `patient_id` (P001 or P002) and `--passes N` arguments.
`run_pass()` builds and invokes the LangGraph graph or falls back to direct agent
calls. `print_report()` formats: risk level, emergency flag, clinical alerts and
warnings, action breakdown (executed/approved/queued/skipped), session memory
summary, and the clinical progress note.

Multiple passes demonstrate memory deduplication (same alert suppressed in pass 2),
trend detection (worsening/improving labels appear in pass 2+), and HITL queue
management across passes.

### config/settings.py

`CareConfig` dataclass. Clinical thresholds for glucose, blood pressure, heart rate,
SpO2, and medication adherence. `hitl_approval_threshold` configures the minimum
urgency level that requires human approval (default: urgent). OpenAI API key from
environment.

### data/simulation.py

Six simulation functions, one per data stream. All use the date-seeded `_rng(
patient_id, salt)` pattern, where the patient ID is included in the seed to ensure
different patients get different readings, and the salt distinguishes data streams.
Each function returns data structured to match its production API counterpart:
vitals as a list of 7 daily dicts, glucose as a list of 14 twice-daily readings,
and so on. Two patient profiles (P001: diabetes/hypertension/CKD, P002: heart
failure/hypertension/diabetes) are defined in `PATIENTS`.

### memory/session_memory.py

`SessionMemory` class with: `record_pass()`, `record_intervention()`,
`add_pending_approval()`, `can_issue_intervention()`, `get_trend()`,
`context_summary()`, and `reset()`. `get_memory(patient_id)` returns the per-patient
singleton from the `_MEMORIES` dict, creating it if it does not exist.

### agents/base.py

`_demo_mode()` and `call_llm()` shared helpers.

### agents/planner_agent.py

`run_planner(state)`: loads all six data streams, reads memory context, calls
`_compute_signals()` for deterministic threshold checks, handles emergency fast-path,
then calls LLM with signal dict and trend context to generate the structured care plan.
`_build_demo_plan(signals, trend, patient)` builds the plan in demo mode.

### agents/executor_agent.py

`run_executor(state)`: iterates care plan, applies emergency bypass, memory
deduplication, and HITL gate to each action. `approve_pending_actions(patient_id,
approved_ids, approve_all_routine)` is the public HITL approval function called by
the human review node. `_dispatch_action()` simulates dispatch and returns a dispatch
record. `_get_channel()` maps recipient type to dispatch channel.

### agents/reporter_agent.py

`run_reporter(state)`: calls `memory.record_pass(state)` to update trend history,
reads trend data, calls LLM (or demo) to build the progress note.
`_build_demo_summary()` constructs the full structured clinical note from state data.

### graph/care_graph.py

`CareState` TypedDict. `human_review_node()` wraps `approve_pending_actions()`.
`_should_skip_hitl()` routes emergency cases directly to reporter. `build_care_graph()`
wires the four nodes with the conditional edge from executor to either human_review
or reporter.

---

## Data Layer and Simulation Design

Patient P001 (Margaret Chen, age 67, diabetes/hypertension/CKD) is designed with
a high-risk profile. Her glucose readings span a wide range (68 to 320 mg/dL) to
ensure both hypoglycaemia and hyperglycaemia scenarios appear. Her BP readings
span 115 to 185 mmHg to trigger both normal and hypertensive crisis scenarios
depending on the day's RNG seed. Her medications include Lisinopril and Amlodipine,
which are critical for her hypertension and CKD management — these are the medications
most likely to show low adherence in the simulation.

Patient P002 (Thomas Rivera, age 58, heart failure/hypertension/diabetes) has
heart-failure-specific fields in his labs (BNP and sodium) that trigger additional
clinical assessments appropriate to heart failure management.

Both patients use day-seeded RNG, ensuring the same patient produces identical
readings within a calendar day. The `--passes N` argument demonstrates memory
accumulation within a single Python process using the same day's seed — a deliberate
design choice to show the deduplication and trend features without requiring multiple
distinct data states.

---

## Patient Profiles

| Field | P001 — Margaret Chen | P002 — Thomas Rivera |
|---|---|---|
| Age | 67 | 58 |
| Conditions | Type 2 diabetes, hypertension, CKD | Heart failure, hypertension, type 2 diabetes |
| Active medications | Metformin, Lisinopril, Amlodipine, Atorvastatin, Aspirin | Carvedilol, Furosemide, Lisinopril, Spironolactone |
| Physician | Dr. Sarah Williams, MD | Dr. Maria Santos, MD |
| Risk level | High | High |
| Special monitoring | eGFR, HbA1c, potassium | BNP, sodium, daily weight |

---

## Clinical Thresholds Reference

All thresholds are derived from ADA Standards of Diabetes Care 2024, ACC/AHA
Hypertension Guidelines 2023, and GOLD COPD guidelines.

| Parameter | Warning Threshold | Alert Threshold | Emergency Threshold |
|---|---|---|---|
| Glucose | above 180 mg/dL | above 250 mg/dL | below 54 or above 350 mg/dL |
| BP systolic | — | above 140 mmHg (avg) | above 180 mmHg (any reading) |
| SpO2 | — | below 92% | below 88% |
| Heart rate | — | above 100 bpm | below 45 or above 130 bpm |
| Medication adherence | below 60% | — | — |
| HbA1c | — | above 10% | — |
| eGFR | — | below 30 mL/min | — |

---

## Production Deployment Guide

### Replacing Glucose Data

Replace `get_glucose()` with Dexcom Share API (for Dexcom CGM users) or Abbott
LibreView API (for FreeStyle Libre users). Both provide 5-minute interval glucose
readings via REST API. Authentication uses OAuth2 with patient consent. Readings
include glucose value, trend arrow (falling fast, falling, stable, rising, rising fast),
and timestamp.

### Replacing Vitals Data

Replace `get_vitals()` with Apple HealthKit integration (iOS HealthKit framework via
a companion app) or Fitbit Web API (heart rate, SpO2, weight, sleep stages). Both
require patient OAuth2 consent via the companion app. Google Health Connect provides
a unified Android API covering most wearable manufacturers.

### Replacing Medication Data

Replace `get_medications()` with Medisafe API (smart pill dispenser + app, tracks
individual pill dispenses with timestamps) or Epic MyChart API (HL7 FHIR R4
`MedicationRequest` and `MedicationStatement` resources for prescription fills and
self-reported doses).

### Replacing Labs

Replace `get_labs()` with an HL7 FHIR R4 `Observation` resource query against the
patient's EHR system (Epic FHIR API or Cerner FHIR API). SMART on FHIR authentication
handles patient consent. Query by observation code (LOINC) for HbA1c (`4548-4`),
eGFR (`62238-1`), potassium (`2823-3`), BNP (`42637-9`).

### Replacing the Memory Layer with PostgreSQL

To upgrade from in-memory to persistent storage, create a `PersistentSessionMemory`
subclass that overrides `record_pass()`, `record_intervention()`, and
`can_issue_intervention()` to read/write from a PostgreSQL database. The `get_memory(
patient_id)` factory function can return this subclass when a database connection is
configured. All planner, executor, and reporter code that calls `memory.context_summary(
)`, `memory.can_issue_intervention()`, and `memory.record_pass()` continues to work
without changes.

---

## Run Modes and Quick Start

### Demo Mode — Single Pass

```bash
cd chronic_v2
python main.py P001
python main.py P002
```

### Demo Mode — Multiple Passes (demonstrates memory)

```bash
python main.py P001 --passes 3
```

Pass 1 establishes baseline. Pass 2 shows memory trend context in planner output and
deduplication of any previously issued interventions. Pass 3 shows either escalation
(if trend is worsening) or suppression (if trend is stable/improving).

### Live LLM Mode

```bash
pip install langgraph langchain-core langchain-openai python-dotenv
cp .env.example .env
# Add OPENAI_API_KEY=sk-your-key to .env
python main.py P001 --passes 2
```

---

## Sample Output

```
AI CHRONIC DISEASE MANAGEMENT SYSTEM v2
Architecture: Planner-Executor + HITL + In-Memory
Patient: Margaret Chen | ID: P001
Conditions: type2_diabetes, hypertension, chronic_kidney_disease
Mode: DEMO (no API key)

[PLANNER] Building care plan for Margaret Chen (pass #1)...
  Memory context: SESSION MEMORY: Pass #1 — no prior data. Establishing baseline.
  [PLANNER] EMERGENCY DETECTED: Hypertensive crisis: SBP 183 mmHg

[EXECUTOR] Processing 1 planned action(s)...
  [EMERGENCY] EXECUTED: EMERGENCY: Hypertensive crisis: SBP 183 mmHg

RISK LEVEL: CRITICAL
EMERGENCY:  True

CLINICAL ALERTS (1):
  Low SpO2: 91.6%

WARNINGS (3):
  Elevated glucose: 319 mg/dL (target <180)
  Low adherence: Lisinopril at 57.1%
  Low adherence: Amlodipine at 42.9%

ACTIONS:
  Executed       : 1
  Approved (HITL): 1
  Queued pending : 0
  Skipped (dedup): 0

    [EMERGENCY] emergency_alert -> emergency_services_and_care_team
                via emergency_alert_system

SESSION MEMORY:
  SESSION MEMORY: Pass #1 — no prior data. Establishing baseline.

--- Pass 2 ---

[PLANNER] Building care plan for Margaret Chen (pass #2)...
  Memory: BP trend=STABLE | Glucose trend=STABLE | Adherence=63.4%
  [EXECUTOR] SKIP: emergency_alert (same urgency issued in pass 1)
  [HITL REVIEW] 2 pending action(s):
    [URGENT]  physician_notification: Clinical Alert — Low SpO2...
    [ROUTINE] adherence_support: Lisinopril — call patient...
  [APPROVED + DISPATCHED] ACT-001
  [APPROVED + DISPATCHED] ACT-002

SESSION MEMORY (pass #2):
  BP trend      : STABLE
  Glucose trend : STABLE
  Adherence avg : 63.4%
  Interventions : 3 issued this session
```

---

## Disclaimer

CLINICAL DECISION SUPPORT ONLY. All recommendations produced by this system require
review by a licensed healthcare provider before implementation. In an emergency,
call 911 immediately. This system does not replace clinical judgment and is not
approved as a medical device. Simulation data does not represent actual patient
conditions.
