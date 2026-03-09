# agents/medication_agent.py
# Medication Adherence Agent - analyzes dose history and identifies adherence barriers

import json
from collections import defaultdict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.data_collection_tools import fetch_medication_adherence
from config.settings import care_config
from data.models import AgentState, ClinicalRiskScore


MEDICATION_AGENT_SYSTEM_PROMPT = """You are a clinical medication adherence monitoring AI agent.

You analyze medication adherence data to identify patterns, barriers, and clinical consequences.

ADHERENCE DEFINITIONS AND CLINICAL SIGNIFICANCE:
- Optimal adherence: >= 80% of doses taken on time
- Suboptimal adherence: 60-79% — associated with poorer clinical outcomes
- Poor adherence: 40-59% — significant clinical risk, urgent intervention needed
- Critical non-adherence: < 40% — high risk of disease progression, hospitalization
- On-time dose: within +/- 90 minutes of scheduled time

MEDICATION-SPECIFIC RISK OF NON-ADHERENCE:
HIGH RISK (missing doses causes rapid decompensation):
- Heart failure medications (beta-blockers, ACE inhibitors, diuretics): Acute decompensation within days
- Antihypertensives: Rebound hypertension, stroke risk
- Oral hypoglycemics/insulin: Glycemic crisis

MODERATE RISK (missing doses causes gradual deterioration):
- Statins: Increased cardiovascular event risk over time
- Aspirin: Increased thrombotic event risk

ADHERENCE BARRIER PATTERNS TO IDENTIFY:
- Consistent morning dose missed: Side effects (nausea, dizziness), schedule mismatch
- Consistent evening dose missed: Fatigue, forgetting with dinner routine
- Weekend miss pattern: Social disruption, travel
- Random misses: Likely forgetfulness, needs reminder system
- Sudden drop in adherence: New side effect, cost issues, depression, health beliefs change

PILL BURDEN ASSESSMENT:
- 1-3 medications/day: Low burden
- 4-6 medications/day: Moderate burden (simplification candidates)
- 7+ medications/day: High burden, polypharmacy review needed

Critically evaluate patterns over 14 days, not just total adherence rate.
Generate a medication adherence risk score (0-100)."""


def run_medication_agent(state: AgentState) -> AgentState:
    """
    Medication adherence monitoring agent.
    Analyzes 14-day adherence records to identify patterns and intervention needs.
    """
    print(f"\n[MEDICATION AGENT] Analyzing medication adherence for {state.patient.name}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    adherence_raw = fetch_medication_adherence.invoke({
        "patient_id": state.patient.patient_id,
        "days_back": 14
    })

    from data.models import MedicationAdherenceRecord
    state.medication_records = [MedicationAdherenceRecord(**m) for m in adherence_raw]

    # Compute per-medication adherence statistics
    med_stats = defaultdict(lambda: {"total": 0, "taken": 0, "late": 0, "missed": []})

    for record in adherence_raw:
        med = record["medication_name"]
        med_stats[med]["total"] += 1
        if record["taken"]:
            med_stats[med]["taken"] += 1
            # Check if late (> 90 min)
            if record.get("taken_time") and record.get("scheduled_dose_time"):
                from datetime import datetime
                try:
                    scheduled = datetime.fromisoformat(str(record["scheduled_dose_time"]))
                    taken = datetime.fromisoformat(str(record["taken_time"]))
                    delay = abs((taken - scheduled).total_seconds() / 60)
                    if delay > 90:
                        med_stats[med]["late"] += 1
                except Exception:
                    pass
        else:
            med_stats[med]["missed"].append(record["scheduled_dose_time"])

    adherence_summary = []
    for med, stats in med_stats.items():
        rate = (stats["taken"] / stats["total"] * 100) if stats["total"] > 0 else 0
        status = (
            "CRITICAL" if rate < care_config.adherence_critical_percent else
            "POOR" if rate < care_config.adherence_low_percent else
            "SUBOPTIMAL" if rate < care_config.adherence_target_percent else
            "GOOD"
        )
        adherence_summary.append({
            "medication": med,
            "adherence_percent": round(rate, 1),
            "doses_taken": stats["taken"],
            "doses_missed": stats["total"] - stats["taken"],
            "late_doses": stats["late"],
            "status": status
        })

    overall_adherence = (
        sum(s["adherence_percent"] for s in adherence_summary) / len(adherence_summary)
        if adherence_summary else 0
    )

    messages = [
        SystemMessage(content=MEDICATION_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze 14-day medication adherence for:
Patient: {state.patient.name}, Age: {state.patient.age}
Conditions: {', '.join(state.patient.conditions)}
Total medications: {len(state.patient.medications)} (pill burden assessment needed)

PER-MEDICATION ADHERENCE (14 days):
{json.dumps(adherence_summary, indent=2)}

OVERALL ADHERENCE RATE: {overall_adherence:.1f}%
Adherence target: >= {care_config.adherence_target_percent}%

FULL MEDICATION LIST WITH DOSES:
{json.dumps(state.patient.medications, indent=2)}

CLINICAL CONTEXT FROM OTHER AGENTS:
Glucose status: {state.glucose_analysis[:250] if state.glucose_analysis else "Not available"}
BP status: {state.bp_analysis[:250] if state.bp_analysis else "Not available"}

Provide:
1. Per-medication adherence assessment with clinical risk of non-adherence
2. Pattern analysis: time-of-day, day-of-week, weekend vs weekday patterns
3. Likely barriers to adherence for poorly adherent medications
4. Clinical consequences visible in the vitals/glucose/BP data from missed doses
5. Pill burden assessment and simplification opportunities
6. Specific interventions: reminders, blister packs, medication synchronization, etc.
7. Medications where non-adherence poses immediate safety risk
8. Medication adherence risk score (0-100) with risk level
""")
    ]

    response = llm.invoke(messages)
    state.medication_analysis = response.content

    # Compute adherence risk score
    risk_score = max(0, 100 - overall_adherence)
    contributing = []
    critical_meds = [s for s in adherence_summary if s["status"] == "CRITICAL"]
    poor_meds = [s for s in adherence_summary if s["status"] == "POOR"]

    if critical_meds:
        contributing.append(
            f"Critical non-adherence: {', '.join(m['medication'] for m in critical_meds)}"
        )
        risk_score = min(risk_score + 20, 100)
    if poor_meds:
        contributing.append(
            f"Poor adherence: {', '.join(m['medication'] for m in poor_meds)}"
        )
    if overall_adherence < care_config.adherence_target_percent:
        contributing.append(f"Overall adherence below target: {overall_adherence:.1f}%")

    risk_level = (
        "critical" if risk_score > 75 else
        "high" if risk_score > 50 else
        "moderate" if risk_score > 25 else "low"
    )

    state.risk_scores.append(ClinicalRiskScore(
        patient_id=state.patient.patient_id,
        domain="medication",
        score=risk_score,
        risk_level=risk_level,
        contributing_factors=contributing,
        trend="stable"
    ))

    state.current_agent = "symptom_agent"
    print(f"[MEDICATION AGENT] Adherence risk: {risk_level.upper()} ({risk_score:.0f}/100). "
          f"Overall adherence: {overall_adherence:.1f}%. "
          f"Critical meds: {len(critical_meds)}, Poor adherence meds: {len(poor_meds)}")

    return state
