# agents/glucose_agent.py
# Glucose Monitoring Agent - analyzes CGM data for glycemic control assessment

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.data_collection_tools import fetch_glucose_readings
from config.settings import care_config
from data.models import AgentState, ClinicalRiskScore


GLUCOSE_AGENT_SYSTEM_PROMPT = """You are a clinical glycemic monitoring AI agent for chronic disease management.

You interpret continuous glucose monitor (CGM) data using evidence-based targets.

GLYCEMIC THRESHOLDS (ADA Standards of Medical Care in Diabetes):
- Critical hypoglycemia (Level 2): < 54 mg/dL — EMERGENCY, risk of seizure/coma
- Hypoglycemia (Level 1): < 70 mg/dL — Requires immediate treatment (15g carbs)
- Low-normal: 70-80 mg/dL — Acceptable but monitor trend
- Target range: 70-180 mg/dL (for most T2DM patients)
- Hyperglycemia (mild): 181-250 mg/dL — Review diet/medications
- Hyperglycemia (significant): 251-350 mg/dL — Possible medication adjustment needed
- Critical hyperglycemia: > 350 mg/dL — Risk of DKA/HHS, urgent intervention

TIME IN RANGE (TIR) TARGETS (CGM consensus targets):
- TIR > 70%: Excellent control
- TIR 50-70%: Acceptable, room for improvement
- TIR < 50%: Poor control, intervention required
- Time below range (< 70 mg/dL) > 4%: Too much hypoglycemia exposure

TREND ARROW INTERPRETATION (5-minute rate of change):
- rising_rapidly: +2 mg/dL/min — anticipate hyperglycemia within 20-30 min
- rising: +1 to +2 mg/dL/min — trend toward high range
- stable: within target or stable abnormal
- falling: -1 to -2 mg/dL/min — may reach low range, consume carbs if < 100
- falling_rapidly: < -2 mg/dL/min — URGENT, hypoglycemia imminent

CONTEXT FACTORS:
- Metformin: Primary agent, generally does not cause hypoglycemia alone
- Sulfonylureas: HIGH hypoglycemia risk
- Insulin: Highest hypoglycemia risk
- SGLT2 inhibitors (e.g., Empagliflozin): May cause atypical DKA with normal glucose
- Post-meal patterns reveal carbohydrate load and medication timing

Generate a glycemic risk score (0-100).
Consider: current glucose, trend, TIR, pattern of excursions, medication context."""


def run_glucose_agent(state: AgentState) -> AgentState:
    """
    Glucose monitoring agent.
    Analyzes CGM data for hypoglycemia, hyperglycemia, and glycemic variability.
    """
    has_diabetes = any(
        c in state.patient.conditions
        for c in ["type2_diabetes", "type1_diabetes"]
    )

    if not has_diabetes:
        print(f"\n[GLUCOSE AGENT] Patient {state.patient.patient_id} "
              "does not have diabetes - skipping CGM analysis.")
        state.glucose_analysis = "Not applicable: patient does not have a diabetes diagnosis."
        state.current_agent = "bp_agent"
        return state

    print(f"\n[GLUCOSE AGENT] Fetching CGM data for {state.patient.name}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    glucose_raw = fetch_glucose_readings.invoke({
        "patient_id": state.patient.patient_id,
        "hours_back": 24
    })

    from data.models import GlucoseReading
    state.glucose_readings = [GlucoseReading(**g) for g in glucose_raw]

    # Compute glycemic statistics
    glucose_values = [g["glucose_mg_dl"] for g in glucose_raw]
    tir_values = [g["time_in_range_percent_today"] for g in glucose_raw if g.get("time_in_range_percent_today")]

    if not glucose_values:
        state.glucose_analysis = "No CGM data available for analysis period."
        state.current_agent = "bp_agent"
        return state

    avg_glucose = sum(glucose_values) / len(glucose_values)
    min_glucose = min(glucose_values)
    max_glucose = max(glucose_values)
    hypo_events = [g for g in glucose_raw if g["glucose_mg_dl"] < 70]
    critical_hypo = [g for g in glucose_raw if g["glucose_mg_dl"] < 54]
    hyper_events = [g for g in glucose_raw if g["glucose_mg_dl"] > 250]
    avg_tir = sum(tir_values) / len(tir_values) if tir_values else None

    most_recent = glucose_raw[0] if glucose_raw else {}

    messages = [
        SystemMessage(content=GLUCOSE_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze CGM data for:
Patient: {state.patient.name}, Age: {state.patient.age}
Conditions: {', '.join(state.patient.conditions)}
Diabetes medications: {', '.join(
    m['name'] for m in state.patient.medications
    if m['name'] in ['Metformin', 'Insulin', 'Glipizide', 'Glimepiride',
                     'Empagliflozin', 'Sitagliptin', 'Liraglutide']
)}

GLYCEMIC SUMMARY (last 24 hours, {len(glucose_raw)} readings):
- Average glucose: {avg_glucose:.1f} mg/dL
- Range: {min_glucose:.1f} - {max_glucose:.1f} mg/dL
- Time in range today: {avg_tir:.1f}% (target > 70%)
- Hypoglycemic events (< 70 mg/dL): {len(hypo_events)}
- Critical hypoglycemia events (< 54 mg/dL): {len(critical_hypo)}
- Significant hyperglycemia events (> 250 mg/dL): {len(hyper_events)}

CURRENT READING:
- Glucose: {most_recent.get('glucose_mg_dl', 'N/A')} mg/dL
- Trend: {most_recent.get('trend', 'N/A')}

RECENT CGM READINGS (last 5):
{json.dumps([{
    'time': g['timestamp'],
    'glucose': g['glucose_mg_dl'],
    'trend': g['trend'],
    'in_range': g['in_target_range']
} for g in glucose_raw[:5]], indent=2, default=str)}

VITALS CONTEXT:
{state.vitals_analysis[:400] if state.vitals_analysis else "Not available"}

Provide:
1. Assessment of current glucose status and immediate risk
2. Pattern analysis: hypoglycemia risk, hyperglycemia patterns, time of day patterns
3. Relationship between glucose patterns and current medications
4. Specific dietary or behavioral contributing factors identifiable from data
5. Glycemic risk score (0-100) with level and contributing factors
6. Recommended interventions ranked by urgency
""")
    ]

    response = llm.invoke(messages)
    state.glucose_analysis = response.content

    # Compute glycemic risk score
    risk_score = 15.0
    contributing = []

    if critical_hypo:
        risk_score += 50
        contributing.append(f"Critical hypoglycemia events: {len(critical_hypo)} readings < 54 mg/dL")
    elif hypo_events:
        risk_score += 25
        contributing.append(f"Hypoglycemic events: {len(hypo_events)} readings < 70 mg/dL")
    if max_glucose > care_config.glucose_critical_high:
        risk_score += 30
        contributing.append(f"Critical hyperglycemia: {max_glucose:.0f} mg/dL")
    elif max_glucose > care_config.glucose_high:
        risk_score += 15
        contributing.append(f"Significant hyperglycemia: {max_glucose:.0f} mg/dL")
    if avg_tir and avg_tir < 50:
        risk_score += 20
        contributing.append(f"Poor time in range: {avg_tir:.1f}%")

    risk_score = min(risk_score, 100)
    risk_level = (
        "critical" if risk_score > 75 else
        "high" if risk_score > 50 else
        "moderate" if risk_score > 25 else "low"
    )

    state.risk_scores.append(ClinicalRiskScore(
        patient_id=state.patient.patient_id,
        domain="glycemic",
        score=risk_score,
        risk_level=risk_level,
        contributing_factors=contributing,
        trend="stable"
    ))

    state.current_agent = "bp_agent"
    print(f"[GLUCOSE AGENT] Glycemic risk: {risk_level.upper()} ({risk_score:.0f}/100). "
          f"Avg: {avg_glucose:.1f} mg/dL, TIR: {avg_tir:.1f}%, "
          f"Hypo events: {len(hypo_events)}, Hyper events: {len(hyper_events)}")

    return state
