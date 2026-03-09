# agents/vitals_agent.py
# Wearable Vitals Monitoring Agent - analyzes HR, SpO2, activity, and sleep

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.data_collection_tools import fetch_wearable_vitals
from config.settings import care_config
from data.models import AgentState, ClinicalRiskScore


VITALS_AGENT_SYSTEM_PROMPT = """You are a clinical vitals monitoring AI agent for chronic disease management.

You analyze continuous wearable device data for patients with chronic conditions.

CLINICAL THRESHOLDS TO EVALUATE:
- Heart rate critical low: < {hr_critical_low} bpm (bradycardia, risk of syncope)
- Heart rate critical high: > {hr_critical_high} bpm (tachycardia - could indicate decompensation)
- SpO2 critical: < {spo2_critical}% (hypoxemia - URGENT in COPD/heart failure)
- SpO2 concerning: < {spo2_low}% (monitor closely, may need supplemental O2)
- Activity: < 2000 steps consistently suggests functional decline
- Sleep < 5 hours consistently: affects glycemic control, BP, immune function

CONDITION-SPECIFIC INTERPRETATION:
- Heart failure patients: Weight gain proxy via activity decline, SpO2 trends,
  resting HR increase (sympathetic activation) may signal decompensation
- COPD patients: SpO2 is paramount - drops indicate exacerbation onset
- Diabetes patients: Activity directly affects insulin sensitivity and glucose
- Hypertension: Elevated resting HR + stress score may drive BP up
- CKD patients: Fatigue and activity decline may signal uremic progression

IMPORTANT CLINICAL SAFETY RULES:
1. Never interpret isolated readings - always consider trends over 24-48 hours
2. Night-time HR drops are expected - do not flag as bradycardia
3. Post-exercise HR elevation is expected - context matters
4. HRV decrease combined with elevated resting HR is a stronger signal than either alone
5. Your analysis supports clinical decision-making; it does not replace physician judgment

Generate a ClinicalRiskScore for cardiovascular domain (0-100).
Risk level mapping: 0-25 = low, 26-50 = moderate, 51-75 = high, 76-100 = critical"""


def run_vitals_agent(state: AgentState) -> AgentState:
    """
    Wearable vitals monitoring agent.
    Fetches and analyzes heart rate, SpO2, activity, and sleep data.
    """
    print(f"\n[VITALS AGENT] Fetching wearable data for patient {state.patient.patient_id}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    vitals_raw = fetch_wearable_vitals.invoke({
        "patient_id": state.patient.patient_id,
        "hours_back": 24
    })

    from data.models import WearableVitalsReading
    state.vitals_readings = [WearableVitalsReading(**v) for v in vitals_raw]

    # Compute summary statistics for LLM
    hrs = [v["heart_rate_bpm"] for v in vitals_raw if v["heart_rate_bpm"]]
    spo2s = [v["spo2_percent"] for v in vitals_raw if v["spo2_percent"]]
    hrv_values = [v["heart_rate_variability_ms"] for v in vitals_raw if v.get("heart_rate_variability_ms")]

    avg_hr = sum(hrs) / len(hrs) if hrs else None
    min_hr = min(hrs) if hrs else None
    max_hr = max(hrs) if hrs else None
    avg_spo2 = sum(spo2s) / len(spo2s) if spo2s else None
    min_spo2 = min(spo2s) if spo2s else None
    avg_hrv = sum(hrv_values) / len(hrv_values) if hrv_values else None

    most_recent = vitals_raw[0] if vitals_raw else {}

    messages = [
        SystemMessage(content=VITALS_AGENT_SYSTEM_PROMPT.format(
            hr_critical_low=care_config.hr_critical_low,
            hr_critical_high=care_config.hr_critical_high,
            spo2_critical=care_config.spo2_critical_low,
            spo2_low=care_config.spo2_low
        )),
        HumanMessage(content=f"""
Analyze wearable vitals for:
Patient: {state.patient.name}, Age: {state.patient.age}
Conditions: {', '.join(state.patient.conditions)}
Medications: {', '.join(m['name'] for m in state.patient.medications)}

VITAL STATISTICS (last 24 hours):
- Heart rate: avg={avg_hr:.1f}, min={min_hr:.1f}, max={max_hr:.1f} bpm
- SpO2: avg={avg_spo2:.1f}%, min={min_spo2:.1f}%
- Heart rate variability: avg={avg_hrv:.1f} ms (lower = more physiologic stress)
- Steps today: {most_recent.get('steps_today', 'N/A')}
- Active minutes: {most_recent.get('active_minutes_today', 'N/A')}
- Sleep last night: {most_recent.get('sleep_hours_last_night', 'N/A')} hours
- Sleep quality score: {most_recent.get('sleep_quality_score', 'N/A')}/100
- Stress score: {most_recent.get('stress_score', 'N/A')}/100

RECENT READINGS (last 3):
{json.dumps(vitals_raw[:3], indent=2, default=str)}

Provide:
1. Clinical assessment of each vital parameter
2. Trend analysis (concerning vs reassuring patterns)
3. Condition-specific interpretation for this patient's diagnoses
4. Any urgent findings requiring immediate escalation
5. A cardiovascular risk score (0-100) with risk level and trend
6. Specific monitoring priorities for next 24 hours
""")
    ]

    response = llm.invoke(messages)
    state.vitals_analysis = response.content

    # Compute a simple risk score based on thresholds
    risk_score = 25.0
    contributing = []

    if avg_hr and avg_hr > care_config.hr_high:
        risk_score += 20
        contributing.append(f"Elevated average HR: {avg_hr:.1f} bpm")
    if min_spo2 and min_spo2 < care_config.spo2_low:
        risk_score += 25
        contributing.append(f"Low SpO2 nadir: {min_spo2:.1f}%")
    if most_recent.get("steps_today", 5000) < 2000:
        risk_score += 10
        contributing.append("Severely reduced daily activity")
    if most_recent.get("sleep_hours_last_night") and most_recent["sleep_hours_last_night"] < 5:
        risk_score += 10
        contributing.append("Insufficient sleep")

    risk_score = min(risk_score, 100)
    risk_level = (
        "critical" if risk_score > 75 else
        "high" if risk_score > 50 else
        "moderate" if risk_score > 25 else "low"
    )

    state.risk_scores.append(ClinicalRiskScore(
        patient_id=state.patient.patient_id,
        domain="cardiovascular",
        score=risk_score,
        risk_level=risk_level,
        contributing_factors=contributing,
        trend="stable"
    ))

    state.current_agent = "glucose_agent"
    print(f"[VITALS AGENT] Cardiovascular risk: {risk_level.upper()} ({risk_score:.0f}/100). "
          f"Avg HR: {avg_hr:.1f} bpm, Min SpO2: {min_spo2:.1f}%")

    return state
