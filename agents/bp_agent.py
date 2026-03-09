# agents/bp_agent.py
# Blood Pressure Monitoring Agent - analyzes BP readings and hypertensive risk

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.data_collection_tools import fetch_blood_pressure_readings
from config.settings import care_config
from data.models import AgentState, ClinicalRiskScore


BP_AGENT_SYSTEM_PROMPT = """You are a clinical blood pressure monitoring AI agent for chronic disease management.

You interpret home blood pressure monitoring (HBPM) data using ACC/AHA guidelines.

BLOOD PRESSURE CLASSIFICATIONS (ACC/AHA 2017):
- Normal: < 120/80 mmHg
- Elevated: 120-129 / < 80 mmHg
- Stage 1 Hypertension: 130-139 / 80-89 mmHg
- Stage 2 Hypertension: >= 140 / >= 90 mmHg
- Hypertensive Urgency: >= 180 / >= 120 without end-organ damage — URGENT, contact physician
- Hypertensive Emergency: >= 180 / >= 120 with symptoms (chest pain, vision changes, headache,
  neurological symptoms) — EMERGENCY, call 911

HOME BP TARGETS BY CONDITION:
- Hypertension (uncomplicated): < 130/80 mmHg
- Diabetes + Hypertension: < 130/80 mmHg
- CKD (no proteinuria): < 140/90 mmHg
- CKD (with proteinuria): < 130/80 mmHg
- Heart Failure: personalized, often 100-130/60-80 mmHg
- Elderly (> 65): < 130/80 (if tolerated, avoid < 110 systolic)

MEASUREMENT INTERPRETATION RULES:
- Use average of last 7 readings, not single measurements
- Morning surge (first AM reading often highest) is expected
- White coat effect: Home readings 10-15 mmHg lower than clinic readings
- Diastolic > 120: Always urgent regardless of systolic
- Irregular heartbeat flag: Warrants ECG evaluation

CRITICAL DRUG INTERACTIONS FOR BP:
- NSAIDs can raise BP significantly and interfere with antihypertensives
- Lisinopril/ACE inhibitors: Watch for hyperkalemia in CKD
- Calcium channel blockers: Can cause ankle edema (mimic heart failure edema)
- Diuretics: Monitor for hypokalemia and dehydration in elderly

Generate a cardiovascular risk score component for BP domain."""


def run_bp_agent(state: AgentState) -> AgentState:
    """
    Blood pressure monitoring agent.
    Analyzes HBPM data for hypertension control and crisis detection.
    """
    has_hypertension = any(
        c in state.patient.conditions
        for c in ["hypertension", "heart_failure", "chronic_kidney_disease", "type2_diabetes"]
    )

    if not has_hypertension:
        print(f"\n[BP AGENT] Skipping - no BP-relevant conditions for {state.patient.patient_id}")
        state.bp_analysis = "Not applicable for this patient's conditions."
        state.current_agent = "medication_agent"
        return state

    print(f"\n[BP AGENT] Fetching blood pressure data for {state.patient.name}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    bp_raw = fetch_blood_pressure_readings.invoke({
        "patient_id": state.patient.patient_id,
        "days_back": 7
    })

    from data.models import BloodPressureReading
    state.bp_readings = [BloodPressureReading(**b) for b in bp_raw]

    # Compute BP statistics
    systolics = [b["systolic_mmhg"] for b in bp_raw]
    diastolics = [b["diastolic_mmhg"] for b in bp_raw]
    pulses = [b["pulse_bpm"] for b in bp_raw]
    irregular_detected = [b for b in bp_raw if b.get("irregular_heartbeat_detected")]

    avg_systolic = sum(systolics) / len(systolics) if systolics else None
    avg_diastolic = sum(diastolics) / len(diastolics) if diastolics else None
    max_systolic = max(systolics) if systolics else None
    max_diastolic = max(diastolics) if diastolics else None

    # Count readings above threshold
    high_readings = [
        b for b in bp_raw
        if b["systolic_mmhg"] >= care_config.bp_systolic_high
        or b["diastolic_mmhg"] >= 90
    ]
    critical_readings = [
        b for b in bp_raw
        if b["systolic_mmhg"] >= care_config.bp_systolic_critical_high
        or b["diastolic_mmhg"] >= care_config.bp_diastolic_critical_high
    ]

    messages = [
        SystemMessage(content=BP_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze 7-day home blood pressure data for:
Patient: {state.patient.name}, Age: {state.patient.age}
Conditions: {', '.join(state.patient.conditions)}
Antihypertensive medications: {', '.join(
    f"{m['name']} {m['dose']}"
    for m in state.patient.medications
    if m['name'] in ['Lisinopril', 'Amlodipine', 'Carvedilol', 'Furosemide',
                     'Spironolactone', 'Metoprolol', 'Losartan', 'Hydrochlorothiazide']
)}

7-DAY BP SUMMARY ({len(bp_raw)} readings):
- Average: {avg_systolic:.0f}/{avg_diastolic:.0f} mmHg
- Maximum: {max_systolic:.0f}/{max_diastolic:.0f} mmHg
- Readings above stage 2 threshold (>=140/90): {len(high_readings)} of {len(bp_raw)}
- Hypertensive urgency/emergency readings (>=180 systolic): {len(critical_readings)}
- Irregular heartbeat detected: {len(irregular_detected)} times

RECENT READINGS (last 5):
{json.dumps([{
    'time': b['timestamp'],
    'bp': f"{b['systolic_mmhg']:.0f}/{b['diastolic_mmhg']:.0f}",
    'pulse': b['pulse_bpm'],
    'irregular': b['irregular_heartbeat_detected']
} for b in bp_raw[:5]], indent=2, default=str)}

BP TARGET FOR THIS PATIENT:
Based on conditions (diabetes + CKD): Target < 130/80 mmHg

VITALS CONTEXT:
{state.vitals_analysis[:300] if state.vitals_analysis else "Not available"}

Provide:
1. BP control assessment vs patient-specific target
2. Trend analysis: morning surge, evening readings, variability
3. Relationship to current antihypertensive regimen
4. Any urgent findings (hypertensive urgency/emergency flags)
5. Impact of poor BP control on concurrent CKD and diabetes
6. Recommended interventions (lifestyle, medication review, escalation)
7. BP-specific risk score (0-100) with contributing factors
""")
    ]

    response = llm.invoke(messages)
    state.bp_analysis = response.content

    # Compute BP risk score
    risk_score = 15.0
    contributing = []

    if critical_readings:
        risk_score += 55
        contributing.append(
            f"Hypertensive urgency readings: {len(critical_readings)} readings >= 180 systolic"
        )
    elif len(high_readings) > len(bp_raw) * 0.5:
        risk_score += 30
        contributing.append(
            f"Majority of readings elevated: {len(high_readings)}/{len(bp_raw)} above 140/90"
        )
    elif avg_systolic and avg_systolic > care_config.bp_systolic_target_high:
        risk_score += 20
        contributing.append(f"Average systolic above target: {avg_systolic:.0f} mmHg")
    if irregular_detected:
        risk_score += 15
        contributing.append(f"Irregular heartbeat detected {len(irregular_detected)} times")

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

    state.current_agent = "medication_agent"
    print(f"[BP AGENT] BP risk: {risk_level.upper()} ({risk_score:.0f}/100). "
          f"Avg: {avg_systolic:.0f}/{avg_diastolic:.0f} mmHg. "
          f"Critical readings: {len(critical_readings)}, Irregular rhythm: {len(irregular_detected)}")

    return state
