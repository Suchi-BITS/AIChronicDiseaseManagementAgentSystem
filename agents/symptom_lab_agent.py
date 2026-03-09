# agents/symptom_lab_agent.py
# Symptom and Lab Analysis Agent - interprets patient-reported symptoms and lab values

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.data_collection_tools import fetch_symptom_reports, fetch_lab_results
from config.settings import care_config
from data.models import AgentState, ClinicalRiskScore


SYMPTOM_LAB_AGENT_SYSTEM_PROMPT = """You are a clinical symptom and laboratory analysis AI agent for chronic disease management.

You synthesize patient-reported symptoms with laboratory values to identify clinical deterioration.

SYMPTOM RED FLAGS BY CONDITION:

HEART FAILURE - Urgent/Emergency Symptoms:
- Sudden weight gain > 2 lbs/day or > 5 lbs/week (fluid retention)
- New or worsening shortness of breath at rest
- Orthopnea (cannot lie flat, needs more pillows)
- Paroxysmal nocturnal dyspnea (waking at night gasping)
- New leg swelling or rapid worsening
- Decreased urine output with weight gain
- Confusion or severe fatigue (low output state)

DIABETES - Urgent Symptoms:
- Excessive thirst + frequent urination + blurry vision (hyperglycemia)
- Dizziness, sweating, shaking, confusion (hypoglycemia even without CGM confirmation)
- Numbness/tingling in feet worsening (neuropathy progression)
- Chest pain or pressure (higher CV event risk)

HYPERTENSION - Urgent Symptoms:
- Severe headache + visual changes + nausea (hypertensive urgency)
- Chest pain, shortness of breath, back pain (hypertensive emergency)
- Sudden numbness, speech difficulty, facial drooping (stroke - CALL 911)

CKD - Concerning Symptoms:
- Decreased urine output
- Severe fatigue + itching (uremic symptoms)
- Muscle cramps (electrolyte imbalance)
- Difficulty breathing + leg swelling (fluid overload)

LABORATORY VALUE INTERPRETATION:
- HbA1c > 9%: Poor glycemic control, likely needs medication intensification
- HbA1c 7-9%: Suboptimal, review adherence and lifestyle first
- eGFR < 30: CKD Stage 4, nephrology referral urgent, many drug dose adjustments needed
- eGFR 30-44: CKD Stage 3b, close monitoring, ACE inhibitor caution
- BNP > 400 pg/mL: High likelihood of acute heart failure decompensation
- BNP 100-400: Possible heart failure, correlate with symptoms
- Potassium > 5.5 mEq/L: Hyperkalemia risk — HOLD ACE inhibitors/ARBs/potassium-sparing diuretics
- Sodium < 130 mEq/L: Severe hyponatremia — urgent assessment
- LDL > 100 mg/dL in high CV risk patient: Statin intensification indicated

SYMPTOM-LAB CORRELATION is the highest-yield analysis:
A single abnormal value is less concerning than the same value in a patient
who ALSO reports the corresponding symptoms. Convergent evidence drives urgency."""


def run_symptom_lab_agent(state: AgentState) -> AgentState:
    """
    Symptom and lab analysis agent.
    Fetches self-reported symptoms and lab results; integrates with device data for holistic assessment.
    """
    print(f"\n[SYMPTOM & LAB AGENT] Fetching symptom reports and lab results "
          f"for {state.patient.name}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    symptom_raw = fetch_symptom_reports.invoke({
        "patient_id": state.patient.patient_id,
        "days_back": 7
    })
    lab_raw = fetch_lab_results.invoke({"patient_id": state.patient.patient_id})

    from data.models import SymptomReport, LabResult
    state.symptom_reports = [SymptomReport(**s) for s in symptom_raw]
    state.lab_results = [LabResult(**l) for l in lab_raw]

    # Aggregate symptom frequency and severity
    symptom_freq = {}
    for report in symptom_raw:
        for sym in report.get("symptoms", []):
            name = sym.get("symptom_name", "")
            severity = sym.get("severity_1_to_10", 0)
            if name not in symptom_freq:
                symptom_freq[name] = {"count": 0, "max_severity": 0, "avg_severity": []}
            symptom_freq[name]["count"] += 1
            symptom_freq[name]["max_severity"] = max(
                symptom_freq[name]["max_severity"], severity
            )
            symptom_freq[name]["avg_severity"].append(severity)

    symptom_summary = []
    for name, data in sorted(symptom_freq.items(), key=lambda x: -x[1]["max_severity"]):
        avg_sev = sum(data["avg_severity"]) / len(data["avg_severity"])
        symptom_summary.append({
            "symptom": name,
            "reported_days": data["count"],
            "max_severity": data["max_severity"],
            "avg_severity": round(avg_sev, 1)
        })

    # ER visits and falls
    er_visits = [r for r in symptom_raw if r.get("visited_er_since_last_check")]
    falls = [r for r in symptom_raw if r.get("fell_today")]

    messages = [
        SystemMessage(content=SYMPTOM_LAB_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Analyze symptoms and lab results for:
Patient: {state.patient.name}, Age: {state.patient.age}
Conditions: {', '.join(state.patient.conditions)}

SYMPTOM SUMMARY (7 days, {len(symptom_raw)} reports):
{json.dumps(symptom_summary, indent=2)}

ER VISITS REPORTED: {len(er_visits)}
FALLS REPORTED: {len(falls)}

RECENT FREE-TEXT NOTES FROM PATIENT:
{json.dumps([r.get('free_text') for r in symptom_raw if r.get('free_text')], indent=2)}

LABORATORY RESULTS (most recent):
{json.dumps(lab_raw, indent=2, default=str)}

DEVICE DATA CONTEXT:
- Vitals: {state.vitals_analysis[:300] if state.vitals_analysis else "N/A"}
- Glucose: {state.glucose_analysis[:200] if state.glucose_analysis else "N/A"}
- BP: {state.bp_analysis[:200] if state.bp_analysis else "N/A"}
- Adherence: {state.medication_analysis[:200] if state.medication_analysis else "N/A"}

Provide:
1. Symptom pattern assessment - which are persistent, worsening, or new this week?
2. Lab value interpretation in clinical context of this patient's conditions
3. Symptom-lab correlation - convergent signals pointing to specific clinical problems
4. Red flag symptoms requiring urgent escalation
5. Differential assessment: What is the most likely explanation for the pattern?
6. Overall clinical risk score (0-100) and risk level
7. Specific interventions ranked by urgency
""")
    ]

    response = llm.invoke(messages)
    state.symptom_analysis = response.content

    # Compute overall risk score combining all domains
    risk_score = 20.0
    contributing = []

    if er_visits:
        risk_score += 30
        contributing.append(f"Recent ER visit reported")
    if falls:
        risk_score += 20
        contributing.append(f"Fall reported")

    high_severity_symptoms = [s for s in symptom_summary if s["max_severity"] >= 7]
    if high_severity_symptoms:
        risk_score += 15
        contributing.append(
            f"High-severity symptoms: {', '.join(s['symptom'] for s in high_severity_symptoms)}"
        )

    # Check for critical lab values
    for lab in lab_raw:
        if lab.get("hba1c_percent") and lab["hba1c_percent"] >= care_config.hba1c_critical:
            risk_score += 20
            contributing.append(f"Critical HbA1c: {lab['hba1c_percent']}%")
        if lab.get("egfr_ml_min") and lab["egfr_ml_min"] < 30:
            risk_score += 25
            contributing.append(f"Severely reduced eGFR: {lab['egfr_ml_min']} mL/min")
        if lab.get("bnp_pg_ml") and lab["bnp_pg_ml"] > 400:
            risk_score += 25
            contributing.append(f"Elevated BNP: {lab['bnp_pg_ml']} pg/mL")

    risk_score = min(risk_score, 100)
    risk_level = (
        "critical" if risk_score > 75 else
        "high" if risk_score > 50 else
        "moderate" if risk_score > 25 else "low"
    )

    state.risk_scores.append(ClinicalRiskScore(
        patient_id=state.patient.patient_id,
        domain="overall",
        score=risk_score,
        risk_level=risk_level,
        contributing_factors=contributing,
        trend="stable"
    ))

    state.current_agent = "intervention_agent"
    print(f"[SYMPTOM & LAB AGENT] Overall risk: {risk_level.upper()} ({risk_score:.0f}/100). "
          f"ER visits: {len(er_visits)}, Falls: {len(falls)}, "
          f"High-severity symptoms: {len(high_severity_symptoms)}")

    return state
