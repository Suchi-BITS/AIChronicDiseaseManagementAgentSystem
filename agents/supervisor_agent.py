# agents/supervisor_agent.py
# Supervisor Agent - orchestrates agent graph and produces final patient progress summary

import json
from datetime import datetime, date, timedelta
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.intervention_tools import log_automated_clinical_note
from config.settings import care_config
from data.models import AgentState, PatientProgressSummary


SUPERVISOR_SYSTEM_PROMPT = """You are the Clinical Supervisor AI for a chronic disease management system.

You produce the final patient progress summary that goes to:
1. The patient's care team (physician, care coordinator)
2. The patient themselves (adapted version)
3. The EHR as an automated monitoring note

Your output must be:
- Clinically precise: reference actual values, not vague descriptions
- Actionable: care team must know exactly what to do next
- Balanced: acknowledge progress as well as concerns
- Honest about system limitations: AI supports, not replaces, clinical judgment

SAFETY AND ETHICS REMINDERS:
- This is decision support only — all clinical actions require human clinician approval
- Emergency situations should have already triggered emergency protocols from the intervention agent
- Maintain patient dignity in all documentation
- Never include discriminatory or stigmatizing language

PROGRESS SUMMARY STRUCTURE:
1. Overall health status this period (excellent/good/fair/poor/critical)
2. Key findings per domain (glycemic, BP, medication, symptoms, labs)
3. Interventions already executed by AI system
4. Care plan changes recommended for physician review
5. Patient achievements to reinforce positive behavior
6. Areas needing focused attention next week
7. Goals for next monitoring period

DISCLAIMER TO ALWAYS INCLUDE:
{disclaimer}"""


def run_supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor agent: entry-point routing and final synthesis.
    """
    print("\n[SUPERVISOR] Processing state...")

    # Initial entry — load patient profile and route to first monitoring agent
    if not state.patient:
        from tools.data_collection_tools import get_patient_profile
        print("[SUPERVISOR] Loading patient profile and initiating monitoring cycle...")

        # Default to P001 (can be parameterized in production)
        patient_raw = get_patient_profile.invoke({"patient_id": "P001"})
        from data.models import PatientProfile
        import datetime as dt

        # Convert date strings to date objects if needed
        patient_raw["enrolled_since"] = date.today() - timedelta(days=500)
        state.patient = PatientProfile(**patient_raw)
        state.current_agent = "vitals_agent"
        state.iteration_count += 1
        print(f"[SUPERVISOR] Monitoring {state.patient.name} "
              f"({', '.join(state.patient.conditions)})")
        return state

    # Final synthesis after all monitoring and intervention agents have completed
    print("[SUPERVISOR] Synthesizing final patient progress summary...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    # Build risk score summary
    risk_by_domain = {rs.domain: rs for rs in state.risk_scores}
    overall_score = max((rs.score for rs in state.risk_scores), default=0)

    # Build intervention summary
    intervention_summary = []
    for inv in state.interventions:
        intervention_summary.append({
            "type": inv.intervention_type,
            "severity": inv.severity,
            "title": inv.title,
            "recipient": inv.recipient
        })

    care_plan_summary = []
    for adj in state.care_plan_adjustments:
        care_plan_summary.append({
            "change_type": adj.adjustment_type,
            "recommendation": adj.recommended_change,
            "urgency": adj.urgency,
            "guideline_basis": adj.evidence_level
        })

    # Determine overall status
    if state.emergency_triggered or overall_score > 80:
        overall_status = "critical"
    elif overall_score > 60:
        overall_status = "poor"
    elif overall_score > 40:
        overall_status = "fair"
    elif overall_score > 20:
        overall_status = "good"
    else:
        overall_status = "excellent"

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT.format(
            disclaimer=care_config.disclaimer
        )),
        HumanMessage(content=f"""
Generate a comprehensive patient progress summary:

PATIENT: {state.patient.name} (ID: {state.patient.patient_id})
Age: {state.patient.age} | Sex: {state.patient.sex}
Conditions: {', '.join(state.patient.conditions)}
Risk Stratification: {state.patient.risk_stratification.upper()}
Care Team: {json.dumps(state.patient.care_team)}
Monitoring Period: {date.today() - timedelta(days=7)} to {date.today()}

RISK SCORES BY DOMAIN:
{json.dumps([{
    'domain': rs.domain,
    'score': rs.score,
    'level': rs.risk_level,
    'factors': rs.contributing_factors
} for rs in state.risk_scores], indent=2)}

OVERALL COMPOSITE RISK: {overall_score:.0f}/100 -> Status: {overall_status.upper()}

INTERVENTIONS ALREADY EXECUTED ({len(state.interventions)}):
{json.dumps(intervention_summary, indent=2)}

CARE PLAN CHANGES RECOMMENDED ({len(state.care_plan_adjustments)}) [PENDING PHYSICIAN REVIEW]:
{json.dumps(care_plan_summary, indent=2)}

EMERGENCY TRIGGERED: {state.emergency_triggered}

MONITORING ANALYSES:
- Vitals: {state.vitals_analysis[:400] if state.vitals_analysis else "N/A"}
- Glycemic: {state.glucose_analysis[:400] if state.glucose_analysis else "N/A"}
- Blood Pressure: {state.bp_analysis[:400] if state.bp_analysis else "N/A"}
- Medication: {state.medication_analysis[:400] if state.medication_analysis else "N/A"}
- Symptoms/Labs: {state.symptom_analysis[:400] if state.symptom_analysis else "N/A"}

Produce:
1. Overall status assessment with clinical justification
2. Key findings per monitoring domain
3. Summary of AI interventions already taken
4. Prioritized action list for the care team (physician/coordinator)
5. Patient achievements this week (positive reinforcement)
6. Personalized goals for next monitoring period
7. Include the required safety disclaimer at the end
""")
    ]

    response = llm.invoke(messages)
    narrative = response.content

    # Log automated clinical note to EHR
    log_automated_clinical_note.invoke({
        "patient_id": state.patient.patient_id,
        "note_type": "remote_monitoring_note",
        "monitoring_period": f"{date.today() - timedelta(days=7)} to {date.today()}",
        "clinical_observations": (
            f"AI monitoring cycle completed. Risk domains assessed: "
            f"{', '.join(rs.domain for rs in state.risk_scores)}. "
            f"Overall composite risk: {overall_score:.0f}/100 ({overall_status})."
        ),
        "interventions_taken": (
            f"{len(state.interventions)} interventions executed: "
            f"{', '.join(set(i.intervention_type for i in state.interventions))}. "
            f"Emergency protocols activated: {state.emergency_triggered}."
        ),
        "plan": (
            f"{len(state.care_plan_adjustments)} care plan modifications pending "
            f"physician review. Next monitoring cycle: standard protocol."
        )
    })

    # Build progress summary
    state.progress_summary = PatientProgressSummary(
        patient_id=state.patient.patient_id,
        period_start=date.today() - timedelta(days=7),
        period_end=date.today(),
        overall_status=overall_status,
        glycemic_control_summary=state.glucose_analysis[:200] if state.glucose_analysis else None,
        bp_control_summary=state.bp_analysis[:200] if state.bp_analysis else None,
        medication_adherence_summary=state.medication_analysis[:200] if state.medication_analysis else "",
        symptom_summary=state.symptom_analysis[:200] if state.symptom_analysis else "",
        risk_scores=state.risk_scores,
        areas_of_concern=[f.strip() for rs in state.risk_scores for f in rs.contributing_factors],
        narrative=narrative
    )

    state.current_agent = "complete"
    print(f"[SUPERVISOR] Monitoring complete. Patient status: {overall_status.upper()}. "
          f"Risk: {overall_score:.0f}/100. "
          f"Interventions: {len(state.interventions)}. "
          f"Care plan recs: {len(state.care_plan_adjustments)}.")

    return state
