# agents/intervention_agent.py
# Intervention Planning Agent - generates targeted, evidence-based health interventions

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tools.intervention_tools import (
    send_patient_notification,
    alert_care_team,
    trigger_emergency_response,
    recommend_care_plan_change
)
from config.settings import care_config
from data.models import AgentState, HealthIntervention


INTERVENTION_AGENT_SYSTEM_PROMPT = """You are a clinical intervention planning AI agent for chronic disease management.

You synthesize all monitoring analyses to generate targeted, evidence-based interventions.

INTERVENTION DECISION FRAMEWORK:

EMERGENCY (call trigger_emergency_response):
- Critical hypoglycemia (glucose < 54 mg/dL) with patient unresponsive or symptoms
- Hypertensive emergency (SBP >= 180 with chest pain, vision changes, neurological symptoms)
- SpO2 < 88% with severe dyspnea in COPD/heart failure
- Acute decompensated heart failure with rapid deterioration

URGENT (alert_care_team with urgency='urgent' + send_patient_notification):
- BP readings consistently >= 180 systolic (hypertensive urgency without emergency symptoms)
- Glucose > 350 mg/dL without emergency symptoms
- SpO2 90-92% with worsening symptoms in COPD/heart failure
- BNP > 500 pg/mL with worsening edema/dyspnea symptoms
- New irregular heartbeat detection
- Falls reported

CLINICAL CARE COORDINATION (alert_care_team urgency='soon'):
- HbA1c >= 9% — medication intensification consideration
- eGFR < 30 — nephrology referral, dose adjustments
- Medication adherence < 50% for critical medications
- LDL > 100 mg/dL on maximum statin dose
- Potassium > 5.5 mEq/L on ACE inhibitor/spironolactone

PATIENT COACHING (send_patient_notification):
- Glucose trending high after meals — dietary guidance
- Activity below weekly target — motivational messaging
- Medication adherence 60-80% — reminder strategies
- Sleep < 6 hours — sleep hygiene tips
- Mild to moderate symptom burden — self-management guidance

CARE PLAN MODIFICATIONS (recommend_care_plan_change):
- Any lab value suggesting medication dose adjustment
- Consistent out-of-range vitals despite adherence
- Pattern suggesting specialist referral needed
- Monitoring frequency changes based on risk level change

SAFETY RULES:
1. Always provide clear, non-alarmist language in patient-facing messages
2. Never recommend specific medication doses in patient messages (physician domain)
3. Emergency actions should also trigger patient notification asking them to call 911
4. All care plan recommendations require physician approval flag
5. Document the clinical basis for every intervention

GUIDELINES TO REFERENCE:
- ADA Standards of Medical Care in Diabetes (2024)
- ACC/AHA Hypertension Guidelines (2017)
- AHA/ACC Heart Failure Guidelines (2022)
- KDIGO CKD Guidelines (2022)
- JNC 8 Hypertension Treatment Targets"""


def run_intervention_agent(state: AgentState) -> AgentState:
    """
    Intervention planning agent.
    Uses all monitoring analyses to generate and execute targeted interventions.
    """
    print(f"\n[INTERVENTION AGENT] Planning interventions for {state.patient.name}...")

    llm = ChatOpenAI(
        model=care_config.model_name,
        temperature=care_config.temperature,
        api_key=care_config.openai_api_key
    )

    tools = [
        send_patient_notification,
        alert_care_team,
        trigger_emergency_response,
        recommend_care_plan_change
    ]
    llm_with_tools = llm.bind_tools(tools)

    # Build comprehensive risk summary
    risk_summary = []
    for rs in state.risk_scores:
        risk_summary.append({
            "domain": rs.domain,
            "score": rs.score,
            "level": rs.risk_level,
            "factors": rs.contributing_factors
        })

    overall_risk = max((rs.score for rs in state.risk_scores), default=0)
    highest_risk_domain = max(state.risk_scores, key=lambda x: x.score, default=None)

    messages = [
        SystemMessage(content=INTERVENTION_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Generate interventions for:
Patient: {state.patient.name} (ID: {state.patient.patient_id})
Age: {state.patient.age} | Conditions: {', '.join(state.patient.conditions)}
Risk stratification: {state.patient.risk_stratification.upper()}
Care team: {state.patient.care_team}
Emergency contact: {state.patient.emergency_contact}
Estimated patient location: [on file with care management system]

RISK SCORES SUMMARY:
{json.dumps(risk_summary, indent=2)}

HIGHEST RISK DOMAIN: {highest_risk_domain.domain if highest_risk_domain else 'N/A'}
OVERALL COMPOSITE RISK SCORE: {overall_risk:.0f}/100

VITALS ANALYSIS:
{state.vitals_analysis or "Not available"}

GLYCEMIC ANALYSIS:
{state.glucose_analysis or "Not available"}

BLOOD PRESSURE ANALYSIS:
{state.bp_analysis or "Not available"}

MEDICATION ADHERENCE ANALYSIS:
{state.medication_analysis or "Not available"}

SYMPTOM AND LAB ANALYSIS:
{state.symptom_analysis or "Not available"}

Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Execute interventions in this priority order:
1. First: Any emergency triggers (life-threatening conditions)
2. Second: Urgent care team alerts (significant deterioration)
3. Third: Patient notifications (actionable self-management)
4. Fourth: Care plan recommendations (optimization for physician review)

Call multiple tools as needed. Be specific in clinical rationale.
Write patient messages in plain, empathetic, non-alarmist language.
""")
    ]

    # Agentic intervention loop (up to 2 rounds)
    max_rounds = 2
    intervention_count = 0

    for _ in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not hasattr(response, "tool_calls") or not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            result = None

            if tool_name == "send_patient_notification":
                result = send_patient_notification.invoke(args)
                state.interventions.append(HealthIntervention(
                    intervention_id=result.get("action_id", f"INT-{datetime.now().strftime('%H%M%S')}"),
                    patient_id=state.patient.patient_id,
                    intervention_type="in_app_nudge" if args.get("severity") == "info" else "medication_reminder",
                    severity=args.get("severity", "info"),
                    title=args.get("title", ""),
                    message=args.get("message", ""),
                    action_required=args.get("action_prompt", ""),
                    recipient="patient",
                    delivery_channel=args.get("channels", ["push_notification"]),
                    clinical_basis="Monitoring data analysis",
                    requires_acknowledgment=args.get("requires_acknowledgment", False)
                ))

            elif tool_name == "alert_care_team":
                result = alert_care_team.invoke(args)
                state.interventions.append(HealthIntervention(
                    intervention_id=result.get("action_id", f"INT-{datetime.now().strftime('%H%M%S')}"),
                    patient_id=state.patient.patient_id,
                    intervention_type=(
                        "care_coordinator_alert"
                        if args.get("recipient_role") == "care_coordinator"
                        else "physician_alert"
                    ),
                    severity=("urgent" if args.get("urgency") in ["urgent", "stat"] else "warning"),
                    title=args.get("subject", ""),
                    message=args.get("clinical_summary", ""),
                    action_required=args.get("recommended_action", ""),
                    recipient=args.get("recipient_role", "care_coordinator"),
                    delivery_channel=["ehr_secure_message", "pager"],
                    clinical_basis=str(args.get("relevant_data_points", []))
                ))

            elif tool_name == "trigger_emergency_response":
                result = trigger_emergency_response.invoke(args)
                state.emergency_triggered = True
                state.interventions.append(HealthIntervention(
                    intervention_id=result.get("action_id", f"EMG-{datetime.now().strftime('%H%M%S')}"),
                    patient_id=state.patient.patient_id,
                    intervention_type="emergency_alert",
                    severity="emergency",
                    title="EMERGENCY RESPONSE ACTIVATED",
                    message=args.get("reason", ""),
                    action_required="Emergency services notified",
                    recipient="emergency_services",
                    delivery_channel=["911_dispatch", "emergency_contact", "physician_pager"],
                    clinical_basis=str(args.get("vital_signs", {})),
                    requires_acknowledgment=True
                ))

            elif tool_name == "recommend_care_plan_change":
                result = recommend_care_plan_change.invoke(args)
                from data.models import CarePlanAdjustment
                state.care_plan_adjustments.append(CarePlanAdjustment(
                    patient_id=state.patient.patient_id,
                    adjustment_type=args.get("change_type", "lifestyle_goal_update"),
                    current_state=args.get("current_state", ""),
                    recommended_change=args.get("recommended_change", ""),
                    clinical_rationale=args.get("clinical_rationale", ""),
                    evidence_level="guideline_based",
                    urgency=args.get("urgency", "routine"),
                    requires_physician_approval=args.get("requires_physician_approval", True),
                    estimated_benefit="Improved disease control and reduced complication risk"
                ))

            if result:
                messages.append(ToolMessage(
                    content=json.dumps(result) if isinstance(result, dict) else str(result),
                    tool_call_id=tool_call["id"]
                ))
                intervention_count += 1

    state.current_agent = "supervisor"
    print(f"[INTERVENTION AGENT] Generated {len(state.interventions)} interventions, "
          f"{len(state.care_plan_adjustments)} care plan recommendations. "
          f"Emergency triggered: {state.emergency_triggered}")

    return state
