# agents/reporter_agent.py
# Reporter Agent — builds the final patient progress summary and updates memory.

import json
from datetime import datetime, timedelta
from config.settings import care_config
from memory.session_memory import get_memory
from agents.base import call_llm
from data.simulation import PATIENTS


def run_reporter(state: dict) -> dict:
    """
    Builds the final monitoring pass report and updates session memory.
    """
    patient_id = state.get("patient_id", "P001")
    patient    = PATIENTS.get(patient_id, {})
    memory     = get_memory(patient_id)
    signals    = state.get("signals", {})
    executed   = state.get("executed_actions", [])
    queued     = state.get("queued_for_approval", [])

    print(f"\n[REPORTER] Generating patient progress summary...")

    risk_level = signals.get("overall_risk", "moderate")
    state["risk_level"] = risk_level

    # Update memory with this pass
    memory.record_pass(state)
    trend = memory.get_trend()

    demo = _build_demo_summary(patient, signals, executed, queued, trend, risk_level)

    summary = call_llm(
        system_prompt=(
            f"You are the Clinical Reporter AI for {care_config.system_name}.\n"
            f"Produce a concise patient monitoring pass summary for the care team.\n"
            f"Be clinical, precise, and use standard medical abbreviations.\n\n"
            f"SESSION MEMORY:\n{memory.context_summary()}"
        ),
        user_prompt=(
            f"Patient: {patient.get('name')}, "
            f"conditions: {', '.join(patient.get('conditions', []))}\n"
            f"Risk level: {risk_level.upper()}\n"
            f"Signals: {json.dumps(signals, indent=2)}\n"
            f"Actions executed: {len(executed)}\n"
            f"Pending human approval: {len(queued)}\n"
            f"Trend: {json.dumps(trend, indent=2)}\n\n"
            f"Produce a structured clinical progress note."
        ),
        demo_response=demo,
    )

    state["progress_summary"] = summary
    state["current_agent"]    = "complete"

    print(f"  [REPORTER] Summary complete. Risk: {risk_level.upper()} | "
          f"Pass #{memory.pass_count}")
    return state


def _build_demo_summary(patient, signals, executed, queued, trend, risk_level) -> str:
    name = patient.get("name", "Patient")
    conds = ", ".join(patient.get("conditions", []))
    alerts  = signals.get("alerts", [])
    warns   = signals.get("warnings", [])
    mem_context = ""
    if trend.get("available"):
        mem_context = (
            f"\nTREND (from session memory):\n"
            f"  BP trend: {trend.get('bp_trend','N/A').upper()} | "
            f"Glucose trend: {trend.get('glucose_trend','N/A').upper()} | "
            f"Adherence: {trend.get('adherence_avg','N/A')}%"
        )
        if trend.get("consecutive_high_bp", 0) >= 3:
            mem_context += (
                f"\n  *** {trend['consecutive_high_bp']} consecutive passes with "
                f"elevated BP — escalation recommended"
            )

    return (
        f"PATIENT MONITORING SUMMARY — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"Patient: {name} | Conditions: {conds}\n"
        f"Risk Level: {risk_level.upper()}\n\n"
        f"CLINICAL STATUS:\n"
        + (f"  Glucose max: {signals.get('max_glucose','N/A')} mg/dL | "
           f"min: {signals.get('min_glucose','N/A')} mg/dL\n"
           if signals.get("max_glucose") else "")
        + (f"  BP max systolic: {signals.get('max_bp_systolic','N/A')} mmHg | "
           f"avg: {signals.get('avg_bp_systolic','N/A')} mmHg\n"
           if signals.get("max_bp_systolic") else "")
        + (f"  Alerts: {', '.join(alerts)}\n" if alerts else "  Alerts: None\n")
        + (f"  Warnings: {', '.join(warns)}\n" if warns else "")
        + mem_context
        + f"\n\nACTIONS TAKEN:\n"
        + "\n".join(
            f"  [{a.get('urgency','?').upper()}] {a.get('type','?')} -> "
            f"{a.get('recipient','?')}: dispatched via {a.get('channel','?')}"
            for a in executed
        )
        + (f"\n\nAWAITING HUMAN APPROVAL ({len(queued)}):\n"
           + "\n".join(
               f"  [{q.get('urgency','?').upper()}] {q.get('title','')[:70]}"
               for q in queued
           )
           if queued else "")
        + f"\n\n{care_config.disclaimer}"
    )
