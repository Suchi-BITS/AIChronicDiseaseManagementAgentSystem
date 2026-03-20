# agents/planner_agent.py
# Care Planner Agent
#
# ══════════════════════════════════════════════════════════════════════════════
# PLANNER ROLE IN PLANNER-EXECUTOR PATTERN
# ══════════════════════════════════════════════════════════════════════════════
#
# The planner does NOT execute actions. It:
#   1. Reads all patient data (vitals, glucose, BP, medications, symptoms, labs)
#   2. Reads session memory (trend context from prior monitoring passes)
#   3. Computes a structured care plan: a prioritised list of actions to take
#   4. Tags each action with: urgency, type, recipient, HITL required (yes/no)
#   5. Passes the plan to the executor agent
#
# Actions tagged HITL=True are queued for physician/nurse approval.
# Actions tagged HITL=False (routine, low-urgency) are executed immediately.
#
# This separation is clinically critical: the LLM never directly triggers
# medication changes or emergency alerts — it produces a plan that is
# validated by the executor and filtered through human approval gates.

import json
from datetime import datetime
from config.settings import care_config
from data.simulation import (
    PATIENTS, get_vitals, get_glucose, get_bp_readings,
    get_medications, get_symptoms, get_labs,
)
from memory.session_memory import get_memory
from agents.base import call_llm


def run_planner(state: dict) -> dict:
    """
    Planner agent: gathers all data, reads memory context, produces a care plan.
    """
    patient_id = state.get("patient_id", "P001")
    patient    = PATIENTS.get(patient_id, {})
    memory     = get_memory(patient_id)

    print(f"\n[PLANNER] Building care plan for {patient.get('name')} "
          f"(pass #{memory.pass_count + 1})...")

    # ── Load all patient data ──────────────────────────────────────────────────
    vitals  = get_vitals(patient_id)
    glucose = get_glucose(patient_id)
    bp      = get_bp_readings(patient_id)
    meds    = get_medications(patient_id)
    syms    = get_symptoms(patient_id)
    labs    = get_labs(patient_id)

    state.update({
        "vitals_readings":    vitals,
        "glucose_readings":   glucose,
        "bp_readings":        bp,
        "medication_records": meds,
        "symptom_reports":    syms,
        "lab_results":        labs,
    })

    # ── Compute clinical signals ───────────────────────────────────────────────
    signals = _compute_signals(vitals, glucose, bp, meds, syms, labs, patient)

    # ── Memory context ─────────────────────────────────────────────────────────
    mem_summary = memory.context_summary()
    trend       = memory.get_trend()
    print(f"  Memory context: {mem_summary.split(chr(10))[0]}")

    # ── Emergency check (bypass planner, immediate escalation) ────────────────
    if signals.get("emergency"):
        state["emergency_triggered"] = True
        emergency_plan = [{
            "action_id":  f"ACT-EMRG-{datetime.now().strftime('%H%M%S')}",
            "type":       "emergency_alert",
            "urgency":    "emergency",
            "title":      f"EMERGENCY: {signals['emergency_reason']}",
            "recipient":  "emergency_services_and_care_team",
            "instructions": signals["emergency_reason"],
            "hitl_required": False,   # Emergency alerts bypass HITL
            "auto_execute":  True,
        }]
        state["care_plan"] = emergency_plan
        state["signals"]   = signals
        state["current_agent"] = "executor"
        print(f"  [PLANNER] EMERGENCY DETECTED: {signals['emergency_reason']}")
        return state

    # ── Build structured care plan via LLM ────────────────────────────────────
    demo_plan = _build_demo_plan(signals, trend, patient)

    plan_text = call_llm(
        system_prompt=(
            f"You are the Care Planner AI for {care_config.system_name}.\n"
            f"You produce a structured list of care actions — you do NOT execute them.\n"
            f"For each action specify: type, urgency (routine/urgent/emergency), "
            f"recipient (patient/care_team/physician), title, instructions, "
            f"and whether human approval is required (hitl_required: true/false).\n"
            f"Mark hitl_required=true for: medication changes, urgent physician contact, "
            f"care plan modifications.\n"
            f"Mark hitl_required=false for: patient education messages, routine check-in SMS, "
            f"lifestyle reminders.\n\n"
            f"SESSION MEMORY:\n{mem_summary}"
        ),
        user_prompt=(
            f"Patient: {patient.get('name')}, age {patient.get('age')}, "
            f"conditions: {', '.join(patient.get('conditions', []))}\n\n"
            f"CLINICAL SIGNALS:\n{json.dumps(signals, indent=2)}\n\n"
            f"LABS:\n{json.dumps(labs, indent=2)}\n\n"
            f"TREND CONTEXT:\n{json.dumps(trend, indent=2)}\n\n"
            f"Produce a prioritised care plan as a JSON list of action objects."
        ),
        demo_response=json.dumps(demo_plan, indent=2),
    )

    # Parse or use demo plan
    try:
        import re
        json_match = re.search(r"\[.*\]", plan_text.replace("[DEMO] ", ""), re.DOTALL)
        care_plan  = json.loads(json_match.group(0)) if json_match else demo_plan
    except Exception:
        care_plan = demo_plan

    state["care_plan"]       = care_plan
    state["signals"]         = signals
    state["current_agent"]   = "executor"
    state["risk_level"]      = signals.get("overall_risk", "moderate")

    print(f"  [PLANNER] Plan ready: {len(care_plan)} action(s), "
          f"emergency={state.get('emergency_triggered', False)}")
    return state


def _compute_signals(vitals, glucose, bp, meds, syms, labs, patient) -> dict:
    """Deterministic signal computation — no LLM needed for threshold checking."""
    signals = {
        "emergency":         False,
        "emergency_reason":  "",
        "alerts":            [],
        "warnings":          [],
        "low_adherence_meds":[],
        "overall_risk":      "low",
    }

    # Glucose
    if glucose:
        max_gluc = max(r["glucose_mgdl"] for r in glucose)
        min_gluc = min(r["glucose_mgdl"] for r in glucose)
        if min_gluc < care_config.glucose_critical_low:
            signals["emergency"]        = True
            signals["emergency_reason"] = f"Critical hypoglycaemia: glucose {min_gluc} mg/dL"
        elif max_gluc > care_config.glucose_critical_high:
            signals["alerts"].append(f"Very high glucose: {max_gluc} mg/dL — possible DKA risk")
        elif max_gluc > care_config.glucose_target_high:
            signals["warnings"].append(f"Elevated glucose: {max_gluc} mg/dL (target <180)")
        signals["max_glucose"] = max_gluc
        signals["min_glucose"] = min_gluc

    # Blood pressure
    if bp:
        max_sys = max(r["systolic_mmhg"] for r in bp)
        avg_sys = round(sum(r["systolic_mmhg"] for r in bp) / len(bp), 1)
        if max_sys >= care_config.bp_systolic_critical:
            signals["emergency"]        = True
            signals["emergency_reason"] = f"Hypertensive crisis: SBP {max_sys} mmHg"
        elif avg_sys > care_config.bp_systolic_high:
            signals["alerts"].append(f"BP consistently elevated: avg SBP {avg_sys} mmHg")
        signals["max_bp_systolic"] = max_sys
        signals["avg_bp_systolic"] = avg_sys

    # SpO2
    if vitals:
        min_spo2 = min(r["spo2_pct"] for r in vitals)
        if min_spo2 < care_config.spo2_critical:
            signals["emergency"]        = True
            signals["emergency_reason"] = f"Critical SpO2: {min_spo2}%"
        elif min_spo2 < 92:
            signals["alerts"].append(f"Low SpO2: {min_spo2}%")

    # Heart rate
    if vitals:
        for v in vitals:
            hr = v["heart_rate_bpm"]
            if hr < care_config.hr_critical_low or hr > care_config.hr_critical_high:
                signals["emergency"]        = True
                signals["emergency_reason"] = f"Abnormal heart rate: {hr} bpm"
                break

    # Medication adherence
    for m in meds:
        if m["adherence_pct"] < care_config.adherence_low_pct:
            signals["low_adherence_meds"].append({
                "medication": m["medication"],
                "adherence":  m["adherence_pct"],
            })
            signals["warnings"].append(
                f"Low adherence: {m['medication']} at {m['adherence_pct']}%"
            )

    # Labs
    if labs:
        hba1c = labs.get("hba1c_pct")
        if hba1c and hba1c > 10:
            signals["alerts"].append(f"HbA1c critically high: {hba1c}%")
        egfr = labs.get("egfr_ml_min")
        if egfr and egfr < 30:
            signals["alerts"].append(f"Severely reduced eGFR: {egfr} mL/min — review nephrotoxic meds")

    # Overall risk
    if signals["emergency"]:
        signals["overall_risk"] = "critical"
    elif len(signals["alerts"]) >= 2:
        signals["overall_risk"] = "high"
    elif signals["alerts"] or signals["warnings"]:
        signals["overall_risk"] = "moderate"

    return signals


def _build_demo_plan(signals, trend, patient) -> list:
    plan = []
    i = 0

    for alert in signals.get("alerts", []):
        i += 1
        plan.append({
            "action_id":    f"ACT-{i:03d}",
            "type":         "physician_notification",
            "urgency":      "urgent",
            "title":        f"Clinical Alert: {alert[:60]}",
            "recipient":    "physician",
            "instructions": (
                f"Patient {patient.get('name')} — {alert}. "
                f"Review and determine if medication or care plan adjustment is needed."
            ),
            "hitl_required": True,
        })

    for warn in signals.get("warnings", []):
        i += 1
        plan.append({
            "action_id":    f"ACT-{i:03d}",
            "type":         "patient_education",
            "urgency":      "routine",
            "title":        f"Patient reminder: {warn[:60]}",
            "recipient":    "patient",
            "instructions": (
                f"Reminder about {warn}. "
                f"Please continue taking medications as prescribed and monitor closely."
            ),
            "hitl_required": False,
        })

    for med in signals.get("low_adherence_meds", []):
        i += 1
        plan.append({
            "action_id":    f"ACT-{i:03d}",
            "type":         "adherence_support",
            "urgency":      "routine",
            "title":        f"Adherence support: {med['medication']}",
            "recipient":    "care_coordinator",
            "instructions": (
                f"{med['medication']} adherence is {med['adherence']}% — "
                f"call patient to identify barriers and reinforce importance."
            ),
            "hitl_required": False,
        })

    if trend.get("available") and trend.get("consecutive_high_bp", 0) >= 3:
        i += 1
        plan.append({
            "action_id":    f"ACT-{i:03d}",
            "type":         "care_plan_modification",
            "urgency":      "urgent",
            "title":        "Persistent hypertension — medication review required",
            "recipient":    "physician",
            "instructions": (
                f"Patient has had {trend['consecutive_high_bp']} consecutive monitoring "
                f"passes with elevated BP. Current regimen may need dose adjustment."
            ),
            "hitl_required": True,
        })

    if not plan:
        plan.append({
            "action_id":    "ACT-001",
            "type":         "routine_check_in",
            "urgency":      "routine",
            "title":        "Routine monitoring — no urgent issues",
            "recipient":    "patient",
            "instructions": "All monitored parameters within acceptable range. "
                            "Continue current regimen. Next check-in in 24 hours.",
            "hitl_required": False,
        })

    return plan
