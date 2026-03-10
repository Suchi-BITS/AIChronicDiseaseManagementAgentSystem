#!/usr/bin/env python3
"""
AI Chronic Disease Management Agent System
==========================================
Run:  python main.py            (demo mode — no API key needed)
Run:  python main.py P002       (run for Thomas Rivera — heart failure patient)
Set OPENAI_API_KEY in .env for live LLM reasoning.
"""

import sys
import os
import json
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.models import make_agent_state
from data.simulation import PATIENTS
from config.settings import care_config


SEPARATOR = "=" * 72


def print_header(patient_id: str):
    p = PATIENTS.get(patient_id, PATIENTS["P001"])
    print(SEPARATOR)
    print(f"  {care_config.system_name}  —  {care_config.organization}")
    print(f"  Run time : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Patient  : {p['name']}  (ID: {patient_id})")
    print(f"  Age      : {p['age']}  |  Sex: {p['sex']}")
    print(f"  Conditions: {', '.join(p['conditions'])}")
    print(f"  Risk level: {p['risk_stratification'].upper()}")
    print(f"  Meds      : {len(p['medications'])} medications")
    print(SEPARATOR)
    print(f"\n  NOTE: {care_config.disclaimer}\n")


def print_report(final_state: dict):
    summary = final_state.get("progress_summary", {})
    if not summary:
        print("No summary generated.")
        return

    status   = summary.get("overall_status", "unknown").upper()
    risk     = summary.get("composite_risk", 0)
    status_icons = {"EXCELLENT":"[++]","GOOD":"[+]","FAIR":"[~]","POOR":"[-]","CRITICAL":"[!!]"}
    icon     = status_icons.get(status, "[ ]")

    print(f"\n{SEPARATOR}")
    print(f"  PATIENT MONITORING REPORT   {icon} {status}   (composite risk: {risk:.0f}/100)")
    print(SEPARATOR)

    # Risk scores table
    print("\n  RISK SCORES BY DOMAIN:")
    print(f"  {'Domain':<25} {'Score':>6}   {'Level':<10}  Key factors")
    print(f"  {'-'*65}")
    for rs in summary.get("risk_scores", []):
        bar     = "#" * int(rs["score"] / 10) + "." * (10 - int(rs["score"] / 10))
        factors = " | ".join(rs["factors"][:2]) if rs["factors"] else "—"
        print(f"  {rs['domain']:<25} {rs['score']:>5.0f}   [{bar}]  {rs['risk_level'].upper():<8}  {factors[:45]}")

    # Interventions
    interventions = summary.get("interventions", [])
    if interventions:
        print(f"\n  INTERVENTIONS EXECUTED ({len(interventions)}):")
        for inv in interventions:
            sev  = inv.get("severity","info").upper()
            tag  = {"EMERGENCY":"[!!!]","URGENT":"[!!]","WARNING":"[!]","INFO":"[i]"}.get(sev,"[i]")
            print(f"    {tag} [{inv.get('type','').replace('_',' ').upper()}] -> {inv.get('recipient','')}")
            print(f"       {inv.get('title','')}")
            print(f"       Basis: {inv.get('clinical_basis','')[:80]}")

    # Care plan recommendations
    care_plan = summary.get("care_plan_adjustments", [])
    if care_plan:
        print(f"\n  CARE PLAN RECOMMENDATIONS — PENDING PHYSICIAN REVIEW ({len(care_plan)}):")
        for i, adj in enumerate(care_plan, 1):
            urg = adj.get("urgency","routine").upper()
            print(f"    {i}. [{urg}] {adj.get('change_type','').replace('_',' ').upper()}")
            print(f"       {adj.get('recommendation','')[:100]}")
            if adj.get("guideline"):
                print(f"       Guideline: {adj.get('guideline','')}")

    # Emergency flag
    if summary.get("emergency_triggered"):
        print("\n  !!!  EMERGENCY PROTOCOLS WERE ACTIVATED THIS CYCLE  !!!")
        print("       Emergency contact, EMS, and physician STAT page were sent.")

    # Full narrative
    print(f"\n  CLINICAL SUMMARY:")
    print("-" * 72)
    for line in summary.get("narrative","").strip().splitlines():
        print(f"  {line}")

    print(f"\n{SEPARATOR}\n")


def save_report(final_state: dict, patient_id: str) -> str:
    fname = f"care_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    print(f"  Full report saved to: {fname}")
    return fname


def run_monitoring_cycle(patient_id: str = "P001"):
    print_header(patient_id)

    # ---------------------------------------------------------------
    # Try to import LangGraph. If not installed, run agents directly.
    # ---------------------------------------------------------------
    try:
        from graph.care_graph import build_care_graph

        graph      = build_care_graph(patient_id)
        init_state = make_agent_state()
        init_state["target_patient_id"] = patient_id

        config = {"configurable": {"thread_id": f"care-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"}}

        print("Running agent graph via LangGraph...\n")
        final_state = None
        for step in graph.stream(init_state, config=config):
            for node_name, state in step.items():
                print(f"    Completed node: [{node_name}]")
                final_state = state

    except ImportError:
        # LangGraph not installed — run the agent pipeline directly
        print("  LangGraph not installed. Running agent pipeline directly.\n")
        from agents.all_agents import (
            run_supervisor, run_vitals_agent, run_glucose_agent,
            run_bp_agent, run_medication_agent, run_symptom_lab_agent,
            run_intervention_agent,
        )
        state = make_agent_state()
        state["target_patient_id"] = patient_id

        state = run_supervisor(state)       # init: load patient
        state = run_vitals_agent(state)
        state = run_glucose_agent(state)
        state = run_bp_agent(state)
        state = run_medication_agent(state)
        state = run_symptom_lab_agent(state)
        state = run_intervention_agent(state)
        state = run_supervisor(state)       # synthesis: generate summary
        final_state = state

    if final_state:
        print_report(final_state)
        save_report(final_state, patient_id)
    else:
        print("No output generated.")


if __name__ == "__main__":
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P001"
    if patient_id not in PATIENTS:
        print(f"Unknown patient. Available: {list(PATIENTS.keys())}")
        sys.exit(1)
    run_monitoring_cycle(patient_id)
