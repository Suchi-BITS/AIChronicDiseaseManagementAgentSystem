# main.py
# Entry point for the AI Chronic Disease Management Agent System

import os
import sys
import json
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.care_graph import build_care_graph, get_graph_description
from data.models import AgentState
from config.settings import care_config


def print_header():
    print("=" * 72)
    print("  AI CHRONIC DISEASE MANAGEMENT SYSTEM")
    print(f"  System: {care_config.system_name}")
    print(f"  Organization: {care_config.organization}")
    print(f"  Conditions managed: {', '.join(care_config.supported_conditions)}")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print()
    print(f"  DISCLAIMER: {care_config.disclaimer}")
    print()


def print_patient_report(final_state: dict):
    """Print formatted patient monitoring report."""
    summary = final_state.get("progress_summary")
    patient = final_state.get("patient")

    if not summary or not patient:
        print("No summary generated.")
        return

    status = summary.get("overall_status", "unknown").upper()
    status_map = {
        "EXCELLENT": "[++]", "GOOD": "[+]", "FAIR": "[~]",
        "POOR": "[-]", "CRITICAL": "[!!]"
    }
    indicator = status_map.get(status, "[ ]")

    print("\n" + "=" * 72)
    print(f"  PATIENT MONITORING REPORT  {indicator} {status}")
    print("=" * 72)
    print(f"  Patient: {patient.get('name')}  |  ID: {patient.get('patient_id')}")
    print(f"  Age: {patient.get('age')}  |  "
          f"Conditions: {', '.join(patient.get('conditions', []))}")
    print(f"  Risk Level: {patient.get('risk_stratification', 'N/A').upper()}")
    print(f"  Period: {summary.get('period_start')} to {summary.get('period_end')}")

    # Risk scores
    risk_scores = final_state.get("risk_scores", [])
    if risk_scores:
        print("\n--- RISK SCORES BY DOMAIN ---")
        for rs in risk_scores:
            level = rs.get("risk_level", "").upper()
            score = rs.get("score", 0)
            bar = "#" * int(score / 10) + "-" * (10 - int(score / 10))
            print(f"  {rs.get('domain'):20s} [{bar}] {score:.0f}/100  {level}")

    # Interventions executed
    interventions = final_state.get("interventions", [])
    if interventions:
        print(f"\n--- INTERVENTIONS EXECUTED ({len(interventions)}) ---")
        for inv in interventions:
            sev = inv.get("severity", "info").upper()
            print(f"  [{sev}] {inv.get('intervention_type')} -> {inv.get('recipient')}")
            print(f"    {inv.get('title')}")

    # Care plan recommendations
    care_plan = final_state.get("care_plan_adjustments", [])
    if care_plan:
        print(f"\n--- CARE PLAN RECOMMENDATIONS PENDING PHYSICIAN REVIEW ({len(care_plan)}) ---")
        for adj in care_plan:
            print(f"  [{adj.get('urgency', 'routine').upper()}] "
                  f"{adj.get('adjustment_type')}")
            print(f"    -> {adj.get('recommended_change', '')[:100]}")

    # Emergency status
    if final_state.get("emergency_triggered"):
        print("\n  !!! EMERGENCY PROTOCOLS WERE ACTIVATED THIS CYCLE !!!")
        print("  Emergency contact and EMS have been notified.")
        print("  Physician was paged STAT.")

    # Summary narrative
    print("\n--- CLINICAL SUMMARY ---")
    print(summary.get("narrative", "No narrative generated."))

    print("\n" + "=" * 72)


def save_report(final_state: dict, filename: str = None) -> str:
    """Save monitoring report to JSON."""
    if not filename:
        patient_id = final_state.get("patient", {}).get("patient_id", "unknown")
        filename = f"care_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        json.dump(final_state, f, indent=2, default=str)

    print(f"\nReport saved to: {filename}")
    return filename


def run_monitoring_cycle(patient_id: str = "P001"):
    """Execute a single chronic disease monitoring cycle for a patient."""
    print_header()

    if not care_config.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    print(get_graph_description())
    print(f"\nStarting monitoring cycle for patient: {patient_id}")
    print("-" * 72)

    graph = build_care_graph()
    initial_state = AgentState().model_dump()

    config = {
        "configurable": {
            "thread_id": f"care-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    }

    final_state = None
    try:
        for step in graph.stream(initial_state, config=config):
            for node_name, state in step.items():
                print(f"  Completed node: [{node_name}]")
                final_state = state

    except Exception as e:
        print(f"\nERROR: {e}")
        raise

    if final_state:
        print_patient_report(final_state)
        save_report(final_state)
        return final_state
    else:
        print("No final state produced.")
        return None


if __name__ == "__main__":
    # Run for Patient P001 (Margaret Chen - T2DM + HTN + CKD)
    # To run for P002 (Thomas Rivera - Heart Failure), change to "P002"
    run_monitoring_cycle(patient_id="P001")
