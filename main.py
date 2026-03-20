#!/usr/bin/env python3
# main.py — AI Chronic Disease Management System v2
# Architecture: Planner-Executor + Human-in-the-Loop + In-Memory Session Memory
#
# Usage:
#   python main.py                     # P001 single pass
#   python main.py P002                # P002 single pass
#   python main.py P001 --passes 3     # 3 monitoring passes (demonstrates memory)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from config.settings import care_config
from data.simulation import PATIENTS


def make_initial_state(patient_id: str) -> dict:
    return {
        "patient_id":          patient_id,
        "risk_level":          "unknown",
        "current_agent":       None,
        "emergency_triggered": False,
        "vitals_readings":     None,
        "glucose_readings":    None,
        "bp_readings":         None,
        "medication_records":  None,
        "symptom_reports":     None,
        "lab_results":         None,
        "signals":             None,
        "care_plan":           None,
        "executed_actions":    [],
        "queued_for_approval": [],
        "skipped_actions":     [],
        "progress_summary":    None,
    }


def print_report(state: dict, pass_num: int) -> None:
    from agents.base import _demo_mode
    from memory.session_memory import get_memory

    patient_id = state.get("patient_id", "P001")
    patient    = PATIENTS.get(patient_id, {})
    memory     = get_memory(patient_id)

    print("\n" + "=" * 70)
    print(f"  PATIENT MONITORING REPORT — Pass #{pass_num}")
    print(f"  {care_config.system_name} | {care_config.organization}")
    print(f"  Patient: {patient.get('name')} | ID: {patient_id}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'DEMO' if _demo_mode() else 'LIVE — GPT-4o'}")
    print("=" * 70)

    signals = state.get("signals") or {}
    print(f"\nRISK LEVEL: {state.get('risk_level','unknown').upper()}")
    print(f"EMERGENCY:  {state.get('emergency_triggered', False)}")

    alerts = signals.get("alerts", [])
    warns  = signals.get("warnings", [])
    if alerts:
        print(f"\nCLINICAL ALERTS ({len(alerts)}):")
        for a in alerts:
            print(f"  {a}")
    if warns:
        print(f"\nWARNINGS ({len(warns)}):")
        for w in warns:
            print(f"  {w}")

    executed = state.get("executed_actions") or []
    queued   = state.get("queued_for_approval") or []
    skipped  = state.get("skipped_actions") or []
    print(f"\nACTIONS:")
    print(f"  Executed   : {len(executed)}")
    print(f"  Approved (HITL) : {sum(1 for a in executed if a.get('dispatched_at'))}")
    print(f"  Queued (pending): {len(memory.pending_approvals)}")
    print(f"  Skipped (dedup) : {len(skipped)}")

    for a in executed:
        print(f"    [{a.get('urgency','?').upper():9}] "
              f"{a.get('type','?')} -> {a.get('recipient','?')} "
              f"via {a.get('channel','?')}")

    print(f"\nSESSION MEMORY:")
    for line in memory.context_summary().split("\n"):
        print(f"  {line}")

    summary = state.get("progress_summary", "")
    if summary:
        print("\nPROGRESS SUMMARY:")
        print("-" * 50)
        print(summary[:1400])

    print("\n" + "=" * 70)
    print(f"  {care_config.disclaimer}")
    print("=" * 70)


def run_pass(patient_id: str, pass_num: int) -> dict:
    try:
        from graph.care_graph import build_care_graph
        graph  = build_care_graph(patient_id)
        result = graph.invoke(make_initial_state(patient_id))
    except ImportError:
        print("\n[INFO] LangGraph not installed — running agents directly...")
        from agents.planner_agent  import run_planner
        from agents.executor_agent import run_executor, approve_pending_actions
        from agents.reporter_agent import run_reporter

        state = make_initial_state(patient_id)
        state = run_planner(state)
        state = run_executor(state)
        if not state.get("emergency_triggered"):
            approve_pending_actions(patient_id, approve_all_routine=True)
        result = run_reporter(state)

    return result


def main():
    args = sys.argv[1:]

    patient_id = "P001"
    n_passes   = 1

    for arg in args:
        if arg in PATIENTS:
            patient_id = arg
        elif arg == "--passes" and args.index(arg) + 1 < len(args):
            try:
                n_passes = int(args[args.index(arg) + 1])
            except ValueError:
                pass
        elif arg.startswith("--passes="):
            try:
                n_passes = int(arg.split("=")[1])
            except ValueError:
                pass

    patient = PATIENTS.get(patient_id, {})
    from agents.base import _demo_mode

    print("=" * 70)
    print("  AI CHRONIC DISEASE MANAGEMENT SYSTEM v2")
    print("  Architecture: Planner-Executor + HITL + In-Memory")
    print(f"  Patient: {patient.get('name')} | ID: {patient_id}")
    print(f"  Conditions: {', '.join(patient.get('conditions', []))}")
    print(f"  Mode: {'DEMO (no API key)' if _demo_mode() else 'LIVE — GPT-4o'}")
    print("=" * 70)

    for i in range(n_passes):
        if n_passes > 1:
            print(f"\n{'─' * 70}")
            print(f"  MONITORING PASS {i + 1} of {n_passes}")
            print(f"{'─' * 70}")

        result = run_pass(patient_id, i + 1)
        print_report(result, i + 1)

        if i < n_passes - 1:
            print(f"\n  [Simulating 6-hour monitoring interval...]\n")
            time.sleep(0.3)


if __name__ == "__main__":
    main()
