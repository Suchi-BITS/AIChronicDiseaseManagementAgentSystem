# graph/care_graph.py
# Chronic Disease Management — Planner-Executor-Reporter LangGraph
#
# Topology:
#   START -> planner -> executor -> human_review -> reporter -> END
#
# human_review node: surfaces HITL queue and calls approve_pending_actions().
# In MVP: auto-approves routine items; holds urgent for console confirmation.
# In production: human_review sends to a clinician dashboard and waits.

from typing import TypedDict, Any, Optional, List
from langgraph.graph import StateGraph, END

from agents.planner_agent  import run_planner
from agents.executor_agent import run_executor, approve_pending_actions
from agents.reporter_agent import run_reporter


class CareState(TypedDict, total=False):
    patient_id:           str
    risk_level:           str
    current_agent:        str
    emergency_triggered:  bool
    # Data
    vitals_readings:      Any
    glucose_readings:     Any
    bp_readings:          Any
    medication_records:   Any
    symptom_reports:      Any
    lab_results:          Any
    signals:              Any
    # Plan and execution
    care_plan:            Any
    executed_actions:     List[Any]
    queued_for_approval:  List[Any]
    skipped_actions:      List[Any]
    # Output
    progress_summary:     Optional[str]


def human_review_node(state: CareState) -> CareState:
    """
    HITL gate node. In MVP: auto-approves routine; holds urgent items.
    In production: this node would pause execution and surface to a dashboard.
    """
    patient_id = state.get("patient_id", "P001")
    queued     = state.get("queued_for_approval", [])

    if not queued:
        return state

    print(f"\n[HUMAN REVIEW] {len(queued)} action(s) awaiting review...")

    # MVP simulation: approve all routine items automatically
    # Urgent items are printed and auto-approved for demo purposes only
    # In production: await clinician sign-off via EHR portal
    newly_approved = approve_pending_actions(
        patient_id          = patient_id,
        approve_all_routine = True,
    )

    # Merge newly approved into executed_actions
    executed = list(state.get("executed_actions") or [])
    executed.extend(newly_approved)
    state["executed_actions"] = executed

    return state


def _should_skip_hitl(state: CareState) -> str:
    # Emergency cases: skip straight to reporter
    if state.get("emergency_triggered"):
        return "reporter"
    return "human_review"


def build_care_graph(patient_id: str = "P001"):
    g = StateGraph(CareState)

    g.add_node("planner",      run_planner)
    g.add_node("executor",     run_executor)
    g.add_node("human_review", human_review_node)
    g.add_node("reporter",     run_reporter)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        _should_skip_hitl,
        {"human_review": "human_review", "reporter": "reporter"},
    )
    g.add_edge("human_review", "reporter")
    g.add_edge("reporter", END)

    return g.compile()
