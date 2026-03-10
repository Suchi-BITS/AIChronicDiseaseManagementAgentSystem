# graph/care_graph.py

from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents.all_agents import (
    run_supervisor, run_vitals_agent, run_glucose_agent,
    run_bp_agent, run_medication_agent, run_symptom_lab_agent,
    run_intervention_agent,
)


# LangGraph requires a TypedDict schema for state annotation.
# We use a catch-all "Any" for all values since the state is a rich dict.
class CareState(TypedDict, total=False):
    patient:              Any
    target_patient_id:    str
    vitals_readings:      Any
    glucose_readings:     Any
    bp_readings:          Any
    medication_records:   Any
    symptom_reports:      Any
    lab_results:          Any
    vitals_analysis:      Any
    glucose_analysis:     Any
    bp_analysis:          Any
    medication_analysis:  Any
    symptom_analysis:     Any
    risk_scores:          Any
    interventions:        Any
    care_plan_adjustments: Any
    progress_summary:     Any
    current_agent:        str
    iteration_count:      int
    emergency_triggered:  bool
    errors:               Any


def build_care_graph(patient_id: str = "P001"):
    """
    Build and compile the LangGraph StateGraph.

    Flow:
      supervisor (init)
        -> vitals_agent
        -> glucose_agent
        -> bp_agent
        -> medication_agent
        -> symptom_lab_agent
        -> intervention_agent
        -> supervisor (synthesis)
        -> END
    """
    workflow = StateGraph(CareState)

    workflow.add_node("supervisor",       run_supervisor)
    workflow.add_node("vitals_agent",     run_vitals_agent)
    workflow.add_node("glucose_agent",    run_glucose_agent)
    workflow.add_node("bp_agent",         run_bp_agent)
    workflow.add_node("medication_agent", run_medication_agent)
    workflow.add_node("symptom_lab_agent",run_symptom_lab_agent)
    workflow.add_node("intervention_agent", run_intervention_agent)

    workflow.set_entry_point("supervisor")

    # Supervisor routes: first call -> vitals, second call (after pipeline) -> END
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("current_agent", "complete"),
        {
            "vitals_agent": "vitals_agent",
            "complete":     END,
        }
    )

    # Linear monitoring pipeline
    workflow.add_edge("vitals_agent",      "glucose_agent")
    workflow.add_edge("glucose_agent",     "bp_agent")
    workflow.add_edge("bp_agent",          "medication_agent")
    workflow.add_edge("medication_agent",  "symptom_lab_agent")
    workflow.add_edge("symptom_lab_agent", "intervention_agent")
    workflow.add_edge("intervention_agent","supervisor")

    memory   = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)
    return compiled
