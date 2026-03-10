# graph/care_graph.py
# LangGraph StateGraph for the Chronic Disease Management Agent System

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data.models import AgentState
from agents.supervisor_agent import run_supervisor_agent
from agents.vitals_agent import run_vitals_agent
from agents.glucose_agent import run_glucose_agent
from agents.bp_agent import run_bp_agent
from agents.medication_agent import run_medication_agent
from agents.symptom_lab_agent import run_symptom_lab_agent
from agents.intervention_agent import run_intervention_agent


def build_care_graph() -> StateGraph:
    """
    Build and compile the LangGraph StateGraph for chronic disease management.

    Graph topology:

    supervisor (init: load patient, route to vitals)
         |
         v
    vitals_agent (wearable HR, SpO2, sleep, activity)
         |
         v
    glucose_agent (CGM glucose readings, TIR, trends)
         |
         v
    bp_agent (home BP monitor readings, 7-day trend)
         |
         v
    medication_agent (14-day adherence records, patterns)
         |
         v
    symptom_lab_agent (self-reported symptoms + EHR labs)
         |
         v
    intervention_agent (execute targeted interventions via tools)
         |
         v
    supervisor (final synthesis, EHR note, progress summary)
         |
         v
        END

    Each monitoring agent enriches the shared AgentState with:
    - Raw data (vitals_readings, glucose_readings, etc.)
    - Domain analysis string (vitals_analysis, glucose_analysis, etc.)
    - ClinicalRiskScore for their domain

    The intervention agent reads ALL domain analyses to make
    evidence-based, prioritized intervention decisions.
    """
    workflow = StateGraph(dict)

    # Wrap Pydantic-based agents for LangGraph dict-based state
    def supervisor_node(state: dict) -> dict:
        result = run_supervisor_agent(AgentState(**state))
        return result.model_dump()

    def vitals_node(state: dict) -> dict:
        result = run_vitals_agent(AgentState(**state))
        return result.model_dump()

    def glucose_node(state: dict) -> dict:
        result = run_glucose_agent(AgentState(**state))
        return result.model_dump()

    def bp_node(state: dict) -> dict:
        result = run_bp_agent(AgentState(**state))
        return result.model_dump()

    def medication_node(state: dict) -> dict:
        result = run_medication_agent(AgentState(**state))
        return result.model_dump()

    def symptom_lab_node(state: dict) -> dict:
        result = run_symptom_lab_agent(AgentState(**state))
        return result.model_dump()

    def intervention_node(state: dict) -> dict:
        result = run_intervention_agent(AgentState(**state))
        return result.model_dump()

    # Register all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("vitals_agent", vitals_node)
    workflow.add_node("glucose_agent", glucose_node)
    workflow.add_node("bp_agent", bp_node)
    workflow.add_node("medication_agent", medication_node)
    workflow.add_node("symptom_lab_agent", symptom_lab_node)
    workflow.add_node("intervention_agent", intervention_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor conditional routing: init -> vitals_agent, final -> END
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["current_agent"],
        {
            "vitals_agent": "vitals_agent",
            "complete": END
        }
    )

    # Sequential monitoring pipeline
    workflow.add_edge("vitals_agent", "glucose_agent")
    workflow.add_edge("glucose_agent", "bp_agent")
    workflow.add_edge("bp_agent", "medication_agent")
    workflow.add_edge("medication_agent", "symptom_lab_agent")

    # Symptom/lab feeds intervention planner
    workflow.add_edge("symptom_lab_agent", "intervention_agent")

    # Interventions complete -> back to supervisor for synthesis
    workflow.add_edge("intervention_agent", "supervisor")

    # Compile with memory for session persistence
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    return compiled


def get_graph_description() -> str:
    """Return a formatted description of the agent graph."""
    return """
    CHRONIC DISEASE MANAGEMENT AI AGENT GRAPH
    ===========================================

    [SUPERVISOR] (init: load patient profile)
         |
         v
    [VITALS AGENT]         Apple Watch / Fitbit / Garmin
    Heart rate, SpO2,      -> wearable device APIs
    sleep, activity        -> Google Health Connect
         |
         v
    [GLUCOSE AGENT]        Dexcom G7 / FreeStyle Libre 3
    CGM readings,          -> CGM manufacturer APIs
    TIR, trend arrows      -> HealthKit glucose data
         |
         v
    [BP AGENT]             Omron Evolv / Withings BPM
    Home BP readings,      -> connected BP monitor APIs
    7-day trend, AF flag   -> Withings Health API
         |
         v
    [MEDICATION AGENT]     Medisafe / Hero Dispenser
    14-day dose history,   -> smart dispenser APIs
    patterns, barriers     -> pharmacy refill records
         |
         v
    [SYMPTOM & LAB AGENT]  Patient app + EHR FHIR
    Self-reported symptoms -> Epic/Cerner FHIR R4 API
    HbA1c, eGFR, BNP,     -> Quest/LabCorp lab APIs
    electrolytes
         |
         v
    [INTERVENTION AGENT]   Executes via tool calls:
    Evidence-based         -> Patient SMS/push (Twilio)
    targeted actions       -> EHR alerts (Epic In-Basket)
    by urgency level       -> Emergency protocols
         |
         v
    [SUPERVISOR] (synthesis: progress summary + EHR note)
         |
         v
       [END]
    """
