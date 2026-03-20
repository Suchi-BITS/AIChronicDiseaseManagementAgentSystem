# agents/executor_agent.py
# Care Executor Agent — with Human-in-the-Loop gate
#
# ══════════════════════════════════════════════════════════════════════════════
# EXECUTOR ROLE IN PLANNER-EXECUTOR PATTERN
# ══════════════════════════════════════════════════════════════════════════════
#
# The executor receives the care plan from the planner and decides what to do
# with each action:
#
#   EXECUTE IMMEDIATELY  — actions tagged hitl_required=False
#                          (patient education, routine reminders, check-ins)
#
#   QUEUE FOR APPROVAL   — actions tagged hitl_required=True
#                          (physician notifications, medication changes,
#                           care plan modifications)
#                          These go into memory.pending_approvals
#                          and wait for a human to call approve_pending_actions()
#
#   EMERGENCY BYPASS     — emergency actions skip HITL entirely
#
# In a production system the HITL queue is surfaced in a clinician dashboard.
# A nurse or physician reviews each queued action and approves or rejects it.
# Approved actions are then dispatched (SMS, EHR task, pager alert).
#
# For this MVP: approve_pending_actions() is called automatically in the
# graph after a simulated "human review" step (main.py prints the queue and
# proceeds with auto-approval of non-critical items).

import json
from datetime import datetime
from config.settings import care_config
from memory.session_memory import get_memory
from agents.base import call_llm, _demo_mode


_executed_actions: list = []
_dispatch_log:     list = []


def run_executor(state: dict) -> dict:
    """
    Executor agent: filters care plan through HITL gate and executes actions.
    """
    patient_id = state.get("patient_id", "P001")
    care_plan  = state.get("care_plan", [])
    memory     = get_memory(patient_id)

    print(f"\n[EXECUTOR] Processing {len(care_plan)} planned action(s)...")

    executed  = []
    queued    = []
    skipped   = []

    for action in care_plan:
        urgency   = action.get("urgency", "routine")
        hitl_req  = action.get("hitl_required", True)
        action_id = action.get("action_id", "ACT-?")
        a_type    = action.get("type", "unknown")

        # Emergency: always execute immediately, no HITL
        if urgency == "emergency":
            result = _dispatch_action(action, patient_id)
            executed.append(result)
            memory.record_intervention(action | {"id": action_id})
            print(f"  [EMERGENCY] EXECUTED: {action.get('title', '')[:60]}")
            continue

        # Check memory deduplication — skip if same type already issued at same or higher urgency
        if not memory.can_issue_intervention(a_type, urgency, patient_id):
            skipped.append(action_id)
            print(f"  [SKIP] Duplicate: {action_id} ({a_type}, {urgency}) already issued")
            continue

        # HITL gate
        if hitl_req:
            memory.add_pending_approval(action.copy())
            queued.append(action)
            print(f"  [QUEUED] Awaiting human approval: {action_id} — "
                  f"{action.get('title', '')[:60]}")
        else:
            result = _dispatch_action(action, patient_id)
            executed.append(result)
            memory.record_intervention(action | {"id": action_id})
            print(f"  [EXECUTED] {action_id} ({urgency}): {action.get('title', '')[:60]}")

    state["executed_actions"] = executed
    state["queued_for_approval"] = queued
    state["skipped_actions"]  = skipped
    state["current_agent"]    = "reporter"

    print(f"\n  [EXECUTOR] Executed: {len(executed)} | "
          f"Queued for approval: {len(queued)} | Skipped (duplicate): {len(skipped)}")
    return state


def approve_pending_actions(patient_id: str,
                            approved_ids: list = None,
                            approve_all_routine: bool = True) -> list:
    """
    Human-in-the-loop approval step.

    Called by the human reviewer (physician/nurse) or in MVP by main.py
    to simulate approval. Returns list of newly dispatched actions.

    Args:
        patient_id:          patient to process pending approvals for
        approved_ids:        specific action_ids to approve (or None to use approve_all_routine)
        approve_all_routine: if True, auto-approve all 'routine' urgency pending actions
    """
    memory   = get_memory(patient_id)
    pending  = memory.pending_approvals
    approved = []

    if not pending:
        return []

    print(f"\n[HITL REVIEW] {len(pending)} pending action(s) for patient {patient_id}:")
    for p in pending:
        print(f"  [{p.get('urgency','?').upper()}] {p.get('action_id','?')}: "
              f"{p.get('title','')[:60]}")

    remaining = []
    for action in pending:
        action_id = action.get("action_id", "")
        urgency   = action.get("urgency", "routine")
        should_approve = (
            (approved_ids and action_id in approved_ids) or
            (approve_all_routine and urgency == "routine")
        )
        if should_approve:
            result = _dispatch_action(action, patient_id)
            approved.append(result)
            memory.record_intervention(action | {"id": action_id,
                                                  "approved_by_human": True})
            print(f"  [APPROVED + DISPATCHED] {action_id}")
        else:
            remaining.append(action)
            print(f"  [HELD] {action_id} — requires physician sign-off")

    memory.pending_approvals = remaining
    return approved


def _dispatch_action(action: dict, patient_id: str) -> dict:
    """
    Simulate action dispatch. In production:
      - patient SMS/push → Twilio API / FCM
      - care_team email  → SendGrid / Epic InBasket
      - physician alert  → Epic MyChart / pager system
      - emergency alert  → 911 CAD system / RapidSOS API
    """
    dispatch_record = {
        "action_id":  action.get("action_id", "ACT-?"),
        "type":       action.get("type", ""),
        "urgency":    action.get("urgency", "routine"),
        "title":      action.get("title", ""),
        "recipient":  action.get("recipient", ""),
        "patient_id": patient_id,
        "dispatched_at": datetime.now().isoformat(timespec="minutes"),
        "channel":    _get_channel(action.get("recipient", ""), action.get("urgency", "routine")),
        "status":     "dispatched" if not _demo_mode() else "demo_dispatched",
    }
    _dispatch_log.append(dispatch_record)
    return dispatch_record


def _get_channel(recipient: str, urgency: str) -> str:
    if urgency == "emergency":
        return "emergency_alert_system"
    channels = {
        "patient":               "sms_and_app",
        "care_coordinator":      "app_task",
        "physician":             "epic_inbasket",
        "care_team":             "secure_message",
        "emergency_services_and_care_team": "emergency_alert_system",
    }
    return channels.get(recipient, "app_notification")
