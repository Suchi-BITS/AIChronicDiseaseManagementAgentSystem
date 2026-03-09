# tools/intervention_tools.py
# Tools for executing health interventions and care plan modifications
#
# PRODUCTION INTEGRATION TARGETS:
# send_patient_notification()      -> Twilio SMS API, Firebase Cloud Messaging (push),
#                                     Epic MyChart Messaging API
# alert_care_team()                -> EHR secure messaging (Epic In-Basket),
#                                     Care management platform (Wellframe, Novu),
#                                     PagerDuty for urgent physician alerts
# trigger_emergency_response()     -> 911 dispatch (if integrated), patient emergency contact,
#                                     EMS notification systems
# update_care_plan()               -> Epic FHIR CarePlan resource (write),
#                                     Cerner PowerChart care plan module
# log_clinical_note()              -> EHR Progress Note via FHIR DocumentReference,
#                                     Automated documentation in Epic/Cerner

from datetime import datetime
from langchain_core.tools import tool

# In-memory log (production: HIPAA-compliant audit database)
_intervention_log: list[dict] = []


@tool
def send_patient_notification(
    patient_id: str,
    severity: str,
    title: str,
    message: str,
    action_prompt: str,
    channels: list[str],
    requires_acknowledgment: bool
) -> dict:
    """
    Send a notification directly to the patient via configured channels.

    Production: Twilio SMS, Firebase push notification, Epic MyChart secure message.
    All messages are logged to the EHR audit trail.

    Args:
        patient_id: Target patient
        severity: 'info', 'warning', 'urgent', or 'emergency'
        title: Short notification title (< 60 chars)
        message: Full notification body
        action_prompt: Clear call to action for the patient
        channels: List from ['push_notification', 'sms', 'app_message', 'email']
        requires_acknowledgment: Whether patient must confirm receipt

    Returns:
        Delivery confirmation per channel
    """
    record = {
        "action_id": f"NOTIFY-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_id}",
        "type": "patient_notification",
        "patient_id": patient_id,
        "severity": severity,
        "title": title,
        "message": message,
        "action_prompt": action_prompt,
        "channels": channels,
        "requires_acknowledgment": requires_acknowledgment,
        "timestamp": datetime.now().isoformat(),
        "status": "delivered"
    }
    _intervention_log.append(record)

    print(f"  [PATIENT NOTIFY - {severity.upper()}] {title}")

    return {
        "success": True,
        "action_id": record["action_id"],
        "channels_delivered": channels,
        "requires_ack": requires_acknowledgment
    }


@tool
def alert_care_team(
    patient_id: str,
    recipient_role: str,
    urgency: str,
    subject: str,
    clinical_summary: str,
    recommended_action: str,
    relevant_data_points: list[str]
) -> dict:
    """
    Send a clinical alert to a member of the patient's care team.

    Production: Epic In-Basket secure message, Cerner message center,
    PagerDuty page for urgent/emergency, care coordinator task queue.

    Args:
        patient_id: Patient the alert concerns
        recipient_role: 'care_coordinator', 'physician', or 'specialist'
        urgency: 'routine', 'soon', 'urgent', or 'stat'
        subject: Alert subject line
        clinical_summary: Summary of clinical situation
        recommended_action: What the clinician should do
        relevant_data_points: Key values that triggered this alert

    Returns:
        Alert delivery confirmation with estimated response time
    """
    record = {
        "action_id": f"CLINALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_id}",
        "type": "care_team_alert",
        "patient_id": patient_id,
        "recipient_role": recipient_role,
        "urgency": urgency,
        "subject": subject,
        "clinical_summary": clinical_summary,
        "recommended_action": recommended_action,
        "relevant_data_points": relevant_data_points,
        "timestamp": datetime.now().isoformat(),
        "status": "delivered_to_ehr_inbox"
    }
    _intervention_log.append(record)

    response_times = {
        "routine": "within 2 business days",
        "soon": "within 4 hours",
        "urgent": "within 1 hour",
        "stat": "immediately"
    }
    print(f"  [CARE TEAM ALERT - {urgency.upper()}] -> {recipient_role}: {subject}")

    return {
        "success": True,
        "action_id": record["action_id"],
        "recipient": recipient_role,
        "expected_response": response_times.get(urgency, "within 24 hours"),
        "delivery_method": "ehr_secure_message" if urgency in ["routine", "soon"] else "pager_and_ehr"
    }


@tool
def trigger_emergency_response(
    patient_id: str,
    reason: str,
    vital_signs: dict,
    patient_location: str,
    emergency_contact_name: str,
    emergency_contact_phone: str
) -> dict:
    """
    Trigger emergency response for a life-threatening clinical situation.

    Production: Integration with 911/EMS dispatch (where available),
    automated call to emergency contact, immediate physician notification,
    EHR emergency flag, hospital ED pre-notification.

    THIS TOOL SHOULD ONLY BE CALLED FOR LIFE-THREATENING CONDITIONS:
    - Critical hypoglycemia (glucose < 54 mg/dL with altered mental status)
    - Hypertensive emergency (SBP > 180 + symptoms)
    - Acute severe dyspnea (suspected acute decompensated heart failure)
    - Critical SpO2 < 88% with symptoms

    Args:
        patient_id: Patient in distress
        reason: Clinical reason for emergency activation
        vital_signs: Current critical vital values
        patient_location: Patient's known address
        emergency_contact_name: Name of emergency contact
        emergency_contact_phone: Phone number to call

    Returns:
        Emergency response activation confirmation
    """
    record = {
        "action_id": f"EMERGENCY-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_id}",
        "type": "EMERGENCY_RESPONSE",
        "patient_id": patient_id,
        "reason": reason,
        "vital_signs": vital_signs,
        "patient_location": patient_location,
        "emergency_contact_name": emergency_contact_name,
        "emergency_contact_phone": emergency_contact_phone,
        "timestamp": datetime.now().isoformat(),
        "status": "EMERGENCY_ACTIVATED"
    }
    _intervention_log.append(record)

    print(f"\n  !!! EMERGENCY RESPONSE ACTIVATED for {patient_id} !!!")
    print(f"  Reason: {reason}")
    print(f"  Actions: Emergency contact notified, physician paged STAT, EMS alert sent")

    return {
        "success": True,
        "action_id": record["action_id"],
        "actions_taken": [
            f"Emergency contact {emergency_contact_name} called at {emergency_contact_phone}",
            "Physician paged STAT via pager and secure message",
            "EMS alert transmitted with patient location and vitals",
            "EHR emergency flag set - patient chart flagged for immediate review"
        ],
        "severity": "LIFE_THREATENING"
    }


@tool
def recommend_care_plan_change(
    patient_id: str,
    change_type: str,
    current_state: str,
    recommended_change: str,
    clinical_rationale: str,
    guideline_reference: str,
    urgency: str,
    requires_physician_approval: bool
) -> dict:
    """
    Submit a care plan modification recommendation to the treating physician.

    Production: Creates a draft order or task in Epic/Cerner for physician review,
    may pre-populate medication adjustment in CPOE system for one-click approval.

    Args:
        patient_id: Patient whose care plan to modify
        change_type: Type of change - e.g., 'medication_dose_change', 'lab_order',
                     'lifestyle_goal_update', 'specialist_referral', 'new_medication_recommendation'
        current_state: Description of current care plan element
        recommended_change: Specific proposed change
        clinical_rationale: Evidence-based reason for the change
        guideline_reference: Clinical guideline supporting this recommendation
        urgency: 'routine', 'soon', or 'urgent'
        requires_physician_approval: Whether physician sign-off is needed (almost always True)

    Returns:
        Recommendation submission confirmation
    """
    record = {
        "action_id": f"CAREPLAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_id}",
        "type": "care_plan_recommendation",
        "patient_id": patient_id,
        "change_type": change_type,
        "current_state": current_state,
        "recommended_change": recommended_change,
        "clinical_rationale": clinical_rationale,
        "guideline_reference": guideline_reference,
        "urgency": urgency,
        "requires_physician_approval": requires_physician_approval,
        "timestamp": datetime.now().isoformat(),
        "status": "submitted_for_physician_review"
    }
    _intervention_log.append(record)

    print(f"  [CARE PLAN REC - {urgency.upper()}] {change_type}: {recommended_change[:80]}")

    return {
        "success": True,
        "action_id": record["action_id"],
        "status": "submitted_for_physician_review",
        "physician_task_created": True,
        "urgency": urgency
    }


@tool
def log_automated_clinical_note(
    patient_id: str,
    note_type: str,
    monitoring_period: str,
    clinical_observations: str,
    interventions_taken: str,
    plan: str
) -> dict:
    """
    Log an automated clinical note to the EHR documenting the monitoring session.

    Production: Creates a structured Progress Note via FHIR DocumentReference
    in Epic or Cerner. Note is marked as AI-generated and requires cosignature
    by the supervising care coordinator or physician.

    Args:
        patient_id: Patient whose chart to document in
        note_type: 'remote_monitoring_note', 'adherence_review', 'symptom_assessment'
        monitoring_period: Time period covered by this note
        clinical_observations: Key findings from monitoring data
        interventions_taken: Actions taken by the AI system
        plan: Follow-up plan and monitoring priorities

    Returns:
        Note filing confirmation
    """
    record = {
        "action_id": f"NOTE-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient_id}",
        "type": "clinical_note",
        "patient_id": patient_id,
        "note_type": note_type,
        "monitoring_period": monitoring_period,
        "clinical_observations": clinical_observations,
        "interventions_taken": interventions_taken,
        "plan": plan,
        "ai_generated": True,
        "requires_cosignature": True,
        "timestamp": datetime.now().isoformat()
    }
    _intervention_log.append(record)

    return {
        "success": True,
        "action_id": record["action_id"],
        "note_filed": True,
        "cosignature_required": True,
        "message": "Note filed to EHR. Pending cosignature by supervising clinician."
    }


@tool
def get_intervention_log(limit: int = 25) -> list[dict]:
    """
    Retrieve recent intervention records for audit and review.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of recent intervention records
    """
    return _intervention_log[-limit:]
