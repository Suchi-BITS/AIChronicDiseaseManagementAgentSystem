# data/models.py
# Pydantic data models for the chronic disease management system

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """Demographic and clinical profile for a monitored patient."""
    patient_id: str
    name: str
    age: int
    sex: Literal["male", "female", "other"]
    conditions: list[str]                   # Active chronic conditions
    medications: list[dict]                 # {name, dose, frequency, route}
    care_team: dict                         # {primary_physician, care_coordinator, specialist}
    emergency_contact: str
    preferences: dict = Field(default_factory=dict)  # notification preferences
    enrolled_since: date
    risk_stratification: Literal["low", "moderate", "high", "very_high"] = "moderate"


# ---------------------------------------------------------------------------
# Raw data from device/sensor sources
# ---------------------------------------------------------------------------

class WearableVitalsReading(BaseModel):
    """Data from a connected wearable device (smartwatch, fitness tracker)."""
    timestamp: datetime = Field(default_factory=datetime.now)
    patient_id: str
    device_type: str                        # e.g., "Apple Watch Series 9"
    heart_rate_bpm: float
    heart_rate_variability_ms: Optional[float] = None
    spo2_percent: Optional[float] = None
    respiratory_rate_rpm: Optional[float] = None
    steps_today: int = 0
    active_minutes_today: int = 0
    calories_burned: Optional[float] = None
    sleep_hours_last_night: Optional[float] = None
    sleep_quality_score: Optional[float] = None   # 0-100
    stress_score: Optional[float] = None          # 0-100


class GlucoseReading(BaseModel):
    """Continuous glucose monitor (CGM) reading."""
    timestamp: datetime = Field(default_factory=datetime.now)
    patient_id: str
    device_type: str                        # e.g., "Dexcom G7"
    glucose_mg_dl: float
    trend: Literal[
        "rising_rapidly",   # > +2 mg/dL/min
        "rising",           # +1 to +2 mg/dL/min
        "stable",           # -1 to +1 mg/dL/min
        "falling",          # -1 to -2 mg/dL/min
        "falling_rapidly"   # < -2 mg/dL/min
    ]
    in_target_range: bool
    time_in_range_percent_today: float      # % of readings in 70-180 mg/dL today


class BloodPressureReading(BaseModel):
    """Connected blood pressure cuff reading."""
    timestamp: datetime = Field(default_factory=datetime.now)
    patient_id: str
    device_type: str                        # e.g., "Omron Evolv"
    systolic_mmhg: float
    diastolic_mmhg: float
    pulse_bpm: float
    irregular_heartbeat_detected: bool = False
    arm: Literal["left", "right"] = "left"
    body_position: Literal["sitting", "standing", "lying"] = "sitting"


class MedicationAdherenceRecord(BaseModel):
    """Medication adherence data from smart pill dispenser or app."""
    patient_id: str
    medication_name: str
    scheduled_dose_time: datetime
    taken_time: Optional[datetime] = None
    taken: bool
    dose_mg: float
    method: Literal["pill", "injection", "inhaler", "patch", "liquid"]
    source: Literal["smart_dispenser", "patient_app", "caregiver_log", "pharmacy_refill"]
    notes: Optional[str] = None


class SymptomReport(BaseModel):
    """Patient-reported symptom data via mobile app or IVR call."""
    timestamp: datetime = Field(default_factory=datetime.now)
    patient_id: str
    reporting_method: Literal["mobile_app", "web_portal", "phone_ivr", "nurse_call"]
    symptoms: list[dict] = Field(
        description="List of {symptom_name, severity_1_to_10, duration_hours, notes}"
    )
    pain_scale: Optional[int] = Field(None, ge=0, le=10)
    fatigue_scale: Optional[int] = Field(None, ge=0, le=10)
    mood_scale: Optional[int] = Field(None, ge=1, le=10)
    fell_today: bool = False
    visited_er_since_last_check: bool = False
    free_text: Optional[str] = None


class LabResult(BaseModel):
    """Laboratory test result from EHR / FHIR integration."""
    result_date: date
    patient_id: str
    lab_name: str
    ordered_by: str
    results: dict = Field(
        description="Dict of {test_name: {value, unit, reference_range, flag}}"
    )
    # Key markers stored explicitly for fast access
    hba1c_percent: Optional[float] = None
    creatinine_mg_dl: Optional[float] = None
    egfr_ml_min: Optional[float] = None
    cholesterol_total_mg_dl: Optional[float] = None
    ldl_mg_dl: Optional[float] = None
    hdl_mg_dl: Optional[float] = None
    potassium_meq_l: Optional[float] = None
    sodium_meq_l: Optional[float] = None
    bnp_pg_ml: Optional[float] = None     # B-type natriuretic peptide (heart failure marker)


# ---------------------------------------------------------------------------
# Analysis outputs
# ---------------------------------------------------------------------------

class ClinicalRiskScore(BaseModel):
    """Composite risk score generated by a monitoring agent."""
    patient_id: str
    domain: Literal["glycemic", "cardiovascular", "renal", "respiratory", "medication", "overall"]
    score: float = Field(ge=0.0, le=100.0)
    risk_level: Literal["low", "moderate", "high", "critical"]
    contributing_factors: list[str]
    trend: Literal["improving", "stable", "worsening"]
    timestamp: datetime = Field(default_factory=datetime.now)


class HealthIntervention(BaseModel):
    """A proactive health intervention recommended or executed by an agent."""
    intervention_id: str
    patient_id: str
    intervention_type: Literal[
        "in_app_nudge",
        "educational_content",
        "medication_reminder",
        "lifestyle_coaching",
        "care_coordinator_alert",
        "physician_alert",
        "medication_adjustment_recommendation",
        "emergency_alert",
        "appointment_recommendation",
        "lab_order_recommendation"
    ]
    severity: Literal["info", "warning", "urgent", "emergency"]
    title: str
    message: str
    action_required: str
    recipient: Literal["patient", "care_coordinator", "physician", "emergency_services"]
    delivery_channel: list[str]     # e.g., ["push_notification", "sms", "ehr_alert"]
    clinical_basis: str             # What data triggered this intervention
    evidence_based_guideline: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_acknowledgment: bool = False
    follow_up_hours: Optional[int] = None


class CarePlanAdjustment(BaseModel):
    """Recommended adjustment to a patient's care plan."""
    patient_id: str
    adjustment_type: Literal[
        "medication_dose_change",
        "new_medication_recommendation",
        "medication_discontinuation",
        "lifestyle_goal_update",
        "monitoring_frequency_change",
        "specialist_referral",
        "lab_order",
        "appointment_scheduling"
    ]
    current_state: str
    recommended_change: str
    clinical_rationale: str
    evidence_level: Literal["guideline_based", "clinical_judgment", "patient_preference"]
    urgency: Literal["routine", "soon", "urgent"] = "routine"
    requires_physician_approval: bool = True
    estimated_benefit: str


class PatientProgressSummary(BaseModel):
    """Weekly progress summary for patient and care team."""
    patient_id: str
    period_start: date
    period_end: date
    overall_status: Literal["excellent", "good", "fair", "poor", "critical"]
    glycemic_control_summary: Optional[str] = None
    bp_control_summary: Optional[str] = None
    medication_adherence_summary: str = ""
    activity_summary: str = ""
    symptom_summary: str = ""
    risk_scores: list[ClinicalRiskScore] = Field(default_factory=list)
    key_achievements: list[str] = Field(default_factory=list)
    areas_of_concern: list[str] = Field(default_factory=list)
    goals_for_next_week: list[str] = Field(default_factory=list)
    narrative: str = ""


# ---------------------------------------------------------------------------
# LangGraph shared state
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """
    Shared state flowing through the LangGraph agent graph.
    Every monitoring and intervention agent reads from and writes to this.
    """
    # Patient context
    patient: Optional[PatientProfile] = None

    # Raw device / sensor data
    vitals_readings: list[WearableVitalsReading] = Field(default_factory=list)
    glucose_readings: list[GlucoseReading] = Field(default_factory=list)
    bp_readings: list[BloodPressureReading] = Field(default_factory=list)
    medication_records: list[MedicationAdherenceRecord] = Field(default_factory=list)
    symptom_reports: list[SymptomReport] = Field(default_factory=list)
    lab_results: list[LabResult] = Field(default_factory=list)

    # Analysis from monitoring agents
    vitals_analysis: Optional[str] = None
    glucose_analysis: Optional[str] = None
    bp_analysis: Optional[str] = None
    medication_analysis: Optional[str] = None
    symptom_analysis: Optional[str] = None
    lab_analysis: Optional[str] = None

    # Risk scores computed by agents
    risk_scores: list[ClinicalRiskScore] = Field(default_factory=list)

    # Outputs from planning/intervention agents
    interventions: list[HealthIntervention] = Field(default_factory=list)
    care_plan_adjustments: list[CarePlanAdjustment] = Field(default_factory=list)

    # Final consolidated output
    progress_summary: Optional[PatientProgressSummary] = None

    # Control flow
    current_agent: str = "supervisor"
    iteration_count: int = 0
    errors: list[str] = Field(default_factory=list)
    emergency_triggered: bool = False

    class Config:
        arbitrary_types_allowed = True
