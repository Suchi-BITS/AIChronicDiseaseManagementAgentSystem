# config/settings.py
# Chronic Disease Management System Configuration

from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class CareSystemConfig(BaseModel):
    """Core configuration for the chronic disease management system."""

    # LLM settings
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    temperature: float = 0.05   # Very low — clinical decisions require consistency
    max_tokens: int = 2048

    # System identity
    system_name: str = "CareCompanion AI"
    organization: str = "Integrated Health Network"

    # Supported chronic conditions
    supported_conditions: list[str] = [
        "type2_diabetes",
        "hypertension",
        "heart_failure",
        "copd",
        "chronic_kidney_disease"
    ]

    # Clinical alert thresholds

    # Glucose (mg/dL)
    glucose_critical_low: float = 54.0
    glucose_low: float = 70.0
    glucose_target_low: float = 80.0
    glucose_target_high: float = 180.0
    glucose_high: float = 250.0
    glucose_critical_high: float = 350.0

    # Blood pressure (mmHg)
    bp_systolic_critical_low: float = 90.0
    bp_systolic_target_high: float = 130.0
    bp_systolic_high: float = 140.0
    bp_systolic_critical_high: float = 180.0
    bp_diastolic_critical_high: float = 120.0

    # Heart rate (bpm)
    hr_critical_low: float = 45.0
    hr_low: float = 55.0
    hr_high: float = 100.0
    hr_critical_high: float = 130.0

    # SpO2 (percent)
    spo2_critical_low: float = 88.0
    spo2_low: float = 92.0
    spo2_target: float = 95.0

    # HbA1c (percent)
    hba1c_target: float = 7.0
    hba1c_high: float = 8.0
    hba1c_critical: float = 10.0

    # Medication adherence
    adherence_target_percent: float = 80.0
    adherence_low_percent: float = 60.0
    adherence_critical_percent: float = 40.0

    # Intervention escalation levels
    # 1 = in-app nudge, 2 = care coordinator call, 3 = physician alert, 4 = emergency services
    escalation_levels: dict = {
        "info": 1,
        "warning": 2,
        "urgent": 3,
        "emergency": 4
    }

    # Monitoring intervals (seconds, for simulation context)
    vitals_interval_seconds: int = 300        # Every 5 minutes (wearable sync)
    glucose_interval_seconds: int = 300       # Every 5 minutes (CGM)
    medication_check_interval_seconds: int = 3600  # Every hour
    symptom_check_interval_seconds: int = 43200    # Twice daily

    # Planning horizon
    planning_horizon_days: int = 7

    # Safety disclaimer
    disclaimer: str = (
        "This system provides decision support only. All clinical decisions "
        "must be reviewed and approved by a licensed healthcare provider. "
        "In an emergency, call 911 or your local emergency number immediately."
    )


care_config = CareSystemConfig()
