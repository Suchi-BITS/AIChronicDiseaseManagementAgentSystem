# config/settings.py
import os
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class CareSystemConfig:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    temperature: float = 0.05

    # System identity
    system_name: str = "CareCompanion AI"
    organization: str = "Integrated Health Network"

    # Conditions supported
    supported_conditions: List[str] = field(default_factory=lambda: [
        "type2_diabetes", "hypertension", "heart_failure",
        "copd", "chronic_kidney_disease"
    ])

    # --- Clinical thresholds ---
    # Glucose (mg/dL)
    glucose_critical_low: float  = 54.0
    glucose_low: float           = 70.0
    glucose_target_high: float   = 180.0
    glucose_high: float          = 250.0
    glucose_critical_high: float = 350.0

    # Blood pressure (mmHg)
    bp_systolic_target_high: float   = 130.0
    bp_systolic_high: float          = 140.0
    bp_systolic_critical_high: float = 180.0
    bp_diastolic_critical_high: float = 120.0

    # Heart rate (bpm)
    hr_critical_low: float  = 45.0
    hr_high: float          = 100.0
    hr_critical_high: float = 130.0

    # SpO2
    spo2_critical_low: float = 88.0
    spo2_low: float          = 92.0

    # HbA1c
    hba1c_target: float   = 7.0
    hba1c_high: float     = 8.0
    hba1c_critical: float = 10.0

    # Adherence
    adherence_target_percent: float   = 80.0
    adherence_low_percent: float      = 60.0
    adherence_critical_percent: float = 40.0

    # Horizon
    planning_horizon_days: int = 7

    disclaimer: str = (
        "CLINICAL DECISION SUPPORT ONLY. All recommendations require review by a "
        "licensed healthcare provider. In an emergency, call 911 immediately."
    )


care_config = CareSystemConfig()
