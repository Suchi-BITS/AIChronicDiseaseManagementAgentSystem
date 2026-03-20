# config/settings.py
# AI Chronic Disease Management System v2
# Architecture: Planner-Executor + Human-in-the-Loop + In-Memory Short-Term Memory

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class CareConfig:
    openai_api_key: str  = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name:     str  = "gpt-4o"
    temperature:    float = 0.05

    system_name:  str = "CareCompanion AI"
    organization: str = "Integrated Health Network"

    supported_conditions: List[str] = field(default_factory=lambda: [
        "type2_diabetes", "hypertension", "heart_failure",
        "copd", "chronic_kidney_disease",
    ])

    # Clinical thresholds
    glucose_critical_low:    float = 54.0
    glucose_low:             float = 70.0
    glucose_target_high:     float = 180.0
    glucose_critical_high:   float = 350.0
    bp_systolic_high:        float = 140.0
    bp_systolic_critical:    float = 180.0
    hr_critical_low:         float = 45.0
    hr_critical_high:        float = 130.0
    spo2_critical:           float = 88.0
    adherence_low_pct:       float = 60.0

    planning_horizon_days: int = 7

    # Human-in-the-loop: actions above this urgency level require approval
    hitl_approval_threshold: str = "urgent"   # 'routine' | 'urgent' | 'emergency'

    disclaimer: str = (
        "CLINICAL DECISION SUPPORT ONLY. All recommendations require review "
        "by a licensed healthcare provider. In an emergency, call 911."
    )


care_config = CareConfig()
