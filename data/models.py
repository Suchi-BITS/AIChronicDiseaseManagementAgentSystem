# data/models.py
# All models use stdlib dataclasses — no pydantic required.
# dict-based AgentState is used directly by LangGraph.

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date


# ---------------------------------------------------------------------------
# Patient profile
# ---------------------------------------------------------------------------

@dataclass
class PatientProfile:
    patient_id: str
    name: str
    age: int
    sex: str                        # "male" | "female" | "other"
    conditions: List[str]
    medications: List[Dict]         # {name, dose, frequency, route}
    care_team: Dict                 # {primary_physician, care_coordinator, specialist}
    emergency_contact: str
    enrolled_since: str             # ISO date string
    risk_stratification: str = "moderate"   # low | moderate | high | very_high
    preferences: Dict = field(default_factory=dict)

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Raw device readings — each stored as plain dicts in AgentState lists
# (avoids any serialization complexity with LangGraph's dict-based state)
# ---------------------------------------------------------------------------

def make_wearable_reading(patient_id: str, heart_rate_bpm: float, spo2_percent: float,
                           hrv_ms: float, steps: int, active_minutes: int,
                           sleep_hours: Optional[float], sleep_quality: Optional[float],
                           stress_score: float, respiratory_rate: float,
                           device_type: str = "Apple Watch Series 9") -> dict:
    return {
        "type": "wearable",
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "device_type": device_type,
        "heart_rate_bpm": round(heart_rate_bpm, 1),
        "heart_rate_variability_ms": round(hrv_ms, 1),
        "spo2_percent": round(spo2_percent, 1),
        "respiratory_rate_rpm": round(respiratory_rate, 1),
        "steps_today": steps,
        "active_minutes_today": active_minutes,
        "sleep_hours_last_night": round(sleep_hours, 1) if sleep_hours else None,
        "sleep_quality_score": round(sleep_quality, 0) if sleep_quality else None,
        "stress_score": round(stress_score, 0),
    }


def make_glucose_reading(patient_id: str, glucose_mg_dl: float, trend: str,
                          tir_percent: float,
                          device_type: str = "Dexcom G7") -> dict:
    return {
        "type": "glucose",
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "device_type": device_type,
        "glucose_mg_dl": round(glucose_mg_dl, 1),
        "trend": trend,
        "in_target_range": 70 <= glucose_mg_dl <= 180,
        "time_in_range_percent_today": round(tir_percent, 1),
    }


def make_bp_reading(patient_id: str, systolic: float, diastolic: float, pulse: float,
                     irregular: bool = False,
                     device_type: str = "Omron Evolv") -> dict:
    return {
        "type": "blood_pressure",
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "device_type": device_type,
        "systolic_mmhg": round(systolic, 0),
        "diastolic_mmhg": round(diastolic, 0),
        "pulse_bpm": round(pulse, 0),
        "irregular_heartbeat_detected": irregular,
        "arm": "left",
        "body_position": "sitting",
    }


def make_medication_record(patient_id: str, medication_name: str, scheduled_iso: str,
                             taken: bool, taken_iso: Optional[str], dose_mg: float,
                             method: str = "pill", source: str = "smart_dispenser") -> dict:
    return {
        "type": "medication",
        "patient_id": patient_id,
        "medication_name": medication_name,
        "scheduled_dose_time": scheduled_iso,
        "taken_time": taken_iso,
        "taken": taken,
        "dose_mg": dose_mg,
        "method": method,
        "source": source,
    }


def make_symptom_report(patient_id: str, symptoms: List[dict], pain_scale: int,
                          fatigue_scale: int, mood_scale: int,
                          fell_today: bool = False,
                          visited_er: bool = False,
                          free_text: Optional[str] = None,
                          method: str = "mobile_app") -> dict:
    return {
        "type": "symptom_report",
        "timestamp": datetime.now().isoformat(),
        "patient_id": patient_id,
        "reporting_method": method,
        "symptoms": symptoms,
        "pain_scale": pain_scale,
        "fatigue_scale": fatigue_scale,
        "mood_scale": mood_scale,
        "fell_today": fell_today,
        "visited_er_since_last_check": visited_er,
        "free_text": free_text,
    }


def make_lab_result(patient_id: str, lab_name: str, ordered_by: str,
                     result_date_str: str, results: dict,
                     hba1c: Optional[float] = None,
                     creatinine: Optional[float] = None,
                     egfr: Optional[float] = None,
                     ldl: Optional[float] = None,
                     potassium: Optional[float] = None,
                     sodium: Optional[float] = None,
                     bnp: Optional[float] = None) -> dict:
    return {
        "type": "lab_result",
        "result_date": result_date_str,
        "patient_id": patient_id,
        "lab_name": lab_name,
        "ordered_by": ordered_by,
        "results": results,
        "hba1c_percent": hba1c,
        "creatinine_mg_dl": creatinine,
        "egfr_ml_min": egfr,
        "ldl_mg_dl": ldl,
        "potassium_meq_l": potassium,
        "sodium_meq_l": sodium,
        "bnp_pg_ml": bnp,
    }


# ---------------------------------------------------------------------------
# AgentState — plain dict passed through LangGraph
# All list values are plain Python lists of dicts.
# ---------------------------------------------------------------------------

def make_agent_state() -> dict:
    """
    Create the initial empty AgentState dict.
    LangGraph uses dict-based state natively — no Pydantic required.
    """
    return {
        # Patient profile (dict)
        "patient": None,

        # Raw device/sensor data — lists of dicts
        "vitals_readings": [],       # from wearable (Apple Watch, Fitbit, etc.)
        "glucose_readings": [],      # from CGM (Dexcom G7, FreeStyle Libre 3)
        "bp_readings": [],           # from BP monitor (Omron, Withings)
        "medication_records": [],    # from smart dispenser / Medisafe app
        "symptom_reports": [],       # from patient app / IVR call
        "lab_results": [],           # from EHR FHIR (Epic/Cerner) / Quest / LabCorp

        # Domain analyses — strings produced by monitoring agents
        "vitals_analysis": None,
        "glucose_analysis": None,
        "bp_analysis": None,
        "medication_analysis": None,
        "symptom_analysis": None,

        # Risk scores — list of dicts {domain, score, risk_level, factors, trend}
        "risk_scores": [],

        # Intervention outputs — list of dicts
        "interventions": [],
        "care_plan_adjustments": [],

        # Final progress summary (dict)
        "progress_summary": None,

        # Control flow
        "current_agent": "supervisor",
        "iteration_count": 0,
        "emergency_triggered": False,
        "errors": [],
    }
