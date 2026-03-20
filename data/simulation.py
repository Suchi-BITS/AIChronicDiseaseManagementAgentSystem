# data/simulation.py
# Patient simulation for Chronic Disease Management v2.
# Date-seeded RNG for reproducible demo data.
#
# Production replacements:
#   get_vitals()        -> Apple HealthKit / Fitbit Web API / Garmin Health API
#   get_glucose()       -> Dexcom Share API / Abbott LibreLink API
#   get_bp_readings()   -> Withings Health API / Omron Connect API
#   get_medications()   -> Medisafe API / SureScripts / Epic MyChart
#   get_symptoms()      -> Patient-reported via app / Twilio IVR
#   get_labs()          -> HL7 FHIR R4 (Epic/Cerner) / Quest Diagnostics API

import random
from datetime import datetime, date, timedelta
from typing import List, Dict


PATIENTS = {
    "P001": {
        "patient_id": "P001",
        "name": "Margaret Chen",
        "age": 67,
        "sex": "female",
        "conditions": ["type2_diabetes", "hypertension", "chronic_kidney_disease"],
        "medications": [
            {"name": "Metformin",    "dose": "1000mg", "frequency": "twice_daily"},
            {"name": "Lisinopril",   "dose": "10mg",   "frequency": "once_daily"},
            {"name": "Amlodipine",   "dose": "5mg",    "frequency": "once_daily"},
            {"name": "Atorvastatin", "dose": "40mg",   "frequency": "once_daily"},
            {"name": "Aspirin",      "dose": "81mg",   "frequency": "once_daily"},
        ],
        "care_team": {
            "physician":    "Dr. Sarah Williams, MD",
            "coordinator":  "Nurse James Park, RN",
            "specialist":   "Dr. Robert Kumar, Endocrinology",
        },
        "risk_level": "high",
        "emergency_contact": "David Chen (Son) — 555-0198",
    },
    "P002": {
        "patient_id": "P002",
        "name": "Thomas Rivera",
        "age": 58,
        "sex": "male",
        "conditions": ["heart_failure", "hypertension", "type2_diabetes"],
        "medications": [
            {"name": "Carvedilol",     "dose": "25mg", "frequency": "twice_daily"},
            {"name": "Furosemide",     "dose": "40mg", "frequency": "once_daily"},
            {"name": "Lisinopril",     "dose": "20mg", "frequency": "once_daily"},
            {"name": "Spironolactone", "dose": "25mg", "frequency": "once_daily"},
        ],
        "care_team": {
            "physician":   "Dr. Maria Santos, MD",
            "coordinator": "Nurse Alex Turner, RN",
            "specialist":  "Dr. James Lee, Cardiology",
        },
        "risk_level": "high",
        "emergency_contact": "Elena Rivera (Wife) — 555-0247",
    },
}


def _rng(patient_id: str, salt: int = 0) -> random.Random:
    pid_num = sum(ord(c) for c in patient_id)
    return random.Random(int(date.today().strftime("%Y%m%d")) + pid_num + salt)


def get_vitals(patient_id: str) -> List[Dict]:
    r = _rng(patient_id, 1)
    readings = []
    for i in range(7):
        dt = datetime.now() - timedelta(days=6 - i)
        readings.append({
            "timestamp":       dt.isoformat(timespec="minutes"),
            "heart_rate_bpm":  r.randint(52, 110),
            "spo2_pct":        round(r.uniform(91, 99), 1),
            "temperature_c":   round(r.uniform(36.1, 37.8), 1),
            "weight_kg":       round(r.uniform(72, 78), 1),
        })
    return readings


def get_glucose(patient_id: str) -> List[Dict]:
    if "type2_diabetes" not in PATIENTS.get(patient_id, {}).get("conditions", []):
        return []
    r = _rng(patient_id, 2)
    readings = []
    for i in range(14):
        dt = datetime.now() - timedelta(hours=12 * (13 - i))
        readings.append({
            "timestamp":   dt.isoformat(timespec="minutes"),
            "glucose_mgdl": r.randint(68, 320),
            "context":      r.choice(["fasting", "post_meal", "bedtime", "random"]),
        })
    return readings


def get_bp_readings(patient_id: str) -> List[Dict]:
    r = _rng(patient_id, 3)
    readings = []
    for i in range(7):
        dt = datetime.now() - timedelta(days=6 - i)
        readings.append({
            "timestamp":     dt.isoformat(timespec="minutes"),
            "systolic_mmhg": r.randint(115, 185),
            "diastolic_mmhg":r.randint(70, 110),
            "pulse_bpm":     r.randint(55, 95),
        })
    return readings


def get_medications(patient_id: str) -> List[Dict]:
    patient = PATIENTS.get(patient_id, {})
    r = _rng(patient_id, 4)
    records = []
    for med in patient.get("medications", []):
        doses_expected = 7 * (2 if med["frequency"] == "twice_daily" else 1)
        doses_taken    = r.randint(int(doses_expected * 0.5), doses_expected)
        records.append({
            "medication":       med["name"],
            "dose":             med["dose"],
            "frequency":        med["frequency"],
            "doses_expected":   doses_expected,
            "doses_taken":      doses_taken,
            "adherence_pct":    round(doses_taken / doses_expected * 100, 1),
            "last_dose":        (datetime.now() - timedelta(hours=r.randint(1, 36))).isoformat(timespec="minutes"),
            "missed_reason":    r.choice([None, None, None, "forgot", "side_effects"]),
        })
    return records


def get_symptoms(patient_id: str) -> List[Dict]:
    r = _rng(patient_id, 5)
    symptom_pool = ["fatigue", "shortness_of_breath", "chest_pain", "dizziness",
                    "swelling_ankles", "blurred_vision", "headache", "nausea"]
    n = r.randint(0, 3)
    return [
        {
            "timestamp":  (datetime.now() - timedelta(days=r.randint(0, 6))).isoformat(timespec="minutes"),
            "symptom":    r.choice(symptom_pool),
            "severity":   r.choice(["mild", "mild", "moderate", "severe"]),
            "notes":      "",
        }
        for _ in range(n)
    ]


def get_labs(patient_id: str) -> Dict:
    r = _rng(patient_id, 6)
    labs = {
        "hba1c_pct":         round(r.uniform(6.5, 11.0), 1),
        "egfr_ml_min":       round(r.uniform(28, 75), 1),
        "potassium_meql":    round(r.uniform(3.3, 5.4), 2),
        "creatinine_mgdl":   round(r.uniform(0.9, 2.8), 2),
        "ldl_mgdl":          r.randint(68, 175),
        "collected_date":    (date.today() - timedelta(days=r.randint(7, 90))).isoformat(),
    }
    if "heart_failure" in PATIENTS.get(patient_id, {}).get("conditions", []):
        labs["bnp_pgml"] = r.randint(200, 1800)
        labs["sodium_meql"] = r.randint(133, 142)
    return labs
