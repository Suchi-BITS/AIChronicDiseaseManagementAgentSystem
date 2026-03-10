# data/simulation.py
#
# Provides deterministic-ish simulated patient data for two demo patients.
# Uses Python random with a seed so results are reproducible.
#
# In production these functions are REPLACED by real API calls:
#   Wearables   -> Apple HealthKit / Google Health Connect / Fitbit Web API
#   Glucose     -> Dexcom Share API / Abbott LibreLink API
#   BP          -> Omron Connect API / Withings Health API
#   Medication  -> Medisafe API / Hero Health API / SureScripts
#   Symptoms    -> Epic MyChart App / Twilio IVR
#   Labs        -> HL7 FHIR R4 (Epic/Cerner) / Quest Diagnostics API

import random
from datetime import datetime, date, timedelta
from data.models import (
    make_wearable_reading, make_glucose_reading, make_bp_reading,
    make_medication_record, make_symptom_report, make_lab_result
)


# -------------------------------------------------------------------------
# Patient profiles (static — in production from EHR FHIR Patient resource)
# -------------------------------------------------------------------------

PATIENTS = {
    "P001": {
        "patient_id": "P001",
        "name": "Margaret Chen",
        "age": 67,
        "sex": "female",
        "conditions": ["type2_diabetes", "hypertension", "chronic_kidney_disease"],
        "medications": [
            {"name": "Metformin",     "dose": "1000mg", "frequency": "twice_daily",  "route": "oral"},
            {"name": "Lisinopril",    "dose": "10mg",   "frequency": "once_daily",   "route": "oral"},
            {"name": "Amlodipine",    "dose": "5mg",    "frequency": "once_daily",   "route": "oral"},
            {"name": "Atorvastatin",  "dose": "40mg",   "frequency": "once_daily",   "route": "oral"},
            {"name": "Aspirin",       "dose": "81mg",   "frequency": "once_daily",   "route": "oral"},
        ],
        "care_team": {
            "primary_physician":  "Dr. Sarah Williams, MD",
            "care_coordinator":   "Nurse James Park, RN",
            "specialist":         "Dr. Robert Kumar, Endocrinology",
        },
        "emergency_contact": "David Chen (Son) — 555-0198",
        "enrolled_since": (date.today() - timedelta(days=500)).isoformat(),
        "risk_stratification": "high",
        "preferences": {"notifications": "sms_and_app", "language": "english"},
    },
    "P002": {
        "patient_id": "P002",
        "name": "Thomas Rivera",
        "age": 58,
        "sex": "male",
        "conditions": ["heart_failure", "hypertension", "type2_diabetes"],
        "medications": [
            {"name": "Carvedilol",     "dose": "25mg",  "frequency": "twice_daily", "route": "oral"},
            {"name": "Furosemide",     "dose": "40mg",  "frequency": "once_daily",  "route": "oral"},
            {"name": "Lisinopril",     "dose": "20mg",  "frequency": "once_daily",  "route": "oral"},
            {"name": "Spironolactone", "dose": "25mg",  "frequency": "once_daily",  "route": "oral"},
            {"name": "Empagliflozin",  "dose": "10mg",  "frequency": "once_daily",  "route": "oral"},
        ],
        "care_team": {
            "primary_physician":  "Dr. Michael Torres, MD",
            "care_coordinator":   "Nurse Lisa Grant, RN",
            "specialist":         "Dr. Anna Fischer, Cardiology",
        },
        "emergency_contact": "Maria Rivera (Wife) — 555-0247",
        "enrolled_since": (date.today() - timedelta(days=730)).isoformat(),
        "risk_stratification": "very_high",
        "preferences": {"notifications": "app_and_email", "language": "english"},
    },
}


def _rng(patient_id: str) -> random.Random:
    """Per-patient seeded RNG so data is reproducible across runs."""
    seed = int(date.today().strftime("%Y%m%d")) + hash(patient_id) % 10000
    return random.Random(seed)


# -------------------------------------------------------------------------
# Wearable vitals  (source: Apple HealthKit / Google Health Connect)
# -------------------------------------------------------------------------

def get_wearable_vitals(patient_id: str, num_readings: int = 12) -> list:
    rng = _rng(patient_id)

    # Patient-specific baselines reflecting their condition
    if patient_id == "P002":          # Heart failure — higher resting HR, lower SpO2
        base_hr    = 84
        base_spo2  = 93.5
        base_sleep = 5.8
        device     = "Fitbit Sense 2"
    else:                             # T2DM + HTN + CKD
        base_hr    = 76
        base_spo2  = 96.2
        base_sleep = 6.5
        device     = "Apple Watch Series 9"

    readings = []
    for i in range(num_readings):
        hr      = base_hr   + rng.gauss(0, 5)
        spo2    = base_spo2 + rng.gauss(0, 1.2)
        spo2    = min(100, max(80, spo2))
        hrv     = rng.uniform(18, 48)
        steps   = int(rng.gauss(3800, 1200)) if i == 0 else int(rng.gauss(350, 120))
        active  = int(steps / 100)
        stress  = rng.uniform(28, 72)
        rr      = rng.uniform(13, 19)
        sleep   = base_sleep + rng.gauss(0, 0.8) if i == 0 else None
        quality = rng.uniform(45, 78) if i == 0 else None

        readings.append(make_wearable_reading(
            patient_id=patient_id,
            heart_rate_bpm=hr,
            spo2_percent=spo2,
            hrv_ms=hrv,
            steps=steps,
            active_minutes=active,
            sleep_hours=sleep,
            sleep_quality=quality,
            stress_score=stress,
            respiratory_rate=rr,
            device_type=device,
        ))
    return readings


# -------------------------------------------------------------------------
# CGM glucose  (source: Dexcom Share API / Abbott LibreLink API)
# -------------------------------------------------------------------------

def get_glucose_readings(patient_id: str, num_readings: int = 20) -> list:
    has_diabetes = "type2_diabetes" in PATIENTS.get(patient_id, {}).get("conditions", [])
    if not has_diabetes:
        return []

    rng = _rng(patient_id)

    # Margaret (P001): poorly controlled, frequent post-meal spikes
    # Thomas  (P002): slightly better controlled but elevated baseline
    base     = 155 if patient_id == "P001" else 145
    tir_base = 52  if patient_id == "P001" else 61

    trends = ["rising_rapidly", "rising", "stable", "falling", "falling_rapidly"]
    trend_weights = [8, 22, 45, 20, 5]

    readings = []
    for i in range(num_readings):
        hour = (datetime.now().hour - i // 2) % 24
        # Post-meal spike simulation
        if hour in [7, 8, 12, 13, 18, 19]:
            adj = rng.uniform(30, 80)
        elif hour in [2, 3, 4]:
            adj = rng.uniform(-20, 10)   # fasting / overnight
        else:
            adj = rng.gauss(0, 18)

        glucose = max(45, min(390, base + adj))
        trend   = rng.choices(trends, weights=trend_weights)[0]
        # Adjust trend to be more realistic: if glucose high, more likely rising/stable
        if glucose > 200:
            trend = rng.choices(["stable", "rising", "falling"], weights=[40, 35, 25])[0]
        tir = max(20, min(95, tir_base + rng.gauss(0, 6)))

        readings.append(make_glucose_reading(
            patient_id=patient_id,
            glucose_mg_dl=glucose,
            trend=trend,
            tir_percent=tir,
        ))
    return readings


# -------------------------------------------------------------------------
# Blood pressure  (source: Omron Connect API / Withings Health API)
# -------------------------------------------------------------------------

def get_bp_readings(patient_id: str, num_readings: int = 14) -> list:
    rng = _rng(patient_id)

    if patient_id == "P001":    # Hypertensive with CKD — poorly controlled
        base_sys = 148
        base_dia = 92
    else:                       # Heart failure — variable, sometimes low
        base_sys = 138
        base_dia = 85

    readings = []
    for i in range(num_readings):
        # Morning surge on even readings
        surge = rng.uniform(8, 18) if i % 2 == 0 else 0
        sys   = base_sys + surge + rng.gauss(0, 10)
        dia   = base_dia + rng.gauss(0, 6)
        pulse = rng.uniform(58, 92)
        irreg = rng.random() < 0.07

        readings.append(make_bp_reading(
            patient_id=patient_id,
            systolic=sys,
            diastolic=dia,
            pulse=pulse,
            irregular=irreg,
        ))
    return readings


# -------------------------------------------------------------------------
# Medication adherence  (source: Medisafe / Hero Health smart dispenser)
# -------------------------------------------------------------------------

def get_medication_records(patient_id: str, days_back: int = 14) -> list:
    rng   = _rng(patient_id)
    meds  = PATIENTS[patient_id]["medications"]

    # Realistic per-medication adherence rates (based on published data)
    adherence_rates = {}
    for med in meds:
        name = med["name"]
        # Statins and aspirin often skipped; critical heart meds better adherence
        if name in ["Atorvastatin", "Aspirin"]:
            adherence_rates[name] = rng.uniform(0.58, 0.72)
        elif name in ["Furosemide", "Carvedilol"]:
            adherence_rates[name] = rng.uniform(0.64, 0.80)
        else:
            adherence_rates[name] = rng.uniform(0.70, 0.92)

    records = []
    for day_offset in range(days_back):
        day = datetime.now() - timedelta(days=day_offset)
        for med in meds:
            freq      = med.get("frequency", "once_daily")
            doses_day = 2 if freq == "twice_daily" else 1
            dose_mg   = float(med["dose"].replace("mg", "").strip())

            for dose_num in range(doses_day):
                hour      = 8 if dose_num == 0 else 20
                scheduled = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                taken     = rng.random() < adherence_rates[med["name"]]
                taken_iso = None
                if taken:
                    delay = rng.gauss(0, 35)
                    taken_iso = (scheduled + timedelta(minutes=delay)).isoformat()

                records.append(make_medication_record(
                    patient_id=patient_id,
                    medication_name=med["name"],
                    scheduled_iso=scheduled.isoformat(),
                    taken=taken,
                    taken_iso=taken_iso,
                    dose_mg=dose_mg,
                ))
    return records


# -------------------------------------------------------------------------
# Symptom reports  (source: Patient mobile app / Epic MyChart / IVR calls)
# -------------------------------------------------------------------------

def get_symptom_reports(patient_id: str, days_back: int = 7) -> list:
    rng = _rng(patient_id)

    symptom_pool = {
        "P001": [
            ("headache", 5),
            ("dizziness", 4),
            ("blurry_vision", 3),
            ("increased_thirst", 6),
            ("frequent_urination", 7),
            ("fatigue", 6),
            ("tingling_feet", 5),
            ("swollen_ankles", 4),
        ],
        "P002": [
            ("shortness_of_breath", 7),
            ("leg_swelling", 6),
            ("fatigue", 8),
            ("chest_tightness", 5),
            ("palpitations", 4),
            ("dizziness", 5),
            ("reduced_exercise_tolerance", 7),
            ("nocturnal_dyspnea", 6),
        ],
    }
    pool       = symptom_pool.get(patient_id, symptom_pool["P001"])
    free_texts = [
        None,
        "Felt unusually tired after morning walk",
        "Missed my evening pill — forgot after dinner",
        "BP seemed high this morning, felt flushed",
        "Legs felt heavier than usual today",
        "Had trouble sleeping, woke up twice",
        None, None,
    ]

    reports = []
    for day_offset in range(days_back):
        ts = datetime.now() - timedelta(days=day_offset, hours=rng.randint(7, 10))
        num_syms = rng.randint(1, 4)
        chosen   = rng.sample(pool, k=min(num_syms, len(pool)))
        symptoms = []
        for sym_name, base_sev in chosen:
            sev = max(1, min(10, base_sev + rng.randint(-2, 3)))
            symptoms.append({
                "symptom_name":       sym_name,
                "severity_1_to_10":   sev,
                "duration_hours":     rng.randint(1, 36),
                "notes": "Worse after exertion" if sev >= 7 and rng.random() > 0.6 else None,
            })

        reports.append(make_symptom_report(
            patient_id=patient_id,
            symptoms=symptoms,
            pain_scale=rng.randint(1, 6),
            fatigue_scale=rng.randint(3, 8),
            mood_scale=rng.randint(4, 8),
            fell_today=rng.random() < 0.04,
            visited_er=rng.random() < 0.03,
            free_text=rng.choice(free_texts),
        ))
    return reports


# -------------------------------------------------------------------------
# Lab results  (source: HL7 FHIR R4 from Epic/Cerner + Quest/LabCorp APIs)
# -------------------------------------------------------------------------

def get_lab_results(patient_id: str) -> list:
    rng         = _rng(patient_id)
    result_date = (date.today() - timedelta(days=rng.randint(12, 35))).isoformat()

    if patient_id == "P001":
        hba1c       = round(rng.uniform(8.1, 9.4), 1)     # Poorly controlled
        creatinine  = round(rng.uniform(1.5, 2.0), 2)
        egfr        = round(rng.uniform(38, 52), 0)        # CKD stage 3b
        ldl         = round(rng.uniform(95, 128), 0)
        potassium   = round(rng.uniform(4.6, 5.3), 1)      # Borderline high on lisinopril
        sodium      = round(rng.uniform(136, 141), 0)
        bnp         = None

        results = {
            "HbA1c": {
                "value": hba1c, "unit": "%",
                "reference_range": "< 7.0% (diabetic target)", "flag": "H"
            },
            "eGFR": {
                "value": egfr, "unit": "mL/min/1.73m2",
                "reference_range": "> 60", "flag": "L"
            },
            "Creatinine": {
                "value": creatinine, "unit": "mg/dL",
                "reference_range": "0.57–1.00 (female)", "flag": "H"
            },
            "LDL Cholesterol": {
                "value": ldl, "unit": "mg/dL",
                "reference_range": "< 70 mg/dL (high CV risk)", "flag": "H"
            },
            "Potassium": {
                "value": potassium, "unit": "mEq/L",
                "reference_range": "3.5–5.0", "flag": "H" if potassium > 5.0 else "N"
            },
            "Urine Microalbumin/Creatinine Ratio": {
                "value": round(rng.uniform(65, 160), 0), "unit": "mg/g",
                "reference_range": "< 30 (normal)", "flag": "H"
            },
        }
        return [make_lab_result(
            patient_id=patient_id,
            lab_name="Quest Diagnostics",
            ordered_by="Dr. Sarah Williams, MD",
            result_date_str=result_date,
            results=results,
            hba1c=hba1c, creatinine=creatinine, egfr=egfr,
            ldl=ldl, potassium=potassium, sodium=sodium,
        )]

    elif patient_id == "P002":
        bnp        = round(rng.uniform(320, 820), 0)   # Elevated — heart failure marker
        hba1c      = round(rng.uniform(7.6, 8.8), 1)
        creatinine = round(rng.uniform(1.3, 1.8), 2)
        egfr       = round(rng.uniform(42, 60), 0)
        potassium  = round(rng.uniform(3.9, 5.0), 1)
        sodium     = round(rng.uniform(132, 139), 0)  # Mild hyponatremia common in HF

        results = {
            "BNP (B-Natriuretic Peptide)": {
                "value": bnp, "unit": "pg/mL",
                "reference_range": "< 100 pg/mL", "flag": "HH"
            },
            "HbA1c": {
                "value": hba1c, "unit": "%",
                "reference_range": "< 7.0% (diabetic target)", "flag": "H"
            },
            "Sodium": {
                "value": sodium, "unit": "mEq/L",
                "reference_range": "136–145", "flag": "L" if sodium < 136 else "N"
            },
            "Potassium": {
                "value": potassium, "unit": "mEq/L",
                "reference_range": "3.5–5.0", "flag": "N"
            },
            "Creatinine": {
                "value": creatinine, "unit": "mg/dL",
                "reference_range": "0.74–1.35 (male)", "flag": "H"
            },
            "eGFR": {
                "value": egfr, "unit": "mL/min/1.73m2",
                "reference_range": "> 60", "flag": "L" if egfr < 60 else "N"
            },
        }
        return [make_lab_result(
            patient_id=patient_id,
            lab_name="LabCorp",
            ordered_by="Dr. Anna Fischer, Cardiology",
            result_date_str=result_date,
            results=results,
            hba1c=hba1c, creatinine=creatinine, egfr=egfr,
            potassium=potassium, sodium=sodium, bnp=bnp,
        )]

    return []
