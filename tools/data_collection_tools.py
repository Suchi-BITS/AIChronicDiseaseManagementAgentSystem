# tools/data_collection_tools.py
# Simulated patient data collection tools
#
# DATA SOURCES (production equivalents):
# -------------------------------------------------------------------
# fetch_wearable_vitals()    -> Apple HealthKit API, Google Health Connect,
#                               Fitbit Web API, Garmin Connect API
#                               Data: HR, HRV, SpO2, steps, sleep, stress
#
# fetch_glucose_readings()   -> Dexcom Share API, Abbott LibreView API,
#                               Medtronic Guardian API (CGM feeds)
#                               Data: glucose mg/dL, trend arrows, TIR
#
# fetch_blood_pressure()     -> Omron Connect API, Withings Health API,
#                               QardioArm API, iHealth API
#                               Data: systolic, diastolic, pulse, AF detection
#
# fetch_medication_adherence()-> Medisafe API, Hero Health API,
#                               AdhereTech smart bottle API,
#                               Pharmacy refill records (e.g., SureScripts)
#                               Data: dose times, taken/missed, refill dates
#
# fetch_symptom_reports()    -> Patient app (custom or Epic MyChart),
#                               Automated IVR phone calls,
#                               Care coordinator nurse call logs
#                               Data: symptom severity, free-text, falls, ER visits
#
# fetch_lab_results()        -> HL7 FHIR R4 API (Epic, Cerner, Athena),
#                               Quest Diagnostics API, LabCorp API
#                               Data: HbA1c, creatinine, eGFR, lipids, BNP, electrolytes
#
# get_patient_profile()      -> EHR FHIR Patient resource,
#                               Care management platform (Wellframe, Novu)
#                               Data: demographics, conditions, medications, care team
# -------------------------------------------------------------------

import random
from datetime import datetime, timedelta, date
from langchain_core.tools import tool
from data.models import (
    PatientProfile, WearableVitalsReading, GlucoseReading,
    BloodPressureReading, MedicationAdherenceRecord,
    SymptomReport, LabResult
)


@tool
def get_patient_profile(patient_id: str) -> dict:
    """
    Retrieve patient demographic and clinical profile from the EHR system.

    Production source: HL7 FHIR R4 Patient + Condition + MedicationStatement resources.
    Connects to: Epic FHIR API, Cerner Millennium FHIR API, or care management platform.

    Args:
        patient_id: Unique patient identifier

    Returns:
        Patient profile including conditions, medications, and care team
    """
    profiles = {
        "P001": PatientProfile(
            patient_id="P001",
            name="Margaret Chen",
            age=67,
            sex="female",
            conditions=["type2_diabetes", "hypertension", "chronic_kidney_disease"],
            medications=[
                {"name": "Metformin", "dose": "1000mg", "frequency": "twice_daily", "route": "oral"},
                {"name": "Lisinopril", "dose": "10mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Amlodipine", "dose": "5mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Atorvastatin", "dose": "40mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Aspirin", "dose": "81mg", "frequency": "once_daily", "route": "oral"},
            ],
            care_team={
                "primary_physician": "Dr. Sarah Williams, MD",
                "care_coordinator": "Nurse James Park, RN",
                "specialist": "Dr. Robert Kumar, Endocrinology"
            },
            emergency_contact="David Chen (Son) - 555-0198",
            preferences={"notifications": "sms_and_app", "language": "english"},
            enrolled_since=date(2022, 3, 15),
            risk_stratification="high"
        ).model_dump(mode="json"),

        "P002": PatientProfile(
            patient_id="P002",
            name="Thomas Rivera",
            age=58,
            sex="male",
            conditions=["heart_failure", "hypertension", "type2_diabetes"],
            medications=[
                {"name": "Carvedilol", "dose": "25mg", "frequency": "twice_daily", "route": "oral"},
                {"name": "Furosemide", "dose": "40mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Lisinopril", "dose": "20mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Spironolactone", "dose": "25mg", "frequency": "once_daily", "route": "oral"},
                {"name": "Empagliflozin", "dose": "10mg", "frequency": "once_daily", "route": "oral"},
            ],
            care_team={
                "primary_physician": "Dr. Michael Torres, MD",
                "care_coordinator": "Nurse Lisa Grant, RN",
                "specialist": "Dr. Anna Fischer, Cardiology"
            },
            emergency_contact="Maria Rivera (Wife) - 555-0247",
            preferences={"notifications": "app_and_email", "language": "english"},
            enrolled_since=date(2021, 9, 8),
            risk_stratification="very_high"
        ).model_dump(mode="json")
    }

    return profiles.get(patient_id, profiles["P001"])


@tool
def fetch_wearable_vitals(patient_id: str, hours_back: int = 24) -> list[dict]:
    """
    Fetch recent vitals from the patient's connected wearable device.

    Production source: Apple HealthKit API (iOS), Google Health Connect (Android),
    Fitbit Web API, Garmin Connect API, Samsung Health API.
    Sync frequency: Every 5 minutes for HR/SpO2, hourly for aggregated metrics.

    Args:
        patient_id: Patient identifier
        hours_back: How many hours of history to retrieve

    Returns:
        List of wearable vitals readings (most recent first)
    """
    readings = []
    num_readings = min(hours_back * 2, 48)  # Up to 2 per hour

    # Simulate different health states per patient
    if patient_id == "P002":   # Heart failure patient - more variable
        base_hr = random.uniform(72, 95)
        base_spo2 = random.uniform(91, 96)
        base_sleep = random.uniform(5.0, 7.5)
    else:
        base_hr = random.uniform(65, 85)
        base_spo2 = random.uniform(94, 99)
        base_sleep = random.uniform(6.0, 8.5)

    for i in range(num_readings):
        ts = datetime.now() - timedelta(hours=i * (hours_back / num_readings))
        hr_variation = random.uniform(-8, 12)
        spo2_variation = random.uniform(-2, 1)

        reading = WearableVitalsReading(
            timestamp=ts,
            patient_id=patient_id,
            device_type="Apple Watch Series 9" if patient_id == "P001" else "Fitbit Sense 2",
            heart_rate_bpm=round(base_hr + hr_variation, 1),
            heart_rate_variability_ms=round(random.uniform(18, 55), 1),
            spo2_percent=round(min(100, base_spo2 + spo2_variation), 1),
            respiratory_rate_rpm=round(random.uniform(12, 20), 1),
            steps_today=random.randint(800, 7500),
            active_minutes_today=random.randint(5, 45),
            calories_burned=round(random.uniform(1400, 2200), 0),
            sleep_hours_last_night=round(base_sleep + random.uniform(-1, 1), 1) if i == 0 else None,
            sleep_quality_score=round(random.uniform(45, 85), 0) if i == 0 else None,
            stress_score=round(random.uniform(20, 75), 0)
        )
        readings.append(reading.model_dump(mode="json"))

    return readings[:10]  # Return most recent 10 for context window efficiency


@tool
def fetch_glucose_readings(patient_id: str, hours_back: int = 24) -> list[dict]:
    """
    Fetch continuous glucose monitor (CGM) readings.

    Production source: Dexcom Share API (G6/G7), Abbott LibreLink API (FreeStyle Libre 3),
    Medtronic CareLink API (Guardian Sensor 3).
    Reading frequency: Every 5 minutes (CGM standard).

    Args:
        patient_id: Patient identifier (only relevant for diabetic patients)
        hours_back: Hours of CGM history to retrieve

    Returns:
        List of glucose readings with trend arrows
    """
    # Only diabetic patients have CGM
    has_diabetes = True  # Both demo patients have T2DM

    if not has_diabetes:
        return []

    readings = []
    num_readings = min(hours_back * 12, 144)  # 12 per hour for CGM

    # Simulate daily glucose pattern (dawn phenomenon, post-meal spikes)
    base_glucose = random.uniform(110, 175)
    trend_options = ["rising_rapidly", "rising", "stable", "falling", "falling_rapidly"]
    trend_weights = [5, 20, 50, 20, 5]

    for i in range(min(num_readings, 20)):  # Cap at 20 for context efficiency
        ts = datetime.now() - timedelta(minutes=i * 5)
        hour = ts.hour

        # Simulate time-of-day patterns
        if 6 <= hour <= 9:
            glucose_adj = random.uniform(10, 40)  # Dawn phenomenon
        elif 12 <= hour <= 14 or 18 <= hour <= 20:
            glucose_adj = random.uniform(20, 60)  # Post-meal spikes
        else:
            glucose_adj = random.uniform(-15, 15)

        glucose = max(40, min(400, base_glucose + glucose_adj + random.uniform(-10, 10)))
        trend = random.choices(trend_options, weights=trend_weights)[0]
        in_range = 70 <= glucose <= 180
        tir = round(random.uniform(45, 85), 1)

        reading = GlucoseReading(
            timestamp=ts,
            patient_id=patient_id,
            device_type="Dexcom G7",
            glucose_mg_dl=round(glucose, 1),
            trend=trend,
            in_target_range=in_range,
            time_in_range_percent_today=tir
        )
        readings.append(reading.model_dump(mode="json"))

    return readings


@tool
def fetch_blood_pressure_readings(patient_id: str, days_back: int = 7) -> list[dict]:
    """
    Fetch blood pressure readings from connected BP monitor.

    Production source: Omron Connect API (Omron Evolv/Complete),
    Withings Health API (Withings BPM Connect),
    iHealth Align API, QardioArm API.
    Reading frequency: Patient-initiated, typically 1-3x daily per protocol.

    Args:
        patient_id: Patient identifier
        days_back: Days of BP history to retrieve

    Returns:
        List of blood pressure readings
    """
    readings = []
    num_readings = days_back * 2  # ~2 readings per day

    # Different BP profiles per patient
    if patient_id == "P001":   # Hypertensive diabetic with CKD
        base_systolic = random.uniform(135, 155)
        base_diastolic = random.uniform(82, 95)
    elif patient_id == "P002":  # Heart failure patient
        base_systolic = random.uniform(110, 145)
        base_diastolic = random.uniform(70, 90)
    else:
        base_systolic = random.uniform(120, 145)
        base_diastolic = random.uniform(75, 90)

    for i in range(num_readings):
        ts = datetime.now() - timedelta(days=i * days_back / num_readings)
        systolic = round(base_systolic + random.uniform(-15, 20), 0)
        diastolic = round(base_diastolic + random.uniform(-8, 12), 0)
        pulse = round(random.uniform(58, 95), 0)

        reading = BloodPressureReading(
            timestamp=ts,
            patient_id=patient_id,
            device_type="Omron Evolv",
            systolic_mmhg=systolic,
            diastolic_mmhg=diastolic,
            pulse_bpm=pulse,
            irregular_heartbeat_detected=random.random() < 0.08,
            arm="left",
            body_position="sitting"
        )
        readings.append(reading.model_dump(mode="json"))

    return readings


@tool
def fetch_medication_adherence(patient_id: str, days_back: int = 14) -> list[dict]:
    """
    Fetch medication adherence records from smart dispenser or patient app.

    Production source: Medisafe API (patient app), Hero Health API (smart dispenser),
    AdhereTech smart bottle sensors, Pharmacy refill records via SureScripts,
    Epic MyChart medication log.
    Data: Timestamps of dispenser openings, app-logged dose events, refill dates.

    Args:
        patient_id: Patient identifier
        days_back: Days of adherence history

    Returns:
        List of medication dose records (taken/missed)
    """
    profile_raw = get_patient_profile.invoke({"patient_id": patient_id})
    medications = profile_raw.get("medications", [])

    records = []
    # Simulate adherence rate per medication (some patients miss some meds more)
    adherence_rates = {med["name"]: random.uniform(0.55, 0.98) for med in medications}

    for day_offset in range(days_back):
        day = datetime.now() - timedelta(days=day_offset)
        for med in medications:
            freq = med.get("frequency", "once_daily")
            doses_per_day = 2 if freq == "twice_daily" else 1

            for dose_num in range(doses_per_day):
                hour = 8 if dose_num == 0 else 20
                scheduled = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                taken = random.random() < adherence_rates[med["name"]]

                taken_time = None
                if taken:
                    delay_minutes = random.gauss(0, 45)
                    taken_time = scheduled + timedelta(minutes=delay_minutes)

                record = MedicationAdherenceRecord(
                    patient_id=patient_id,
                    medication_name=med["name"],
                    scheduled_dose_time=scheduled,
                    taken_time=taken_time,
                    taken=taken,
                    dose_mg=float(med["dose"].replace("mg", "").strip()),
                    method="pill",
                    source="smart_dispenser" if taken else "smart_dispenser",
                    notes="Late dose" if taken and abs(delay_minutes) > 60 else None
                )
                records.append(record.model_dump(mode="json"))

    return records


@tool
def fetch_symptom_reports(patient_id: str, days_back: int = 7) -> list[dict]:
    """
    Fetch patient-reported symptom assessments.

    Production source: Custom patient mobile app (iOS/Android),
    Epic MyChart patient portal, Twilio IVR automated phone check-ins,
    Care coordinator nursing call documentation in EHR.
    Frequency: Daily symptom check-in prompts, plus ad-hoc patient reports.

    Args:
        patient_id: Patient identifier
        days_back: Days of symptom history

    Returns:
        List of symptom reports
    """
    reports = []
    symptom_library = {
        "P001": [  # Diabetic with hypertension
            "headache", "dizziness", "blurry_vision", "increased_thirst",
            "frequent_urination", "fatigue", "tingling_feet", "swollen_ankles"
        ],
        "P002": [  # Heart failure
            "shortness_of_breath", "leg_swelling", "fatigue", "chest_tightness",
            "orthopnea", "palpitations", "dizziness", "reduced_exercise_tolerance",
            "nocturnal_dyspnea", "cough"
        ]
    }

    patient_symptoms = symptom_library.get(patient_id, symptom_library["P001"])

    for day_offset in range(days_back):
        ts = datetime.now() - timedelta(days=day_offset, hours=random.randint(7, 10))

        # Generate 1-4 symptoms per report, with random severity
        num_symptoms = random.randint(0, 4)
        reported_symptoms = []
        for sym in random.sample(patient_symptoms, k=min(num_symptoms, len(patient_symptoms))):
            severity = random.randint(1, 8)
            reported_symptoms.append({
                "symptom_name": sym,
                "severity_1_to_10": severity,
                "duration_hours": random.randint(1, 48),
                "notes": "Worse after exertion" if severity > 6 and random.random() > 0.5 else None
            })

        report = SymptomReport(
            timestamp=ts,
            patient_id=patient_id,
            reporting_method=random.choices(
                ["mobile_app", "web_portal", "phone_ivr"],
                weights=[60, 20, 20]
            )[0],
            symptoms=reported_symptoms,
            pain_scale=random.randint(0, 7),
            fatigue_scale=random.randint(1, 8),
            mood_scale=random.randint(3, 9),
            fell_today=random.random() < 0.03,
            visited_er_since_last_check=random.random() < 0.02,
            free_text=random.choice([
                None, None, None,
                "Felt really tired after morning walk today",
                "Skipped my evening dose by mistake",
                "BP reading seemed high this morning"
            ])
        )
        reports.append(report.model_dump(mode="json"))

    return reports


@tool
def fetch_lab_results(patient_id: str) -> list[dict]:
    """
    Fetch recent laboratory results from EHR integration.

    Production source: HL7 FHIR R4 DiagnosticReport and Observation resources.
    Connects to: Epic FHIR API, Cerner Millennium API, Athenahealth API.
    Also integrates with: Quest Diagnostics MyQuest API, LabCorp Patient API.
    Results flow automatically when ordered tests are resulted in the lab system.

    Args:
        patient_id: Patient identifier

    Returns:
        List of recent lab result panels
    """
    results = []

    if patient_id == "P001":
        # Diabetic with hypertension and CKD - recent labs
        results.append(LabResult(
            result_date=date.today() - timedelta(days=random.randint(14, 45)),
            patient_id=patient_id,
            lab_name="Quest Diagnostics",
            ordered_by="Dr. Sarah Williams, MD",
            results={
                "HbA1c": {
                    "value": round(random.uniform(7.2, 9.5), 1),
                    "unit": "%",
                    "reference_range": "< 5.7% (normal), < 7.0% (diabetic target)",
                    "flag": "H"
                },
                "eGFR": {
                    "value": round(random.uniform(38, 58), 0),
                    "unit": "mL/min/1.73m2",
                    "reference_range": "> 60",
                    "flag": "L"
                },
                "Creatinine": {
                    "value": round(random.uniform(1.4, 2.1), 2),
                    "unit": "mg/dL",
                    "reference_range": "0.57-1.00 (female)",
                    "flag": "H"
                },
                "Potassium": {
                    "value": round(random.uniform(4.2, 5.4), 1),
                    "unit": "mEq/L",
                    "reference_range": "3.5-5.0",
                    "flag": "H" if random.random() > 0.5 else "N"
                },
                "LDL Cholesterol": {
                    "value": round(random.uniform(88, 135), 0),
                    "unit": "mg/dL",
                    "reference_range": "< 70 (high CV risk)",
                    "flag": "H"
                },
                "Urine Microalbumin/Creatinine Ratio": {
                    "value": round(random.uniform(45, 180), 0),
                    "unit": "mg/g",
                    "reference_range": "< 30",
                    "flag": "H"
                }
            },
            hba1c_percent=round(random.uniform(7.2, 9.5), 1),
            creatinine_mg_dl=round(random.uniform(1.4, 2.1), 2),
            egfr_ml_min=round(random.uniform(38, 58), 0),
            cholesterol_total_mg_dl=round(random.uniform(165, 210), 0),
            ldl_mg_dl=round(random.uniform(88, 135), 0),
            potassium_meq_l=round(random.uniform(4.2, 5.4), 1)
        ).model_dump(mode="json"))

    elif patient_id == "P002":
        # Heart failure patient
        results.append(LabResult(
            result_date=date.today() - timedelta(days=random.randint(7, 30)),
            patient_id=patient_id,
            lab_name="LabCorp",
            ordered_by="Dr. Anna Fischer, Cardiology",
            results={
                "BNP (B-Natriuretic Peptide)": {
                    "value": round(random.uniform(280, 950), 0),
                    "unit": "pg/mL",
                    "reference_range": "< 100 pg/mL (age-adjusted)",
                    "flag": "HH"
                },
                "Sodium": {
                    "value": round(random.uniform(132, 140), 0),
                    "unit": "mEq/L",
                    "reference_range": "136-145",
                    "flag": "L" if random.random() > 0.4 else "N"
                },
                "Potassium": {
                    "value": round(random.uniform(3.8, 5.1), 1),
                    "unit": "mEq/L",
                    "reference_range": "3.5-5.0",
                    "flag": "N"
                },
                "Creatinine": {
                    "value": round(random.uniform(1.2, 1.9), 2),
                    "unit": "mg/dL",
                    "reference_range": "0.74-1.35 (male)",
                    "flag": "H"
                },
                "HbA1c": {
                    "value": round(random.uniform(7.5, 9.0), 1),
                    "unit": "%",
                    "reference_range": "< 7.0% (diabetic target)",
                    "flag": "H"
                }
            },
            hba1c_percent=round(random.uniform(7.5, 9.0), 1),
            creatinine_mg_dl=round(random.uniform(1.2, 1.9), 2),
            egfr_ml_min=round(random.uniform(40, 62), 0),
            sodium_meq_l=round(random.uniform(132, 140), 0),
            potassium_meq_l=round(random.uniform(3.8, 5.1), 1),
            bnp_pg_ml=round(random.uniform(280, 950), 0)
        ).model_dump(mode="json"))

    return results
