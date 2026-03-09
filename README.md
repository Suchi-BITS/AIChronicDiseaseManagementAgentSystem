# AI Chronic Disease Management Agent System

An agentic AI system built with LangGraph and LangChain that continuously monitors patient
lifestyle data, medication adherence, and symptoms to provide proactive, evidence-based
health interventions for patients with chronic conditions.

---

## Data Sources

This system ingests data from six distinct real-world source categories:

| Agent | Data Type | Production Source | Frequency |
|-------|-----------|-------------------|-----------|
| Vitals Agent | Heart rate, SpO2, HRV, steps, sleep | Apple HealthKit, Google Health Connect, Fitbit API, Garmin Connect | Every 5 min (HR/SpO2), hourly (aggregates) |
| Glucose Agent | Continuous glucose readings, TIR, trend | Dexcom Share API (G6/G7), Abbott LibreLink (FreeStyle Libre 3) | Every 5 minutes (CGM standard) |
| BP Agent | Systolic/diastolic BP, pulse, AF detection | Omron Connect API, Withings Health API, iHealth API | 1-3x daily (patient-initiated per protocol) |
| Medication Agent | Dose timestamps, taken/missed, late doses | Medisafe API, Hero Health (smart dispenser), SureScripts (refill) | Event-driven + hourly check |
| Symptom & Lab Agent | Patient-reported symptoms, HbA1c, eGFR, BNP, electrolytes | Epic MyChart app, Twilio IVR, HL7 FHIR R4 (Epic/Cerner), Quest/LabCorp APIs | Twice daily (symptoms), event-driven (labs) |
| Intervention Agent | Executes notifications, EHR alerts, emergency responses | Twilio SMS/voice, Firebase FCM, Epic In-Basket, PagerDuty | Real-time (triggered by monitoring findings) |

All data in this codebase is **fully simulated** with clinically realistic ranges.
HIPAA compliance requires a BAA with all production data vendors.

---

## Agent Graph Architecture

```
[SUPERVISOR] (load patient profile from EHR)
      |
      v
[VITALS AGENT]       -- wearable: HR, SpO2, HRV, sleep, activity
      |
      v
[GLUCOSE AGENT]      -- CGM: glucose mg/dL, trend arrows, time-in-range
      |
      v
[BP AGENT]           -- home BP monitor: systolic/diastolic, 7-day average, AF flag
      |
      v
[MEDICATION AGENT]   -- smart dispenser/app: 14-day dose history, pattern analysis
      |
      v
[SYMPTOM & LAB AGENT]-- patient app: self-reported symptoms + EHR lab results
      |
      v
[INTERVENTION AGENT] -- executes targeted interventions via LangChain tool calls
      |
      v
[SUPERVISOR]         -- synthesizes progress summary, files EHR note
      |
      v
    [END]
```

---

## Intervention Escalation Framework

The system applies a four-tier escalation model:

| Level | Trigger Examples | Actions Taken |
|-------|-----------------|---------------|
| Emergency | Glucose < 54 mg/dL with symptoms, SBP > 180 + chest pain, SpO2 < 88% with acute dyspnea | EMS alert, emergency contact call, physician STAT page, EHR emergency flag |
| Urgent | SBP consistently >= 180, glucose > 350, new arrhythmia, fall reported, BNP > 500 | Physician/care coordinator urgent alert + patient notification |
| Care Coordination | HbA1c >= 9%, eGFR < 30, adherence < 50% for critical meds, LDL > 100 on max statin | EHR task for physician review, coordinator follow-up scheduling |
| Patient Coaching | Post-meal glucose spikes, activity below target, mild symptoms, sleep disruption | In-app nudge, SMS reminder, educational content, motivational message |

---

## Supported Chronic Conditions

- **Type 2 Diabetes**: Glycemic monitoring via CGM, HbA1c trending, hypoglycemia detection, medication adherence for oral hypoglycemics
- **Hypertension**: HBPM trend analysis, hypertensive urgency/emergency detection, antihypertensive adherence
- **Heart Failure**: Functional decline detection (activity + SpO2), decompensation early warning (weight gain proxy, BNP, edema symptoms), diuretic adherence
- **COPD**: SpO2 monitoring for exacerbation detection, respiratory rate trending, inhaler adherence
- **Chronic Kidney Disease**: eGFR monitoring, electrolyte management (hyperkalemia risk with ACE inhibitors), medication dose adjustment flags

---

## Project Structure

```
chronic_care_agents/
|-- main.py                          # Entry point, runs full monitoring cycle
|-- requirements.txt
|-- .env.example
|
|-- config/
|   |-- settings.py                  # Clinical thresholds, system config
|
|-- data/
|   |-- models.py                    # Pydantic models for all clinical data types
|
|-- agents/
|   |-- supervisor_agent.py          # Orchestrator, patient loader, report synthesizer
|   |-- vitals_agent.py              # Wearable vitals analysis
|   |-- glucose_agent.py             # CGM glucose analysis
|   |-- bp_agent.py                  # Home BP monitor analysis
|   |-- medication_agent.py          # Adherence pattern analysis
|   |-- symptom_lab_agent.py         # Symptoms + lab result interpretation
|   |-- intervention_agent.py        # Evidence-based intervention execution
|
|-- tools/
|   |-- data_collection_tools.py     # Simulated device/EHR data feeds (6 tools)
|   |-- intervention_tools.py        # Action tools: notify, alert, recommend, log (5 tools)
|
|-- graph/
|   |-- care_graph.py                # LangGraph StateGraph definition
```

---

## Clinical Data Models

```python
class AgentState(BaseModel):
    # Patient
    patient: Optional[PatientProfile]

    # Device data (populated by monitoring agents)
    vitals_readings: list[WearableVitalsReading]    # from Apple Watch / Fitbit
    glucose_readings: list[GlucoseReading]           # from Dexcom / Libre
    bp_readings: list[BloodPressureReading]           # from Omron / Withings
    medication_records: list[MedicationAdherenceRecord]  # from Medisafe / Hero
    symptom_reports: list[SymptomReport]             # from patient app / IVR
    lab_results: list[LabResult]                     # from EHR FHIR

    # Analysis (populated by monitoring agents)
    vitals_analysis: Optional[str]
    glucose_analysis: Optional[str]
    bp_analysis: Optional[str]
    medication_analysis: Optional[str]
    symptom_analysis: Optional[str]

    # Risk scores (computed per domain)
    risk_scores: list[ClinicalRiskScore]             # glycemic, cardiovascular, medication, overall

    # Interventions (populated by intervention agent)
    interventions: list[HealthIntervention]
    care_plan_adjustments: list[CarePlanAdjustment]

    # Final output
    progress_summary: Optional[PatientProgressSummary]
    emergency_triggered: bool
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# Run monitoring cycle for Patient P001 (Margaret Chen - T2DM + HTN + CKD)
python main.py

# To monitor Patient P002 (Thomas Rivera - Heart Failure), edit main.py:
# run_monitoring_cycle(patient_id="P002")
```

---

## Clinical Guidelines Referenced

- ADA Standards of Medical Care in Diabetes (2024)
- ACC/AHA Hypertension Guidelines (2017)
- AHA/ACC Heart Failure Guidelines (2022)
- KDIGO CKD Clinical Practice Guidelines (2022)
- AGS/BGS Clinical Practice Guideline for Prevention of Falls in Older Persons

---

## Safety and Ethics

This system is **decision support only**. It does not replace clinical judgment.

- All care plan recommendations require physician review and approval before implementation
- Emergency interventions trigger human escalation — the AI does not autonomously call emergency services in production without human-in-the-loop validation
- All automated notes filed to the EHR require cosignature by a supervising clinician
- Patient notifications are written in plain language and are never alarmist
- The system does not prescribe medications or specific dosages
- All monitoring data is subject to HIPAA and applicable data protection regulations

---

## Production Deployment Checklist

- [ ] Execute BAA (Business Associate Agreement) with all data vendors
- [ ] HL7 FHIR R4 integration with EHR (Epic / Cerner sandbox testing)
- [ ] Device API integrations with OAuth2 (Apple HealthKit, Fitbit, Dexcom)
- [ ] HIPAA-compliant audit logging database (not in-memory)
- [ ] Human-in-the-loop review for all emergency escalations
- [ ] Clinical validation of AI-generated recommendations by physician panel
- [ ] IRB review if used in research context
- [ ] FDA SaMD (Software as a Medical Device) regulatory pathway assessment
- [ ] Patient consent and data sharing agreements
- [ ] Bias audit across demographic groups
