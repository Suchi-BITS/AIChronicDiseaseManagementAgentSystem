# agents/all_agents.py
# All seven monitoring and planning agents.
# Each agent:
#   1. Loads its data from simulation.py (or real APIs in production)
#   2. Computes statistics from raw data
#   3. Calls LLM (or demo text) for clinical analysis
#   4. Writes results back to the shared state dict
#   5. Sets state["current_agent"] to route LangGraph to the next node

import json
from datetime import datetime, timedelta
from collections import defaultdict

from data.simulation import (
    PATIENTS, get_wearable_vitals, get_glucose_readings,
    get_bp_readings, get_medication_records,
    get_symptom_reports, get_lab_results,
)
from config.settings import care_config
from agents.base import call_llm, call_llm_with_tools, DEMO_MODE


# ==========================================================================
# SUPERVISOR  (entry point + final synthesis)
# ==========================================================================

def run_supervisor(state: dict) -> dict:
    print("\n[SUPERVISOR] Processing state...")

    # --- INITIAL ENTRY: load patient and start pipeline ---
    if state["patient"] is None:
        patient_id = state.get("target_patient_id", "P001")
        state["patient"] = PATIENTS[patient_id]
        state["current_agent"] = "vitals_agent"
        state["iteration_count"] += 1
        p = state["patient"]
        print(f"[SUPERVISOR] Loaded patient: {p['name']} | "
              f"Conditions: {', '.join(p['conditions'])} | "
              f"Risk: {p['risk_stratification'].upper()}")
        return state

    # --- FINAL SYNTHESIS: after all agents have run ---
    print("[SUPERVISOR] Synthesizing final patient progress summary...")

    risk_scores  = state["risk_scores"]
    interventions = state["interventions"]
    care_plan    = state["care_plan_adjustments"]
    overall_score = max((r["score"] for r in risk_scores), default=0)

    if state["emergency_triggered"] or overall_score > 80:
        overall_status = "critical"
    elif overall_score > 60:
        overall_status = "poor"
    elif overall_score > 40:
        overall_status = "fair"
    elif overall_score > 20:
        overall_status = "good"
    else:
        overall_status = "excellent"

    p = state["patient"]
    risk_lines = "\n".join(
        f"  {r['domain']:20s}: {r['score']:.0f}/100 [{r['risk_level'].upper()}] "
        f"- {', '.join(r['factors'][:2])}"
        for r in risk_scores
    )

    demo_text = f"""
PATIENT STATUS: {overall_status.upper()}
Patient: {p['name']} | Age: {p['age']} | Risk: {p['risk_stratification'].upper()}
Conditions: {', '.join(p['conditions'])}
Monitoring period: {(datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}

RISK DOMAIN SUMMARY:
{risk_lines}

EXECUTIVE SUMMARY:
{p['name']} is a {p['age']}-year-old patient with {', '.join(p['conditions'])} presenting
with {overall_status} overall health status this monitoring period. The composite risk
score of {overall_score:.0f}/100 reflects findings across {len(risk_scores)} monitoring domains.
A total of {len(interventions)} interventions were executed automatically, and
{len(care_plan)} care plan modifications have been submitted for physician review.

KEY FINDINGS:
- Glycemic control: {state.get('glucose_analysis','N/A')[:120] if state.get('glucose_analysis') else 'Not applicable'}
- Blood pressure: {state.get('bp_analysis','N/A')[:120] if state.get('bp_analysis') else 'Not assessed'}
- Medication adherence: {state.get('medication_analysis','N/A')[:120] if state.get('medication_analysis') else 'Not assessed'}
- Symptoms/Labs: {state.get('symptom_analysis','N/A')[:120] if state.get('symptom_analysis') else 'Not assessed'}

INTERVENTIONS EXECUTED ({len(interventions)}):
{chr(10).join(f"  [{i['severity'].upper()}] {i['type']} -> {i['recipient']}: {i['title']}" for i in interventions) or "  None required"}

CARE PLAN CHANGES PENDING PHYSICIAN REVIEW ({len(care_plan)}):
{chr(10).join(f"  [{a['urgency'].upper()}] {a['change_type']}: {a['recommendation'][:80]}" for a in care_plan) or "  None recommended"}

EMERGENCY STATUS: {"EMERGENCY PROTOCOLS ACTIVATED" if state['emergency_triggered'] else "No emergency this cycle"}

GOALS FOR NEXT MONITORING PERIOD:
1. Review medication adherence for Atorvastatin and Aspirin — consider blister pack
2. Reassess blood pressure control — target < 130/80 mmHg per ADA/ACC guidelines
3. Reinforce dietary carbohydrate management to improve TIR (currently < 70% target)
4. Schedule follow-up lab work in 6 weeks (HbA1c, eGFR, potassium)

{care_config.disclaimer}
"""

    user_prompt = f"Summarize findings for {p['name']}, status={overall_status}, score={overall_score:.0f}"
    narrative = call_llm(
        system_prompt="You are a clinical supervisor AI generating a patient progress report.",
        user_prompt=user_prompt,
        demo_response=demo_text,
    )

    from datetime import date
    state["progress_summary"] = {
        "patient_id":      p["patient_id"],
        "patient_name":    p["name"],
        "period_start":    (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "period_end":      datetime.now().strftime("%Y-%m-%d"),
        "overall_status":  overall_status,
        "composite_risk":  overall_score,
        "risk_scores":     risk_scores,
        "interventions":   interventions,
        "care_plan_adjustments": care_plan,
        "emergency_triggered":   state["emergency_triggered"],
        "narrative":       narrative,
    }

    state["current_agent"] = "complete"
    print(f"[SUPERVISOR] Done. Status: {overall_status.upper()} | "
          f"Risk: {overall_score:.0f}/100 | "
          f"Interventions: {len(interventions)} | "
          f"Care plan recs: {len(care_plan)}")
    return state


# ==========================================================================
# VITALS AGENT
# ==========================================================================

def run_vitals_agent(state: dict) -> dict:
    p = state["patient"]
    print(f"\n[VITALS AGENT] Reading wearable data for {p['name']}...")

    readings = get_wearable_vitals(p["patient_id"], num_readings=12)
    state["vitals_readings"] = readings

    hrs   = [r["heart_rate_bpm"]        for r in readings]
    spo2s = [r["spo2_percent"]           for r in readings if r["spo2_percent"]]
    hrvs  = [r["heart_rate_variability_ms"] for r in readings if r.get("heart_rate_variability_ms")]
    latest = readings[0]

    avg_hr   = sum(hrs) / len(hrs)
    min_spo2 = min(spo2s)
    avg_spo2 = sum(spo2s) / len(spo2s)
    avg_hrv  = sum(hrvs) / len(hrvs) if hrvs else 0

    demo_analysis = f"""
WEARABLE VITALS ANALYSIS — {p['name']}
Source: {latest['device_type']} (last 12 readings, 24-hour window)

Heart Rate:
  Average: {avg_hr:.1f} bpm | Min: {min(hrs):.1f} | Max: {max(hrs):.1f}
  Status: {"ELEVATED — above 100 bpm threshold, warrants monitoring" if avg_hr > care_config.hr_high else "Within normal range for this patient"}
  HRV: {avg_hrv:.1f} ms — {"Reduced variability indicating physiological stress" if avg_hrv < 25 else "Acceptable variability"}

SpO2 (Blood Oxygen):
  Average: {avg_spo2:.1f}% | Minimum recorded: {min_spo2:.1f}%
  Status: {"CRITICAL LOW — below {care_config.spo2_critical_low}%, urgent respiratory assessment needed" if min_spo2 < care_config.spo2_critical_low else ("LOW — below 92%, monitor closely for COPD/HF exacerbation" if min_spo2 < care_config.spo2_low else "Acceptable")}

Activity:
  Steps today: {latest['steps_today']:,} (target: 7,000–10,000/day)
  Active minutes: {latest['active_minutes_today']} min (target: 30+ min/day)
  Status: {"Below recommended activity level — may reflect fatigue or functional decline" if latest['steps_today'] < 3000 else "Adequate activity level maintained"}

Sleep (last night):
  Duration: {latest['sleep_hours_last_night']} hours
  Quality score: {latest['sleep_quality_score']}/100
  Status: {"Insufficient sleep — affects glycemic control and BP regulation" if (latest['sleep_hours_last_night'] or 7) < 6 else "Adequate sleep duration"}

Stress score: {latest['stress_score']:.0f}/100 — {"Elevated — may contribute to BP and glucose variability" if latest['stress_score'] > 60 else "Moderate"}

CARDIOVASCULAR RISK CONTRIBUTION:
{"ELEVATED RISK — SpO2 below safety threshold or HR consistently high" if min_spo2 < care_config.spo2_low or avg_hr > care_config.hr_high else "Moderate baseline cardiovascular risk consistent with known conditions"}
"""

    state["vitals_analysis"] = call_llm(
        system_prompt="You are a clinical vitals monitoring agent.",
        user_prompt=f"Analyze: avg_hr={avg_hr:.1f}, min_spo2={min_spo2:.1f}, steps={latest['steps_today']}, sleep={latest['sleep_hours_last_night']}h",
        demo_response=demo_analysis,
    )

    # Risk score
    risk = 20.0
    factors = []
    if avg_hr > care_config.hr_high:
        risk += 20; factors.append(f"Elevated HR: {avg_hr:.1f} bpm")
    if min_spo2 < care_config.spo2_critical_low:
        risk += 40; factors.append(f"Critical SpO2: {min_spo2:.1f}%")
    elif min_spo2 < care_config.spo2_low:
        risk += 20; factors.append(f"Low SpO2: {min_spo2:.1f}%")
    if latest["steps_today"] < 2000:
        risk += 10; factors.append(f"Very low activity: {latest['steps_today']} steps")
    if (latest.get("sleep_hours_last_night") or 7) < 5:
        risk += 10; factors.append("Insufficient sleep < 5h")

    state["risk_scores"].append({
        "domain": "cardiovascular_vitals",
        "score": min(risk, 100),
        "risk_level": "critical" if risk>75 else "high" if risk>50 else "moderate" if risk>25 else "low",
        "factors": factors or ["No critical vitals findings"],
        "trend": "stable",
    })

    state["current_agent"] = "glucose_agent"
    print(f"[VITALS AGENT] HR: {avg_hr:.1f} bpm | SpO2 min: {min_spo2:.1f}% | "
          f"Steps: {latest['steps_today']} | Sleep: {latest['sleep_hours_last_night']}h | "
          f"Risk: {state['risk_scores'][-1]['risk_level'].upper()}")
    return state


# ==========================================================================
# GLUCOSE AGENT
# ==========================================================================

def run_glucose_agent(state: dict) -> dict:
    p = state["patient"]
    has_diabetes = any(c in p["conditions"] for c in ["type2_diabetes", "type1_diabetes"])

    if not has_diabetes:
        print(f"\n[GLUCOSE AGENT] No diabetes diagnosis — skipping.")
        state["glucose_analysis"] = "Not applicable: no diabetes diagnosis."
        state["current_agent"] = "bp_agent"
        return state

    print(f"\n[GLUCOSE AGENT] Reading CGM data for {p['name']}...")

    readings = get_glucose_readings(p["patient_id"], num_readings=20)
    state["glucose_readings"] = readings

    vals    = [r["glucose_mg_dl"] for r in readings]
    tirs    = [r["time_in_range_percent_today"] for r in readings]
    avg_g   = sum(vals) / len(vals)
    min_g   = min(vals)
    max_g   = max(vals)
    avg_tir = sum(tirs) / len(tirs)
    latest  = readings[0]
    hypo    = [r for r in readings if r["glucose_mg_dl"] < 70]
    crit_h  = [r for r in readings if r["glucose_mg_dl"] < 54]
    hyper   = [r for r in readings if r["glucose_mg_dl"] > 250]

    demo_analysis = f"""
GLYCEMIC ANALYSIS — {p['name']}
Source: {latest['device_type']} ({len(readings)} readings, last 24 hours)

Current Reading:
  Glucose: {latest['glucose_mg_dl']:.1f} mg/dL | Trend: {latest['trend'].replace('_',' ').upper()}
  {"ACTION REQUIRED: Falling rapidly — hypoglycemia risk imminent" if latest['trend'] == 'falling_rapidly' else
   "Monitor: Glucose rising rapidly — post-meal spike likely" if latest['trend'] == 'rising_rapidly' else
   "Trend acceptable"}

24-Hour Summary:
  Average: {avg_g:.1f} mg/dL (target: 80–180 mg/dL)
  Range: {min_g:.1f} – {max_g:.1f} mg/dL
  Time in range (70–180 mg/dL): {avg_tir:.1f}% (target: ≥ 70%)
  Hypoglycemic events (< 70 mg/dL): {len(hypo)} {"⚠ ABOVE ACCEPTABLE LIMIT" if len(hypo) > 1 else ""}
  Critical hypoglycemia (< 54 mg/dL): {len(crit_h)} {"⚠ EMERGENCY THRESHOLD" if crit_h else ""}
  Significant hyperglycemia (> 250 mg/dL): {len(hyper)}

Clinical Assessment:
  TIR of {avg_tir:.1f}% is {"POOR — major intervention needed" if avg_tir < 50 else "SUBOPTIMAL — room for improvement" if avg_tir < 70 else "GOOD — meeting target"}.
  {"Average glucose of " + str(round(avg_g,1)) + " mg/dL suggests HbA1c in the range of approximately " + str(round((avg_g+46.7)/28.7, 1)) + "% (eAG formula)." }

Medication Context:
  Patient on Metformin — low intrinsic hypoglycemia risk.
  {"High TIR variability suggests inconsistent carbohydrate intake or meal timing." if avg_tir < 65 else ""}

Glycemic Risk: {"HIGH — TIR below 50%, significant hyperglycemia events" if avg_tir < 50 or len(hyper) > 2 else "MODERATE — TIR suboptimal, ongoing monitoring needed"}
"""

    state["glucose_analysis"] = call_llm(
        system_prompt="You are a clinical CGM analysis agent.",
        user_prompt=f"Analyze: avg={avg_g:.1f}, TIR={avg_tir:.1f}%, hypo={len(hypo)}, hyper={len(hyper)}, trend={latest['trend']}",
        demo_response=demo_analysis,
    )

    risk = 15.0
    factors = []
    if crit_h:
        risk += 50; factors.append(f"Critical hypoglycemia: {len(crit_h)} events < 54 mg/dL")
    elif hypo:
        risk += 25; factors.append(f"Hypoglycemia events: {len(hypo)} readings < 70 mg/dL")
    if max_g > care_config.glucose_critical_high:
        risk += 30; factors.append(f"Critical hyperglycemia: {max_g:.0f} mg/dL")
    elif max_g > care_config.glucose_high:
        risk += 15; factors.append(f"Significant hyperglycemia: {max_g:.0f} mg/dL")
    if avg_tir < 50:
        risk += 20; factors.append(f"Poor TIR: {avg_tir:.1f}% (target ≥70%)")
    elif avg_tir < 70:
        risk += 10; factors.append(f"Suboptimal TIR: {avg_tir:.1f}%")

    state["risk_scores"].append({
        "domain": "glycemic",
        "score": min(risk, 100),
        "risk_level": "critical" if risk>75 else "high" if risk>50 else "moderate" if risk>25 else "low",
        "factors": factors or ["Glycemic control within acceptable range"],
        "trend": "stable",
    })

    state["current_agent"] = "bp_agent"
    print(f"[GLUCOSE AGENT] Avg: {avg_g:.1f} mg/dL | TIR: {avg_tir:.1f}% | "
          f"Hypo: {len(hypo)} | Hyper: {len(hyper)} | "
          f"Risk: {state['risk_scores'][-1]['risk_level'].upper()}")
    return state


# ==========================================================================
# BLOOD PRESSURE AGENT
# ==========================================================================

def run_bp_agent(state: dict) -> dict:
    p = state["patient"]
    print(f"\n[BP AGENT] Reading blood pressure data for {p['name']}...")

    readings = get_bp_readings(p["patient_id"], num_readings=14)
    state["bp_readings"] = readings

    sys_vals  = [r["systolic_mmhg"]  for r in readings]
    dia_vals  = [r["diastolic_mmhg"] for r in readings]
    irregular = [r for r in readings if r["irregular_heartbeat_detected"]]
    avg_sys   = sum(sys_vals) / len(sys_vals)
    avg_dia   = sum(dia_vals) / len(dia_vals)
    max_sys   = max(sys_vals)
    high_rdgs = [r for r in readings if r["systolic_mmhg"] >= care_config.bp_systolic_high]
    crit_rdgs = [r for r in readings if r["systolic_mmhg"] >= care_config.bp_systolic_critical_high]

    # Patient-specific BP target
    has_ckd      = "chronic_kidney_disease" in p["conditions"]
    has_diabetes = "type2_diabetes" in p["conditions"]
    target_str   = "< 130/80 mmHg" if (has_diabetes or has_ckd) else "< 140/90 mmHg"

    demo_analysis = f"""
BLOOD PRESSURE ANALYSIS — {p['name']}
Source: {readings[0]['device_type']} ({len(readings)} readings, 7-day home monitoring)

7-Day Average: {avg_sys:.0f}/{avg_dia:.0f} mmHg
Maximum recorded: {max_sys:.0f} mmHg systolic
Patient-specific target: {target_str} (ACC/AHA + ADA guidelines for diabetes/CKD)

Control Assessment:
  Readings ≥ 140/90 (Stage 2 HTN): {len(high_rdgs)} of {len(readings)} ({len(high_rdgs)/len(readings)*100:.0f}%)
  Hypertensive urgency readings (≥ 180): {len(crit_rdgs)}
  {"HYPERTENSIVE URGENCY DETECTED — physician contact required" if crit_rdgs else ""}

Pattern Analysis:
  Morning surge pattern: {"Present — first AM readings consistently 8–18 mmHg higher" if len(readings) > 4 else "Insufficient data"}
  Average is {avg_sys - 130:.0f} mmHg above the 130 systolic target.
  {"Diastolic elevation (> 90) suggests significant vascular resistance" if avg_dia > 90 else "Diastolic is borderline acceptable"}

Irregular Heartbeat:
  Detected {len(irregular)} times — {"Warrants ECG evaluation for atrial fibrillation" if irregular else "No significant concern"}

Clinical Impact:
  Poorly controlled BP with concurrent CKD accelerates renal function decline.
  Target organ protection requires BP < 130/80 mmHg consistently.
  {"Current antihypertensive regimen appears insufficient — consider dose titration or addition of agent" if avg_sys > 140 else "Regimen adequate but optimization possible"}

BP Risk: {"CRITICAL — hypertensive urgency detected" if crit_rdgs else "HIGH — majority of readings above target" if len(high_rdgs) > len(readings)*0.5 else "MODERATE — BP elevated but no urgent readings"}
"""

    state["bp_analysis"] = call_llm(
        system_prompt="You are a clinical blood pressure monitoring agent.",
        user_prompt=f"Analyze: avg={avg_sys:.0f}/{avg_dia:.0f}, max={max_sys:.0f}, high_rdgs={len(high_rdgs)}, crit={len(crit_rdgs)}, irreg={len(irregular)}",
        demo_response=demo_analysis,
    )

    risk = 20.0
    factors = []
    if crit_rdgs:
        risk += 50; factors.append(f"Hypertensive urgency: {len(crit_rdgs)} readings ≥ 180 mmHg")
    elif len(high_rdgs) > len(readings) * 0.6:
        risk += 30; factors.append(f"{len(high_rdgs)}/{len(readings)} readings above 140/90")
    elif avg_sys > care_config.bp_systolic_target_high:
        risk += 20; factors.append(f"Average systolic {avg_sys:.0f} mmHg above {care_config.bp_systolic_target_high:.0f} target")
    if irregular:
        risk += 15; factors.append(f"Irregular heartbeat {len(irregular)}x")

    state["risk_scores"].append({
        "domain": "blood_pressure",
        "score": min(risk, 100),
        "risk_level": "critical" if risk>75 else "high" if risk>50 else "moderate" if risk>25 else "low",
        "factors": factors or ["BP within acceptable range"],
        "trend": "stable",
    })

    state["current_agent"] = "medication_agent"
    print(f"[BP AGENT] Avg: {avg_sys:.0f}/{avg_dia:.0f} mmHg | Max sys: {max_sys:.0f} | "
          f"High readings: {len(high_rdgs)}/{len(readings)} | Irregular: {len(irregular)} | "
          f"Risk: {state['risk_scores'][-1]['risk_level'].upper()}")
    return state


# ==========================================================================
# MEDICATION ADHERENCE AGENT
# ==========================================================================

def run_medication_agent(state: dict) -> dict:
    p = state["patient"]
    print(f"\n[MEDICATION AGENT] Analyzing 14-day adherence for {p['name']}...")

    records = get_medication_records(p["patient_id"], days_back=14)
    state["medication_records"] = records

    by_med = defaultdict(lambda: {"total": 0, "taken": 0, "late": 0})
    for r in records:
        m = r["medication_name"]
        by_med[m]["total"] += 1
        if r["taken"]:
            by_med[m]["taken"] += 1
            if r.get("taken_time") and r.get("scheduled_dose_time"):
                try:
                    sched = datetime.fromisoformat(str(r["scheduled_dose_time"]))
                    taken = datetime.fromisoformat(str(r["taken_time"]))
                    if abs((taken - sched).total_seconds() / 60) > 90:
                        by_med[m]["late"] += 1
                except Exception:
                    pass

    stats = []
    for med, s in by_med.items():
        rate = s["taken"] / s["total"] * 100 if s["total"] else 0
        status = (
            "CRITICAL" if rate < care_config.adherence_critical_percent else
            "POOR"     if rate < care_config.adherence_low_percent     else
            "SUBOPTIMAL" if rate < care_config.adherence_target_percent else
            "GOOD"
        )
        stats.append({
            "medication": med, "adherence_pct": round(rate, 1),
            "taken": s["taken"], "missed": s["total"] - s["taken"],
            "late": s["late"], "status": status,
        })

    overall_adh = sum(s["adherence_pct"] for s in stats) / len(stats) if stats else 0
    critical_meds = [s for s in stats if s["status"] == "CRITICAL"]
    poor_meds     = [s for s in stats if s["status"] == "POOR"]
    good_meds     = [s for s in stats if s["status"] == "GOOD"]

    med_lines = "\n".join(
        f"  {s['medication']:20s}: {s['adherence_pct']:5.1f}%  "
        f"({s['taken']}/{s['taken']+s['missed']} doses)  "
        f"Late: {s['late']}  [{s['status']}]"
        for s in sorted(stats, key=lambda x: x["adherence_pct"])
    )

    demo_analysis = f"""
MEDICATION ADHERENCE ANALYSIS — {p['name']}
Source: Smart pill dispenser (Medisafe) + pharmacy refill records (14 days)

Overall Adherence: {overall_adh:.1f}% (target: ≥ {care_config.adherence_target_percent:.0f}%)

Per-Medication Breakdown:
{med_lines}

Summary:
  Critical non-adherence (< {care_config.adherence_critical_percent:.0f}%): {len(critical_meds)} medications
  Poor adherence ({care_config.adherence_critical_percent:.0f}–{care_config.adherence_low_percent:.0f}%): {len(poor_meds)} medications
  Adequate adherence (≥ {care_config.adherence_target_percent:.0f}%): {len(good_meds)} medications

Pattern Analysis:
  {"Statin (Atorvastatin) and Aspirin show lower adherence — common pattern due to perceived low symptom benefit." if any(s['medication'] in ['Atorvastatin','Aspirin'] and s['adherence_pct'] < 80 for s in stats) else ""}
  {"Critical heart failure medications (Carvedilol, Furosemide, Lisinopril) adherence requires urgent attention." if any(s['medication'] in ['Carvedilol','Furosemide','Lisinopril'] and s['status'] in ['CRITICAL','POOR'] for s in stats) else ""}
  Late doses suggest schedule mismatch or side effects in the first hour after waking.

Pill Burden: {len(p['medications'])} medications/day — {"HIGH burden — polypharmacy review recommended" if len(p['medications']) >= 7 else "Moderate burden — consider dose synchronization"}

Clinical Consequence:
  Suboptimal antihypertensive adherence directly correlates with the elevated BP readings observed.
  {"Statin non-adherence with LDL > 100 mg/dL increases cardiovascular event risk." if any(s['medication']=='Atorvastatin' and s['adherence_pct']<80 for s in stats) else ""}

Recommendations:
  1. Enroll in medication synchronization program (all monthly supplies filled same day)
  2. Enable smart dispenser alerts for missed doses > 2 hours overdue
  3. Consider pill box with weekly pre-loading for improved visual tracking
  4. Pharmacist medication review to assess simplification opportunities
"""

    state["medication_analysis"] = call_llm(
        system_prompt="You are a clinical medication adherence monitoring agent.",
        user_prompt=f"Analyze: overall={overall_adh:.1f}%, critical_meds={[s['medication'] for s in critical_meds]}, total_meds={len(p['medications'])}",
        demo_response=demo_analysis,
    )

    risk = max(0, 100 - overall_adh)
    factors = []
    if critical_meds:
        factors.append(f"Critical non-adherence: {', '.join(m['medication'] for m in critical_meds)}")
        risk = min(risk + 20, 100)
    if poor_meds:
        factors.append(f"Poor adherence: {', '.join(m['medication'] for m in poor_meds)}")
    if overall_adh < care_config.adherence_target_percent:
        factors.append(f"Overall adherence {overall_adh:.1f}% below {care_config.adherence_target_percent:.0f}% target")

    state["risk_scores"].append({
        "domain": "medication_adherence",
        "score": min(risk, 100),
        "risk_level": "critical" if risk>75 else "high" if risk>50 else "moderate" if risk>25 else "low",
        "factors": factors or ["Medication adherence within acceptable range"],
        "trend": "stable",
    })

    state["current_agent"] = "symptom_lab_agent"
    print(f"[MEDICATION AGENT] Overall adherence: {overall_adh:.1f}% | "
          f"Critical meds: {len(critical_meds)} | Poor: {len(poor_meds)} | "
          f"Risk: {state['risk_scores'][-1]['risk_level'].upper()}")
    return state


# ==========================================================================
# SYMPTOM + LAB AGENT
# ==========================================================================

def run_symptom_lab_agent(state: dict) -> dict:
    p = state["patient"]
    print(f"\n[SYMPTOM & LAB AGENT] Loading symptom reports and lab results for {p['name']}...")

    symptoms_raw = get_symptom_reports(p["patient_id"], days_back=7)
    labs_raw     = get_lab_results(p["patient_id"])
    state["symptom_reports"] = symptoms_raw
    state["lab_results"]     = labs_raw

    # Aggregate symptoms
    sym_agg = defaultdict(lambda: {"count": 0, "max_sev": 0, "severities": []})
    for rpt in symptoms_raw:
        for s in rpt.get("symptoms", []):
            n = s["symptom_name"]
            v = s["severity_1_to_10"]
            sym_agg[n]["count"]   += 1
            sym_agg[n]["max_sev"]  = max(sym_agg[n]["max_sev"], v)
            sym_agg[n]["severities"].append(v)

    sym_summary = sorted(
        [{"symptom": n, "days_reported": d["count"],
          "max_severity": d["max_sev"],
          "avg_severity": round(sum(d["severities"])/len(d["severities"]), 1)}
         for n, d in sym_agg.items()],
        key=lambda x: -x["max_severity"]
    )

    er_visits = [r for r in symptoms_raw if r.get("visited_er_since_last_check")]
    falls     = [r for r in symptoms_raw if r.get("fell_today")]
    free_texts = [r["free_text"] for r in symptoms_raw if r.get("free_text")]

    # Lab value extraction
    lab = labs_raw[0] if labs_raw else {}
    hba1c    = lab.get("hba1c_percent")
    egfr     = lab.get("egfr_ml_min")
    creat    = lab.get("creatinine_mg_dl")
    ldl      = lab.get("ldl_mg_dl")
    potassium = lab.get("potassium_meq_l")
    sodium   = lab.get("sodium_meq_l")
    bnp      = lab.get("bnp_pg_ml")

    sym_lines = "\n".join(
        f"  {s['symptom']:30s}: severity={s['max_severity']}/10  reported {s['days_reported']}/7 days"
        for s in sym_summary[:6]
    )
    lab_lines = "\n".join(
        f"  {k:45s}: {v['value']} {v['unit']} [{v['flag']}]  (ref: {v['reference_range']})"
        for k, v in lab.get("results", {}).items()
    ) if lab.get("results") else "  No lab results available."

    demo_analysis = f"""
SYMPTOM & LABORATORY ANALYSIS — {p['name']}
Sources: Patient mobile app (symptoms) | {lab.get('lab_name','N/A')} via FHIR R4 (labs)

SYMPTOM SUMMARY (7 days, {len(symptoms_raw)} reports):
{sym_lines if sym_lines else "  No significant symptoms reported"}

Patient Free-Text Notes:
{chr(10).join('  "' + t + '"' for t in free_texts[:3]) if free_texts else "  None"}

Adverse Events:
  ER visits: {len(er_visits)} | Falls: {len(falls)}
  {"FALL RISK ALERT: Fall reported — assess gait, medication side effects, home safety" if falls else ""}
  {"ER VISIT NOTED — clinical review of reason for visit required" if er_visits else ""}

LABORATORY RESULTS (ordered by {lab.get('ordered_by','N/A')}, reported {lab.get('result_date','N/A')}):
{lab_lines}

Lab Interpretation:
  HbA1c: {hba1c}% — {"CRITICALLY ELEVATED > 10%, urgent medication intensification" if hba1c and hba1c >= care_config.hba1c_critical else "ABOVE TARGET > 8%" if hba1c and hba1c >= care_config.hba1c_high else "At/near target" if hba1c else "Not available"}
  eGFR: {egfr} mL/min — {"CKD Stage 3b — nephrology referral appropriate" if egfr and egfr < 45 else "CKD Stage 3a — monitor quarterly" if egfr and egfr < 60 else "Adequate" if egfr else "Not available"}
  {"LDL " + str(ldl) + " mg/dL — above <70 target for high CV risk; statin dose/compliance review" if ldl and ldl > 100 else ""}
  {"Potassium " + str(potassium) + " mEq/L — borderline high on ACE inhibitor; recheck in 4 weeks" if potassium and potassium > 5.0 else ""}
  {"BNP " + str(bnp) + " pg/mL — ELEVATED, indicating active heart failure with fluid overload" if bnp and bnp > 200 else ""}

CONVERGENT SIGNAL ANALYSIS:
  {"Fatigue + shortness_of_breath + elevated BNP = HIGH probability heart failure decompensation" if bnp and bnp > 300 and any(s['symptom'] in ['shortness_of_breath','fatigue'] and s['max_severity'] >= 6 for s in sym_summary) else ""}
  {"Increased thirst + frequent urination + HbA1c > 8% = Poorly controlled diabetes driving osmotic symptoms" if hba1c and hba1c > 8 and any(s['symptom'] in ['increased_thirst','frequent_urination'] for s in sym_summary) else ""}
  {"Dizziness + irregular heartbeat detected = possible paroxysmal AF, warrants cardiac monitoring" if any(s['symptom']=='dizziness' for s in sym_summary) and any(r.get('irregular_heartbeat_detected') for r in state.get('bp_readings',[])) else ""}
"""

    state["symptom_analysis"] = call_llm(
        system_prompt="You are a clinical symptom and laboratory analysis agent.",
        user_prompt=f"Analyze: hba1c={hba1c}, egfr={egfr}, bnp={bnp}, er_visits={len(er_visits)}, falls={len(falls)}, top_symptom={sym_summary[0] if sym_summary else 'none'}",
        demo_response=demo_analysis,
    )

    risk = 20.0
    factors = []
    if er_visits: risk += 30; factors.append("Recent ER visit")
    if falls:     risk += 20; factors.append("Fall reported")
    if hba1c and hba1c >= care_config.hba1c_critical: risk += 20; factors.append(f"Critical HbA1c: {hba1c}%")
    elif hba1c and hba1c >= care_config.hba1c_high:   risk += 10; factors.append(f"Elevated HbA1c: {hba1c}%")
    if egfr and egfr < 30:   risk += 25; factors.append(f"Severe CKD: eGFR {egfr}")
    elif egfr and egfr < 45: risk += 12; factors.append(f"Moderate CKD: eGFR {egfr}")
    if bnp and bnp > 400:    risk += 25; factors.append(f"Elevated BNP: {bnp} pg/mL")
    high_sym = [s for s in sym_summary if s["max_severity"] >= 7]
    if high_sym:             risk += 10; factors.append(f"High-severity symptoms: {', '.join(s['symptom'] for s in high_sym[:3])}")

    state["risk_scores"].append({
        "domain": "symptoms_and_labs",
        "score": min(risk, 100),
        "risk_level": "critical" if risk>75 else "high" if risk>50 else "moderate" if risk>25 else "low",
        "factors": factors or ["No critical symptom or lab findings"],
        "trend": "stable",
    })

    state["current_agent"] = "intervention_agent"
    print(f"[SYMPTOM & LAB AGENT] HbA1c: {hba1c}% | eGFR: {egfr} | BNP: {bnp} | "
          f"ER: {len(er_visits)} | Falls: {len(falls)} | "
          f"Risk: {state['risk_scores'][-1]['risk_level'].upper()}")
    return state


# ==========================================================================
# INTERVENTION AGENT
# ==========================================================================

def run_intervention_agent(state: dict) -> dict:
    p = state["patient"]
    print(f"\n[INTERVENTION AGENT] Planning interventions for {p['name']}...")

    risk_scores   = state["risk_scores"]
    overall_score = max((r["score"] for r in risk_scores), default=0)

    # --- collect labs for thresholds ---
    lab           = state["lab_results"][0]   if state["lab_results"]   else {}
    bp_readings   = state["bp_readings"]
    glucose_rdgs  = state["glucose_readings"]

    hba1c    = lab.get("hba1c_percent")
    egfr     = lab.get("egfr_ml_min")
    ldl      = lab.get("ldl_mg_dl")
    potassium = lab.get("potassium_meq_l")
    bnp      = lab.get("bnp_pg_ml")

    avg_sys  = (sum(r["systolic_mmhg"] for r in bp_readings) / len(bp_readings)
                if bp_readings else 0)
    max_sys  = max((r["systolic_mmhg"] for r in bp_readings), default=0)
    irreg_bp = [r for r in bp_readings if r.get("irregular_heartbeat_detected")]

    hypo_events = [r for r in glucose_rdgs if r["glucose_mg_dl"] < 70]
    falls       = [r for r in state["symptom_reports"] if r.get("fell_today")]
    er_visits   = [r for r in state["symptom_reports"] if r.get("visited_er_since_last_check")]

    interventions = []
    care_plan     = []

    # --- EMERGENCY ---
    if any(r["glucose_mg_dl"] < care_config.glucose_critical_low for r in glucose_rdgs):
        interventions.append({
            "type": "emergency_alert",
            "severity": "emergency",
            "title": "CRITICAL HYPOGLYCEMIA DETECTED",
            "message": f"Glucose reading below 54 mg/dL detected. Emergency contact and physician notified.",
            "recipient": "emergency_services",
            "clinical_basis": f"CGM reading < {care_config.glucose_critical_low} mg/dL",
        })
        state["emergency_triggered"] = True

    if max_sys >= care_config.bp_systolic_critical_high:
        interventions.append({
            "type": "physician_alert",
            "severity": "urgent",
            "title": f"Hypertensive urgency: systolic {max_sys:.0f} mmHg",
            "message": f"BP reading of {max_sys:.0f} mmHg detected. Urgent physician review required.",
            "recipient": "physician",
            "clinical_basis": f"BP ≥ {care_config.bp_systolic_critical_high:.0f} mmHg",
        })

    # --- URGENT CLINICAL ALERTS ---
    if bnp and bnp > 400:
        interventions.append({
            "type": "physician_alert",
            "severity": "urgent",
            "title": f"Elevated BNP {bnp:.0f} pg/mL — possible HF decompensation",
            "message": (f"BNP of {bnp:.0f} pg/mL with symptoms of fatigue and dyspnea. "
                        "Consider diuretic dose adjustment and urgent cardiology review."),
            "recipient": "physician",
            "clinical_basis": f"BNP {bnp:.0f} pg/mL (threshold > 400)",
        })

    if falls:
        interventions.append({
            "type": "care_coordinator_alert",
            "severity": "warning",
            "title": "Fall reported — safety assessment needed",
            "message": "Patient reported a fall. Home safety assessment and medication review recommended.",
            "recipient": "care_coordinator",
            "clinical_basis": "Patient-reported fall in symptom app",
        })

    if irreg_bp:
        interventions.append({
            "type": "physician_alert",
            "severity": "warning",
            "title": f"Irregular heartbeat detected {len(irreg_bp)}x by BP monitor",
            "message": "Irregular rhythm detected on home BP monitor. 12-lead ECG recommended to rule out AF.",
            "recipient": "physician",
            "clinical_basis": "Omron irregular heartbeat flag",
        })

    # --- PATIENT NUDGES ---
    if hypo_events:
        interventions.append({
            "type": "patient_notification",
            "severity": "warning",
            "title": "Low blood sugar episodes detected",
            "message": (f"Your glucose sensor detected {len(hypo_events)} readings below 70 mg/dL. "
                        "Please carry fast-acting glucose tablets and review your meal timing with your care team."),
            "recipient": "patient",
            "clinical_basis": f"{len(hypo_events)} CGM readings < 70 mg/dL",
        })

    if avg_sys > care_config.bp_systolic_target_high:
        interventions.append({
            "type": "patient_notification",
            "severity": "info",
            "title": "Your blood pressure readings are running higher than your target",
            "message": (f"Your average BP this week is {avg_sys:.0f}/{sum(r['diastolic_mmhg'] for r in bp_readings)/len(bp_readings):.0f} mmHg. "
                        "Your target is below 130/80 mmHg. "
                        "Tips: reduce salt intake, limit caffeine, take your medications at the same time each day."),
            "recipient": "patient",
            "clinical_basis": f"Average systolic {avg_sys:.0f} mmHg > 130 target",
        })

    latest_vitals = state["vitals_readings"][0] if state["vitals_readings"] else {}
    if latest_vitals.get("steps_today", 99999) < 3000:
        interventions.append({
            "type": "lifestyle_coaching",
            "severity": "info",
            "title": "Let's get some steps in today",
            "message": (f"You've taken {latest_vitals.get('steps_today',0):,} steps today. "
                        "Even a gentle 10-minute walk after meals helps lower blood sugar and blood pressure. "
                        "Aim for 7,000 steps per day this week."),
            "recipient": "patient",
            "clinical_basis": f"Steps today: {latest_vitals.get('steps_today',0)}",
        })

    # Check medication adherence for patient nudge
    if state["medication_records"]:
        by_med_taken = defaultdict(lambda: {"total":0,"taken":0})
        for r in state["medication_records"]:
            by_med_taken[r["medication_name"]]["total"] += 1
            if r["taken"]: by_med_taken[r["medication_name"]]["taken"] += 1
        missed_meds = [
            med for med, s in by_med_taken.items()
            if s["total"] > 0 and s["taken"]/s["total"] < 0.75
        ]
        if missed_meds:
            interventions.append({
                "type": "medication_reminder",
                "severity": "warning",
                "title": f"Medication reminder: {', '.join(missed_meds[:2])}",
                "message": (f"You've missed some doses of {', '.join(missed_meds[:2])} recently. "
                            "Taking your medications consistently is the most important thing you can do "
                            "to protect your heart and kidneys. Need a reminder setup? Let us know."),
                "recipient": "patient",
                "clinical_basis": f"Adherence < 75% for: {', '.join(missed_meds)}",
            })

    # --- CARE PLAN RECOMMENDATIONS ---
    if hba1c and hba1c >= care_config.hba1c_high:
        care_plan.append({
            "change_type": "medication_adjustment_recommendation",
            "current_state": f"HbA1c {hba1c}% on current regimen",
            "recommendation": (
                f"HbA1c of {hba1c}% exceeds the < 7.0% target. "
                "Consider intensification: add GLP-1 receptor agonist (e.g., semaglutide) "
                "or SGLT2 inhibitor if not contraindicated by eGFR. "
                "Note: Metformin dose limit already at 1000mg BID."
            ),
            "urgency": "soon" if hba1c >= care_config.hba1c_critical else "routine",
            "guideline": "ADA Standards of Medical Care in Diabetes 2024",
        })

    if egfr and egfr < 45:
        care_plan.append({
            "change_type": "specialist_referral",
            "current_state": f"eGFR {egfr} mL/min — CKD Stage 3b",
            "recommendation": (
                "Nephrology referral for CKD Stage 3b management. "
                "Review Metformin safety (generally hold if eGFR < 30). "
                "SGLT2 inhibitor benefit demonstrated down to eGFR 20. "
                "Annual ACR, quarterly eGFR checks recommended."
            ),
            "urgency": "soon",
            "guideline": "KDIGO CKD Guidelines 2022",
        })

    if ldl and ldl > 100:
        care_plan.append({
            "change_type": "medication_adjustment_recommendation",
            "current_state": f"LDL {ldl} mg/dL on Atorvastatin 40mg",
            "recommendation": (
                f"LDL {ldl} mg/dL exceeds < 70 mg/dL target for high cardiovascular risk. "
                "Options: uptitrate to Atorvastatin 80mg, or add ezetimibe 10mg. "
                "Assess adherence to statin before dose change."
            ),
            "urgency": "routine",
            "guideline": "ACC/AHA Cholesterol Guidelines 2019",
        })

    if potassium and potassium > 5.0:
        care_plan.append({
            "change_type": "lab_order",
            "current_state": f"Potassium {potassium} mEq/L on Lisinopril",
            "recommendation": (
                f"Potassium {potassium} mEq/L — borderline hyperkalemia on ACE inhibitor. "
                "Repeat BMP in 1–2 weeks. If confirmed > 5.5, consider dose reduction or "
                "switch to ARB, and dietary potassium counseling."
            ),
            "urgency": "soon" if potassium > 5.5 else "routine",
            "guideline": "ACC/AHA Hypertension Guidelines 2017",
        })

    if avg_sys > care_config.bp_systolic_target_high:
        care_plan.append({
            "change_type": "medication_adjustment_recommendation",
            "current_state": f"Average BP {avg_sys:.0f} mmHg on Lisinopril 10mg + Amlodipine 5mg",
            "recommendation": (
                f"BP consistently above 130/80 target (avg {avg_sys:.0f} mmHg). "
                "Consider: (1) uptitrate Lisinopril 10mg -> 20mg, "
                "(2) uptitrate Amlodipine 5mg -> 10mg, "
                "or (3) add low-dose thiazide diuretic (if not already on)."
            ),
            "urgency": "soon",
            "guideline": "ACC/AHA + ADA joint HTN guidelines",
        })

    state["interventions"]         = interventions
    state["care_plan_adjustments"] = care_plan
    state["current_agent"]         = "supervisor"

    print(f"[INTERVENTION AGENT] {len(interventions)} interventions | "
          f"{len(care_plan)} care plan recommendations | "
          f"Emergency: {state['emergency_triggered']}")
    return state
