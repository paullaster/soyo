from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Optional
from app.dl_engine import SoyoDeepLearningEngine

# --- DATA HELPERS ---
DATA_DIR = "data"
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return []
    try:
        with open(path, "r") as f: return json.load(f)
    except: return []

def save_json(filename, data):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f: json.dump(data, f, indent=2)

# --- MODELS ---
class Director(BaseModel):
    name: str
    national_id: str

class PastContract(BaseModel):
    project_name: str
    start_date: str
    planned_end: str
    actual_end: str
    status: str
    delay_days: int

class Tender(BaseModel):
    id: str
    title: str
    category: str
    budget_kes: float
    deadline_days: int
    anticipated_closure_date: str

class Supplier(BaseModel):
    id: str
    name: str
    registration_number: str
    credit_score: int
    employee_count: int
    age_days: int
    address: str
    phone: str
    directors: List[Director]
    history_sequence: List[List[float]]
    past_contracts: Optional[List[PastContract]] = []

class PredictionRequest(BaseModel):
    tender_id: str
    supplier_id: str

class AnalysisResult(BaseModel):
    tender_id: str
    supplier_id: str
    supplier_name: str
    risk_level: str
    risk_score_probability: float
    predicted_delay_days: float
    timestamp: str
    workload_status: str
    utilization_ratio: float
    conflict_found: bool

# Initialize App
app = FastAPI(title="Soyo Audit API", version="5.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dl_engine = SoyoDeepLearningEngine()

@app.get("/tenders", response_model=List[Tender])
def get_tenders(): return load_json("tenders.json")

@app.get("/suppliers", response_model=List[Supplier])
def get_suppliers(tender_id: Optional[str] = None):
    suppliers = load_json("suppliers.json")
    if tender_id:
        proposals = load_json("proposals.json")
        bidding_supplier_ids = {p["supplier_id"] for p in proposals if p["tender_id"] == tender_id}
        return [s for s in suppliers if s["id"] in bidding_supplier_ids]
    return suppliers

@app.get("/analysis-comparison/{tender_id}")
def get_comparison(tender_id: str):
    results = load_json("analysis_results.json")
    return [r for r in results if r["tender_id"] == tender_id]

@app.post("/predict")
def predict_risk(request: PredictionRequest):
    tenders = load_json("tenders.json")
    suppliers = load_json("suppliers.json")
    awards = load_json("awards.json")
    proposals = load_json("proposals.json")
    
    tender = next((t for t in tenders if t["id"] == request.tender_id), None)
    supplier = next((s for s in suppliers if s["id"] == request.supplier_id), None)
    
    if not tender or not supplier:
        raise HTTPException(status_code=404, detail="Context not found")

    # 1. WORKLOAD CHECK
    active_awards = [a for a in awards if a["supplier_id"] == supplier["id"]]
    total_active_load = sum(a["award_amount"] for a in active_awards)
    
    # 2. CALC UTILIZATION RATIO
    capacity_kes = supplier["employee_count"] * 5_000_000
    utilization_ratio = (total_active_load + tender["budget_kes"]) / capacity_kes

    # 3. CONCONCLUSIVE COLLUSION CHECK
    other_bidder_ids = [p["supplier_id"] for p in proposals if p["tender_id"] == tender["id"] and p["supplier_id"] != supplier["id"]]
    other_suppliers = [s for s in suppliers if s["id"] in other_bidder_ids]
    
    current_ids = {d["national_id"] for d in supplier["directors"]}
    conflicts = []
    for other in other_suppliers:
        other_ids = {d["national_id"] for d in other["directors"]}
        shared_ids = current_ids & other_ids
        if shared_ids:
            shared_names = [d["name"] for d in other["directors"] if d["national_id"] in shared_ids]
            conflicts.append({"partner": other["name"], "directors": list(shared_names)})

    # 4. VERDICT LOGIC
    history = supplier.get("history_sequence", [])
    results = dl_engine.predict({
        "tender_budget_kes": tender["budget_kes"],
        "credit_score": supplier["credit_score"],
        "employee_count": supplier["employee_count"],
        "supplier_age_at_award_days": supplier["age_days"],
        "category": tender["category"],
        "active_workload_kes": total_active_load,
        "utilization_ratio": utilization_ratio
    }, history=history)
    
    if results is None:
        raise HTTPException(status_code=500, detail=f"Deep Learning Model not found at {dl_engine.model_path}. Prediction aborted.")

    risk_class = results["risk_level"]
    predicted_delay = results["predicted_delay_days"]
    risk_prob = results["risk_score_probability"]
    ai_factors = results.get("ai_factors", [])

    # Final combined factors (AI Evidence + System Evidence)
    factors = ai_factors.copy()
    if utilization_ratio > 1.0: 
        factors.append(f"Workload Stress: Active projects (KES {total_active_load:,.0f}) exceed staff capacity.")
    if conflicts: 
        factors.append(f"Cartel Risk: Shared owner '{conflicts[0]['directors'][0]}' identified in competitor '{conflicts[0]['partner']}'.")

    analysis_entry = {
        "tender_id": tender["id"],
        "supplier_id": supplier["id"],
        "supplier_name": supplier["name"],
        "risk_level": risk_class,
        "risk_score_probability": float(risk_prob),
        "predicted_delay_days": float(predicted_delay),
        "timestamp": datetime.now().isoformat(),
        "workload_status": "Overloaded" if utilization_ratio > 1.0 else "Stable",
        "utilization_ratio": float(utilization_ratio),
        "conflict_found": len(conflicts) > 0
    }
    
    all_results = load_json("analysis_results.json")
    existing_idx = next((i for i, r in enumerate(all_results) if r["tender_id"] == tender["id"] and r["supplier_id"] == supplier["id"]), -1)
    if existing_idx > -1: all_results[existing_idx] = analysis_entry
    else: all_results.append(analysis_entry)
    save_json("analysis_results.json", all_results)

    return {
        **analysis_entry,
        "risk_factors": factors if factors else ["Everything looks good with this supplier profile."],
        "active_workload_kes": total_active_load,
        "conflicts": conflicts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
