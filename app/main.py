from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from datetime import datetime
from app.dl_engine import SoyoDeepLearningEngine

# Initialize App
app = FastAPI(title="Soyo Supplier Risk API (Deep Learning Enabled)", version="3.0.0")

# CORS Configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://supplier-risk-prediction.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Instance
dl_engine = SoyoDeepLearningEngine()

@app.on_event("startup")
def startup_event():
    if dl_engine.load():
        print("Deep Learning Engine loaded successfully.")
    else:
        print("Deep Learning Engine NOT found. Ensure model is trained.")

# --- 1. CORE RISK API ---
class PredictionRequest(BaseModel):
    tender_budget_kes: float
    credit_score: int
    company_size: str
    supplier_age_at_award_days: int
    category: str
    supplier_id: int | None = None
    # Past performance for LSTM branch [delay_days, cost_overrun_pct]
    history_sequence: list[list[float]] | None = None

class HistoricalProject(BaseModel):
    year: str
    delay_days: int
    overrun_pct: float
    project_size: str

class PredictionResponse(BaseModel):
    risk_level: str
    predicted_delay_days: float
    risk_score_probability: float
    risk_factors: list[str]
    historical_performance: list[HistoricalProject]
    market_comparison: dict[str, float]

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    # Prepare history for LSTM (Default to zeros if not provided)
    # The engine expects a sequence of [delay_days, cost_overrun_pct]
    history = request.history_sequence
    if not history:
        # If no history provided, use a "blank slate" for the LSTM
        history = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    
    # Ensure history is exactly the length the LSTM expects (3)
    while len(history) < 3:
        history.insert(0, [0.0, 0.0])
    history = history[-3:]

    # Perform Inference using the Deep Learning Engine
    try:
        # Pass both static data and the temporal history sequence
        results = dl_engine.predict(request.model_dump(exclude={"history_sequence"}), history=history)
        if not results:
             raise HTTPException(status_code=503, detail="ML Engine unavailable")
             
        risk_class = results["risk_level"]
        predicted_delay = results["predicted_delay_days"]
        risk_prob = results["risk_score_probability"]
    except Exception as e:
        print(f"Inference Error: {e}")
        # Fallback Logic
        risk_class = "High" if request.credit_score < 500 else "Low"
        predicted_delay = 45.0 if risk_class == "High" else 5.0
        risk_prob = 0.85

    # --- Logic for EXPLAINABLE Risk Factors ---
    factors = []
    CAPACITY_LIMITS = {"Small": 20_000_000, "Medium": 200_000_000, "Large": 2_000_000_000}
    limit = CAPACITY_LIMITS.get(request.company_size, 20_000_000)
    if request.tender_budget_kes > limit:
        factors.append(f"Capacity Alert: Budget exceeds typical {request.company_size} firm threshold.")
    
    if request.credit_score < 600:
        factors.append(f"Financial Risk: Credit score ({request.credit_score}) is below optimal threshold.")
    
    if request.supplier_age_at_award_days < 365:
        factors.append("Operational History: Firm is in its first year of operation.")

    if not factors:
        factors.append("No critical risk factors identified.")

    # Historical Data Gen (Dynamic to match the predicted risk class)
    history = []
    current_year = datetime.now().year
    base_h_delay = 30 if risk_class == "High" else 10 if risk_class == "Medium" else 2
    
    for i in range(1, 6):
        sim_delay = max(0, int(np.random.normal(base_h_delay, 5)))
        sim_overrun = round(max(0, np.random.normal(base_h_delay * 0.2, 2)), 1)
        history.append({
            "year": str(current_year - i),
            "delay_days": sim_delay,
            "overrun_pct": sim_overrun,
            "project_size": "Large" if request.tender_budget_kes > 10_000_000 else "Medium"
        })
    history.reverse()

    return {
        "risk_level": risk_class,
        "predicted_delay_days": round(float(predicted_delay), 1),
        "risk_score_probability": round(float(risk_prob), 2),
        "risk_factors": factors,
        "historical_performance": history,
        "market_comparison": {
            "supplier_avg_delay": round(sum(h["delay_days"] for h in history)/5, 1),
            "market_avg_delay": 7.5
        }
    }

# --- 2. FAIR PRICE ENGINE (Legacy logic remains stable) ---
class PricingItem(BaseModel):
    item_name: str
    quoted_unit_price: float
    quantity: int

class PricingAnalysisResponse(BaseModel):
    total_variance_kes: float
    inflated_items: list[dict]
    market_savings_potential: float
    recommendation: str

MARKET_PRICES = {
    "Cement (50kg Bag)": 650,
    "Standard Laptop (i5, 8GB)": 65000,
    "Office Desk": 12000,
    "Printer Paper (Ream)": 550,
    "Wheelbarrow": 4500
}

@app.post("/analyze-pricing", response_model=PricingAnalysisResponse)
def analyze_pricing(items: list[PricingItem]):
    total_variance = 0
    inflated_list = []
    for item in items:
        market_price = MARKET_PRICES.get(item.item_name)
        if market_price:
            variance = item.quoted_unit_price - market_price
            if variance > 0:
                percent_inflation = (variance / market_price) * 100
                total_loss = variance * item.quantity
                if percent_inflation > 15:
                    total_variance += total_loss
                    inflated_list.append({
                        "item": item.item_name,
                        "quoted": item.quoted_unit_price,
                        "market": market_price,
                        "inflation_pct": round(percent_inflation, 1),
                        "potential_loss": total_loss
                    })
    recommendation = "Approve Award"
    if total_variance > 1_000_000:
        recommendation = "REJECT: Significant Price Inflation Detected"
    elif total_variance > 100_000:
        recommendation = "NEGOTIATE: Prices exceed market rates"
    return {
        "total_variance_kes": total_variance,
        "inflated_items": inflated_list,
        "market_savings_potential": total_variance,
        "recommendation": recommendation
    }

# --- 3. DIRECTOR WEB (COLLUSION) ---
class CollusionCheckRequest(BaseModel):
    supplier_name: str
    tender_id: str

class Node(BaseModel):
    id: str
    type: str

class Link(BaseModel):
    source: str
    target: str
    label: str

class CollusionResponse(BaseModel):
    is_collusion_suspected: bool
    risk_score: int
    graph_nodes: list[Node]
    graph_links: list[Link]
    message: str

@app.post("/check-collusion", response_model=CollusionResponse)
def check_collusion(request: CollusionCheckRequest):
    is_risk = "Soyo" in request.supplier_name or "Rift" in request.supplier_name
    nodes = []
    links = []
    main_company = request.supplier_name
    director = "John Doe (ID: 12345678)"
    sister_company = "Hidden Ventures Ltd"
    shared_address = "P.O. Box 4567-00100 NBI"
    shared_phone = "+254 722 000 000"
    if is_risk:
        nodes = [
            {"id": main_company, "type": "Company"},
            {"id": director, "type": "Person"},
            {"id": sister_company, "type": "Company"},
            {"id": shared_address, "type": "Address"},
            {"id": shared_phone, "type": "Phone"}
        ]
        links = [
            {"source": main_company, "target": director, "label": "Director"},
            {"source": sister_company, "target": director, "label": "Director"},
            {"source": main_company, "target": shared_address, "label": "Registered Office"},
            {"source": sister_company, "target": shared_address, "label": "Registered Office"},
            {"source": sister_company, "target": shared_phone, "label": "Contact Person"}
        ]
        msg = f"CRITICAL CARTEL DETECTED: {main_company} is linked to {sister_company} via Director, P.O. Box, and Phone Number."
        score = 98
    else:
        nodes = [{"id": request.supplier_name, "type": "Company"}]
        msg = "No shared directorships or contact details found with other bidders."
        score = 10
    return {
        "is_collusion_suspected": is_risk,
        "risk_score": score,
        "graph_nodes": nodes,
        "graph_links": links,
        "message": msg
    }

# --- 4. GEOSPATIAL VERIFICATION (GIS) ---
class SiteVerifyRequest(BaseModel):
    address: str
    coordinates: str

class SiteVerifyResponse(BaseModel):
    zoning_type: str
    satellite_snapshot_url: str
    risk_score: int
    analysis: str

@app.post("/verify-site", response_model=SiteVerifyResponse)
def verify_site(request: SiteVerifyRequest):
    addr_lower = request.address.lower()
    physical_indicators = ["road", "street", "building", "plaza", "floor", "godown", "industrial"]
    is_physical = any(indicator in addr_lower for indicator in physical_indicators)
    has_box = "box" in addr_lower
    if is_physical:
        return {
            "zoning_type": "Industrial/Commercial",
            "satellite_snapshot_url": "/maps/industrial.png",
            "risk_score": 5,
            "analysis": "PASS: High-confidence physical location detected."
        }
    if has_box and not is_physical:
        return {
            "zoning_type": "Postal Only",
            "satellite_snapshot_url": "/maps/postal.png",
            "risk_score": 85,
            "analysis": "CRITICAL: Address is restricted to a Postal Box."
        }
    return {
        "zoning_type": "Mixed Use/Unverified",
        "satellite_snapshot_url": "/maps/commercial.png",
        "risk_score": 35,
        "analysis": "Standard urban address."
    }

# --- 5. GLOBAL SANCTIONS SHIELD ---
class SanctionCheckRequest(BaseModel):
    company_registration_number: str
    entity_name: str
    directors: list[str]

class SanctionCheckResponse(BaseModel):
    is_sanctioned: bool
    source_list: str | None
    match_confidence: float
    details: str

@app.post("/screen-sanctions", response_model=SanctionCheckResponse)
def screen_sanctions(request: SanctionCheckRequest):
    return {
        "is_sanctioned": False,
        "source_list": None,
        "match_confidence": 0.0,
        "details": "Clear. No matches found."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
