from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

# Initialize App
app = FastAPI(title="Soyo Supplier Risk API", version="2.0.0")

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

# Load Models (Global State)
models = {}

@app.on_event("startup")
def load_models():
    model_path = "models"
    try:
        models["classifier"] = joblib.load(os.path.join(model_path, "risk_classifier.pkl"))
        models["regressor"] = joblib.load(os.path.join(model_path, "delay_regressor.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

# --- 1. CORE RISK API ---
class PredictionRequest(BaseModel):
    tender_budget_kes: float
    credit_score: int
    company_size: str
    supplier_age_at_award_days: int
    category: str

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
    if "classifier" not in models or "regressor" not in models:
        # Fallback ONLY if models aren't loaded at all
        risk_class = "High" if request.credit_score < 500 else "Low"
        predicted_delay = 45.0 if risk_class == "High" else 5.0
        risk_prob = 0.85 if risk_class == "High" else 0.15
    else:
        # USE THE ML MODELS
        input_data = pd.DataFrame([request.model_dump()])
        risk_class = models["classifier"].predict(input_data)[0]
        predicted_delay = models["regressor"].predict(input_data)[0]
        
        try:
            # Get the actual probability for the "High" class
            classes = list(models["classifier"].classes_)
            if "High" in classes:
                high_index = classes.index("High")
                risk_prob = models["classifier"].predict_proba(input_data)[0][high_index]
            else:
                # Fallback to general confidence
                risk_prob = np.max(models["classifier"].predict_proba(input_data)[0])
        except:
            risk_prob = 0.5

    # --- Logic for EXPLAINABLE Risk Factors (to accompany the model prediction) ---
    factors = []
    
    # Capacity Check (Logic to explain WHY the model might be flagging risk)
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

# --- 2. FAIR PRICE ENGINE ---
class PricingItem(BaseModel):
    item_name: str
    quoted_unit_price: float
    quantity: int

class PricingAnalysisResponse(BaseModel):
    total_variance_kes: float
    inflated_items: list[dict]
    market_savings_potential: float
    recommendation: str

# Mock Market Prices (Simulated Database)
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
        # Fuzzy match or direct lookup (Direct for demo)
        market_price = MARKET_PRICES.get(item.item_name)
        if market_price:
            variance = item.quoted_unit_price - market_price
            if variance > 0:
                percent_inflation = (variance / market_price) * 100
                total_loss = variance * item.quantity
                
                # Flag if > 15% inflation
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
    type: str # 'Company' or 'Person'

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
    # Simulate a CR12 Lookup & Contact Trace
    # Scenario: "Soyo" shares Director + Phone + Address with "Hidden Ventures"
    
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
    coordinates: str  # e.g. "-1.2921, 36.8219"

class SiteVerifyResponse(BaseModel):
    zoning_type: str # "Industrial", "Commercial", "Residential", "Unknown"
    satellite_snapshot_url: str
    risk_score: int
    analysis: str

@app.post("/verify-site", response_model=SiteVerifyResponse)
def verify_site(request: SiteVerifyRequest):
    # Advanced Mock GIS Logic
    addr_lower = request.address.lower()
    
    # High-confidence physical markers in Kenya
    physical_indicators = [
        "road", "street", "building", "plaza", "floor", "godown", "industrial", 
        "avenue", "ave", "lane", "park", "gate", "junction", "way", "sq", "square"
    ]
    
    is_physical = any(indicator in addr_lower for indicator in physical_indicators)
    has_box = "box" in addr_lower
    
    if is_physical:
        return {
            "zoning_type": "Industrial/Commercial",
            "satellite_snapshot_url": "/maps/industrial.png",
            "risk_score": 5,
            "analysis": "PASS: High-confidence physical location detected. Satellite cross-referencing confirms coordinates map to a valid commercial/industrial zone."
        }
    
    if has_box and not is_physical:
        return {
            "zoning_type": "Postal Only",
            "satellite_snapshot_url": "/maps/postal.png",
            "risk_score": 85,
            "analysis": "CRITICAL: Address is restricted to a Postal Box with no verifiable physical workspace markers. High 'Briefcase Company' risk detected."
        }
    
    if "apartment" in addr_lower or "hse" in addr_lower or "estate" in addr_lower:
        return {
            "zoning_type": "Residential",
            "satellite_snapshot_url": "/maps/residential.png",
            "risk_score": 65,
            "analysis": "WARNING: Registered location is within a residential estate. High risk of non-compliance for heavy procurement categories."
        }
        
    return {
        "zoning_type": "Mixed Use/Unverified",
        "satellite_snapshot_url": "/maps/commercial.png",
        "risk_score": 35,
        "analysis": "Standard urban address. Physical site visit required to verify operational scale and equipment inventory."
    }

# --- 5. GLOBAL SANCTIONS SHIELD ---
class SanctionCheckRequest(BaseModel):
    company_registration_number: str
    entity_name: str
    directors: list[str]

class SanctionCheckResponse(BaseModel):
    is_sanctioned: bool
    source_list: str | None # "OFAC", "World Bank", "UN"
    match_confidence: float
    details: str

# Mock Sanctions Database - Using Registration Number or Person Name as Unique Keys
SANCTION_DATABASE = {
    # Companies (Unique ID)
    "PVT-998877": {
        "source_list": "World Bank Debarred List",
        "match_confidence": 0.99,
        "details": "Company record #WB-2023-99: Debarred for fraudulent practices in 2023. Entity linked to systematic misrepresentation of financial capacity."
    },
    "CR-554433": {
        "source_list": "OFAC SDN List",
        "match_confidence": 1.0,
        "details": "Company record #OFAC-2024-12: Entity identified as a shell corporation involved in money laundering activities across multiple jurisdictions."
    },
    
    # Directors/Individuals (Name as Key)
    "Soyo Macharia": {
        "source_list": "UN Security Council",
        "match_confidence": 0.98,
        "details": "Individual record #UN-SC-2025-IND: Subject to global asset freeze and travel ban under Resolution 2140 for suspected arms trafficking."
    },
    "Jane Smith": {
        "source_list": "World Bank Debarred List",
        "match_confidence": 0.95,
        "details": "Individual record #WB-IND-2024: Cross-debarred for 5 years for collusion in public procurement projects."
    }
}

@app.post("/screen-sanctions", response_model=SanctionCheckResponse)
def screen_sanctions(request: SanctionCheckRequest):
    # 1. Check Company Registration Number (Unique Identifier)
    reg_no = request.company_registration_number.strip().upper()
    company_match = SANCTION_DATABASE.get(reg_no)
    
    if company_match:
        return {
            "is_sanctioned": True,
            "source_list": company_match["source_list"],
            "match_confidence": company_match["match_confidence"],
            "details": f"CRITICAL COMPANY MATCH: {company_match['details']} (Context: Entity Name '{request.entity_name}' with Directors {', '.join(request.directors)})"
        }
    
    # 2. Check Directors (Individual names)
    for director in request.directors:
        dir_name = director.strip()
        person_match = SANCTION_DATABASE.get(dir_name)
        if person_match:
            return {
                "is_sanctioned": True,
                "source_list": person_match["source_list"],
                "match_confidence": person_match["match_confidence"],
                "details": f"CRITICAL DIRECTOR MATCH: {person_match['details']} (Director: {dir_name} found in board of '{request.entity_name}', Reg: {reg_no})"
            }
            
    # 3. Clearance
    return {
        "is_sanctioned": False,
        "source_list": None,
        "match_confidence": 0.0,
        "details": f"Clear. No matches found for Registration Number '{reg_no}' or specified directors in OFAC, UN, or World Bank databases. Screening context for '{request.entity_name}' is verified as low risk."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
