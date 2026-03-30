import json
import random
import argparse
import os
from faker import Faker
from datetime import datetime, timedelta

fake = Faker(['en_KE'])

# Kenyan-specific company suffixes
KENYAN_COMPANY_SUFFIXES = ['Limited', 'PLC', 'Enterprises', 'Ventures', 'Group', 'Solutions', 'Holdings', 'Investments', 'Agencies']

# Categories typical in Kenyan procurement
CATEGORIES = [
    'Construction & Civil Works', 
    'ICT Equipment & Software', 
    'Office Stationery & Supplies', 
    'Medical Supplies & Pharmaceuticals', 
    'Consultancy Services',
    'Cleaning & Sanitary Services',
    'Motor Vehicles & Spare Parts'
]

def generate_data(num_tenders, num_suppliers, proposals_per_tender):
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 1. Generate Suppliers
    suppliers = []
    supplier_ids = [f"S-{str(i+1).zfill(3)}" for i in range(num_suppliers)]
    
    for s_id in supplier_ids:
        company_base = fake.company().split(' ')[0]
        company_name = f"{company_base} {random.choice(KENYAN_COMPANY_SUFFIXES)}"
        
        suppliers.append({
            "id": s_id,
            "name": company_name,
            "registration_number": f"CPR/{fake.bothify(text='####/######')}",
            "credit_score": random.randint(300, 850),
            "employee_count": random.randint(5, 1000), # Numerical focus
            "age_days": random.randint(365, 5000),
            "address": f"P.O. Box {random.randint(100, 99999)}, {fake.city()}",
            "phone": f"+254 {fake.msisdn()[3:]}",
            "directors": [
                {"name": fake.name(), "national_id": str(fake.random_number(digits=8, fix_len=True))}
                for _ in range(random.randint(1, 3))
            ],
            "history_sequence": [
                [float(random.randint(0, 30)), round(random.uniform(0, 15), 2)]
                for _ in range(3)
            ],
            "past_contracts": [
                {
                    "project_name": f"Provision of {fake.word().capitalize()} Services",
                    "start_date": (datetime.now() - timedelta(days=random.randint(400, 800))).strftime("%Y-%m-%d"),
                    "planned_end": (datetime.now() - timedelta(days=random.randint(200, 350))).strftime("%Y-%m-%d"),
                    "actual_end": (datetime.now() - timedelta(days=random.randint(100, 190))).strftime("%Y-%m-%d"),
                    "status": "Completed",
                    "delay_days": random.randint(0, 45)
                } for _ in range(2)
            ]
        })

    # 2. Generate Tenders
    tenders = []
    tender_ids = [f"T-{str(i+1).zfill(3)}" for i in range(num_tenders)]
    for t_id in tender_ids:
        tenders.append({
            "id": t_id,
            "title": f"Procurement of {fake.bs().title()}",
            "category": random.choice(CATEGORIES),
            "budget_kes": float(random.randint(500_000, 100_000_000)),
            "deadline_days": random.randint(7, 30),
            "anticipated_closure_date": (datetime.now() + timedelta(days=random.randint(30, 90))).strftime("%Y-%m-%d")
        })

    # 3. Generate Proposals
    proposals = []
    for t_id in tender_ids:
        bidders = random.sample(supplier_ids, min(proposals_per_tender, len(supplier_ids)))
        for s_id in bidders:
            proposals.append({
                "id": f"P-{fake.bothify(text='####-####')}",
                "tender_id": t_id,
                "supplier_id": s_id,
                "bid_amount": float(random.randint(400_000, 110_000_000)),
                "proposal_date": (datetime.now() - timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d")
            })

    # 4. Generate Awards
    awards = []
    for t_id in tender_ids:
        tender_proposals = [p for p in proposals if p["tender_id"] == t_id]
        if tender_proposals:
            winner = random.choice(tender_proposals)
            awards.append({
                "id": f"A-{fake.bothify(text='####-####')}",
                "tender_id": t_id,
                "supplier_id": winner["supplier_id"],
                "award_amount": winner["bid_amount"],
                "award_date": datetime.now().strftime("%Y-%m-%d")
            })

    with open(os.path.join(data_dir, "suppliers.json"), "w") as f: json.dump(suppliers, f, indent=2)
    with open(os.path.join(data_dir, "tenders.json"), "w") as f: json.dump(tenders, f, indent=2)
    with open(os.path.join(data_dir, "proposals.json"), "w") as f: json.dump(proposals, f, indent=2)
    with open(os.path.join(data_dir, "awards.json"), "w") as f: json.dump(awards, f, indent=2)

    print(f"Generated data with exact employee counts for African context.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenders", type=int, default=10)
    parser.add_argument("--suppliers", type=int, default=50)
    parser.add_argument("--proposals", type=int, default=5)
    args = parser.parse_args()
    generate_data(args.tenders, args.suppliers, args.proposals)
