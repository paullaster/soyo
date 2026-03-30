import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with a Kenyan locale
fake = Faker(['en_KE'])
Faker.seed(42)
np.random.seed(42)

# Constants
NUM_SUPPLIERS = 5000
NUM_TENDERS = 50000
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 3, 31)

KENYAN_COMPANY_SUFFIXES = ['Limited', 'PLC', 'Enterprises', 'Ventures', 'Group', 'Solutions', 'Holdings', 'Investments', 'Agencies']

CATEGORIES = [
    'Construction & Civil Works', 
    'ICT Equipment & Software', 
    'Office Stationery & Supplies', 
    'Medical Supplies & Pharmaceuticals', 
    'Consultancy Services',
    'Cleaning & Sanitary Services',
    'Motor Vehicles & Spare Parts'
]

def generate_suppliers(n):
    suppliers = []
    for i in range(n):
        incorporation_date = fake.date_between(start_date='-10y', end_date='today')
        company_base = fake.company().split(' ')[0]
        name = f"{company_base} {random.choice(KENYAN_COMPANY_SUFFIXES)}"
        
        suppliers.append({
            'supplier_id': i + 1,
            'supplier_name': name,
            'kra_pin': f"A{fake.random_number(digits=9, fix_len=True)}{fake.random_letter().upper()}",
            'incorporation_date': incorporation_date,
            'location': fake.city(),
            'employee_count': np.random.randint(5, 1000),
            'credit_score': np.random.randint(300, 850)
        })
    return pd.DataFrame(suppliers)

def generate_tenders(n):
    tenders = []
    entities = [
        'Ministry of Health', 'Ministry of Education', 'Ministry of Transport', 
        'County Government of Nairobi', 'County Government of Mombasa', 
        'Kenya Revenue Authority', 'Kenya Power', 'KenGen', 'ICT Authority'
    ]

    for i in range(n):
        category = np.random.choice(CATEGORIES)
        if category == 'Construction & Civil Works':
            budget = np.random.randint(5_000_000, 500_000_000)
        elif category == 'ICT Equipment & Software':
            budget = np.random.randint(500_000, 50_000_000)
        else:
            budget = np.random.randint(100_000, 10_000_000)
            
        tenders.append({
            'tender_id': i + 1,
            'category': category,
            'tender_budget_kes': budget,
            'tender_open_date': fake.date_between(start_date=START_DATE, end_date=END_DATE),
            'tender_description': f"Supply and delivery of {category.lower()} items",
            'procuring_entity': np.random.choice(entities)
        })
    return pd.DataFrame(tenders)

def generate_performance_data(suppliers_df, tenders_df):
    contracts = []
    tender_ids = tenders_df['tender_id'].tolist()
    
    for tender_id in tender_ids:
        supplier = suppliers_df.sample(1).iloc[0]
        tender = tenders_df[tenders_df['tender_id'] == tender_id].iloc[0]
        
        supplier_age_days = (pd.to_datetime(tender['tender_open_date']).date() - supplier['incorporation_date']).days
        
        # 1. Calculate Static Capacity
        capacity_kes = supplier['employee_count'] * 5_000_000
        
        # 2. Simulate "Pre-existing Workload" at the time of award
        # Most suppliers are already busy. Randomly assign 0-120% utilization.
        active_workload_kes = capacity_kes * np.random.uniform(0.1, 1.2)
        
        # 3. New project impact
        total_load_with_new = active_workload_kes + tender['tender_budget_kes']
        utilization_ratio = total_load_with_new / capacity_kes
        
        # --- RISK MODELLING ---
        # Base delay from budget complexity
        complexity_delay = min(15, (tender['tender_budget_kes'] / 500_000_000) * 15)
        
        # WORKLOAD IMPACT (Non-linear)
        # 0.0 - 0.7: Stable
        # 0.7 - 1.0: Growing Stress (0.5 to 1.5x delay)
        # > 1.0: Exponential Breakdown
        if utilization_ratio < 0.7:
            workload_delay = 0
        elif utilization_ratio < 1.0:
            workload_delay = (utilization_ratio - 0.7) * 30 
        else:
            workload_delay = (utilization_ratio - 1.0) * 100 + 30

        days_delayed = max(0, int(np.random.normal(complexity_delay + workload_delay, 3)))
        cost_overrun_pct = max(0, np.random.normal(utilization_ratio * 5, 2))
        
        # Termination risk spikes at high utilization
        contract_status = 'Completed'
        if utilization_ratio > 1.2 and np.random.random() < 0.15:
            contract_status = 'Terminated'
        
        # Financial Health penalty
        if supplier['credit_score'] < 500:
            days_delayed += np.random.randint(15, 60)

        contracts.append({
            'contract_id': f"C-{tender_id}",
            'tender_id': tender_id,
            'supplier_id': supplier['supplier_id'],
            'award_amount_kes': tender['tender_budget_kes'] * np.random.uniform(0.98, 1.05),
            'contract_start_date': tender['tender_open_date'],
            'days_delayed': days_delayed,
            'cost_overrun_percentage': round(cost_overrun_pct, 2),
            'contract_status': contract_status,
            'supplier_age_at_award_days': supplier_age_days,
            'active_workload_kes': active_workload_kes,
            'utilization_ratio': utilization_ratio
        })
        
    return pd.DataFrame(contracts)

if __name__ == "__main__":
    print("Generating Kenyan Suppliers...")
    df_suppliers = generate_suppliers(NUM_SUPPLIERS)
    print("Generating Tenders...")
    df_tenders = generate_tenders(NUM_TENDERS)
    print("Generating Workload-Aware Performance Data...")
    df_contracts = generate_performance_data(df_suppliers, df_tenders)
    
    df_master = df_contracts.merge(df_suppliers, on='supplier_id').merge(df_tenders, on='tender_id')
    df_master.to_csv('data/procurement_master_dataset.csv', index=False)
    print(f"Done! Created dataset with Active Workload and Utilization Ratio features.")
