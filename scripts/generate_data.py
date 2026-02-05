import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker with a seed for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)

# Constants
NUM_SUPPLIERS = 1000
NUM_TENDERS = 5000 # Increased for better learning
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2025, 12, 31)

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

def generate_suppliers(n):
    suppliers = []
    for i in range(n):
        incorporation_date = fake.date_between(start_date='-10y', end_date='today')
        
        # Simulate different company sizes/types
        company_type = np.random.choice(['Ltd', 'Limited', 'Enterprise', 'Ventures', 'Group', 'Agencies'])
        name = f"{fake.last_name()} {fake.word().capitalize()} {company_type}"
        
        suppliers.append({
            'supplier_id': i + 1,
            'supplier_name': name,
            'kra_pin': f"P{fake.random_number(digits=9, fix_len=True)}{fake.random_letter().upper()}",
            'incorporation_date': incorporation_date,
            'location': fake.city(),
            'company_size': np.random.choice(['Small', 'Medium', 'Large'], p=[0.6, 0.3, 0.1]),
            'credit_score': np.random.randint(300, 850) # Random baseline
        })
    return pd.DataFrame(suppliers)

def generate_tenders(n):
    tenders = []
    for i in range(n):
        category = np.random.choice(CATEGORIES)
        
        # Budget varies by category
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
            'procuring_entity': f"Ministry of {fake.word().capitalize()}"
        })
    return pd.DataFrame(tenders)

def generate_performance_data(suppliers_df, tenders_df):
    contracts = []
    
    # CAPACITY THRESHOLDS
    CAPACITY_LIMITS = {
        'Small': 20_000_000,
        'Medium': 200_000_000,
        'Large': 2_000_000_000
    }
    
    tender_ids = tenders_df['tender_id'].tolist()
    
    for tender_id in tender_ids:
        supplier = suppliers_df.sample(1).iloc[0]
        tender = tenders_df[tenders_df['tender_id'] == tender_id].iloc[0]
        
        supplier_age_days = (pd.to_datetime(tender['tender_open_date']).date() - supplier['incorporation_date']).days
        
        limit = CAPACITY_LIMITS.get(supplier['company_size'], 20_000_000)
        budget = tender['tender_budget_kes']
        
        # --- LOGICAL BASELINE ---
        # 1. Base delay scales with budget size (Complexity)
        # 0 to 15 days baseline depending on budget
        complexity_factor = min(15, (budget / 500_000_000) * 15) 
        
        # 2. Base delay scales with Capacity Utilization
        # Higher utilization of firm capacity = higher risk of delay
        utilization = budget / limit
        utilization_delay = utilization * 10 # Adds up to 10 days for 100% capacity
        
        days_delayed = max(0, int(np.random.normal(complexity_factor + utilization_delay, 2)))
        cost_overrun_pct = max(0, np.random.normal(utilization * 5, 2))
        
        delivery_quality = 'Good'
        contract_status = 'Completed'
        
        # --- RISK INJECTION (Extreme Cases) ---
        
        # Mismatch (Exponential Risk)
        if budget > limit:
            excess = budget / limit
            days_delayed += np.random.randint(20, min(500, int(30 * excess)))
            cost_overrun_pct += np.random.uniform(10, min(100, 15 * excess))
            delivery_quality = 'Poor'
            if np.random.random() < (0.1 * excess): contract_status = 'Terminated'

        # Financial Health
        if supplier['credit_score'] < 500:
            days_delayed += np.random.randint(15, 60)
            cost_overrun_pct += 20
        elif supplier['credit_score'] < 650:
            days_delayed += np.random.randint(5, 20)

        # Young Company
        if supplier_age_days < 365:
            days_delayed += np.random.randint(10, 40)
            
        contracts.append({
            'contract_id': f"C-{tender_id}",
            'tender_id': tender_id,
            'supplier_id': supplier['supplier_id'],
            'award_amount_kes': budget * np.random.uniform(0.98, 1.05),
            'contract_start_date': tender['tender_open_date'],
            'days_delayed': days_delayed,
            'cost_overrun_percentage': round(cost_overrun_pct, 2),
            'delivery_quality': delivery_quality,
            'contract_status': contract_status,
            'supplier_age_at_award_days': supplier_age_days
        })
        
    return pd.DataFrame(contracts)

if __name__ == "__main__":
    print("Generating Suppliers...")
    df_suppliers = generate_suppliers(NUM_SUPPLIERS)
    
    print("Generating Tenders...")
    df_tenders = generate_tenders(NUM_TENDERS)
    
    print("Generating Contract Performance (with embedded risk patterns)...")
    df_contracts = generate_performance_data(df_suppliers, df_tenders)
    
    # Merge for a master view
    df_master = df_contracts.merge(df_suppliers, on='supplier_id').merge(df_tenders, on='tender_id')
    
    # Save to CSV
    df_suppliers.to_csv('data/suppliers.csv', index=False)
    df_tenders.to_csv('data/tenders.csv', index=False)
    df_contracts.to_csv('data/contracts.csv', index=False)
    df_master.to_csv('data/procurement_master_dataset.csv', index=False)
    
    print(f"Done! Generated {len(df_master)} contract records.")
    print("Data saved to server/data/")