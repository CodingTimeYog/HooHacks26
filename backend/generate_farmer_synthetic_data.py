import pandas as pd
import numpy as np
import os

# Seed for reproducibility
np.random.seed(42)
num_rows = 1000

# 1. Agricultural Data Distributions (Matched exactly to your uploaded dataset)
crops = ['Cotton', 'Barley', 'Tomato', 'Sugarcane', 'Soybean', 'Rice', 'Carrot', 'Wheat', 'Potato', 'Maize']
crop_probs = [0.14, 0.14, 0.12, 0.1, 0.1, 0.1, 0.08, 0.08, 0.08, 0.06]

irrigations = ['Drip', 'Flood', 'Sprinkler', 'Rain-fed', 'Manual']
irrig_probs = [0.30, 0.26, 0.18, 0.14, 0.12]

soils = ['Clay', 'Loamy', 'Sandy', 'Silty', 'Peaty']
soil_probs = [0.28, 0.22, 0.22, 0.20, 0.08]

seasons = ['Zaid', 'Kharif', 'Rabi']
season_probs = [0.46, 0.32, 0.22]

# 2. THE ECONOMIC LOGIC (Miles can adjust these!)
# Multiplier for fertilizer sensitivity (Maize/Corn is highly dependent on nitrogen, Soy is low)
fert_multiplier = {
    'Maize': 1.8, 'Sugarcane': 1.6, 'Cotton': 1.4, 'Tomato': 1.2, 'Potato': 1.2,
    'Wheat': 1.0, 'Barley': 0.9, 'Rice': 0.8, 'Carrot': 0.6, 'Soybean': 0.3
}

# 3. Generate Base Data
data = {
    'Borrower_ID': [f"FARM-{1000 + i}" for i in range(num_rows)],
    'Crop_Type': np.random.choice(crops, num_rows, p=crop_probs),
    'Farm_Area_Acres': np.round(np.random.uniform(50, 1500, num_rows), 2),
    'Irrigation_Type': np.random.choice(irrigations, num_rows, p=irrig_probs),
    'Soil_Type': np.random.choice(soils, num_rows, p=soil_probs),
    'Season': np.random.choice(seasons, num_rows, p=season_probs),
}
df = pd.DataFrame(data)

# 4. Add Calculated Financial/ML Features
# Fertilizer needed (tons) varies by acreage and crop sensitivity
df['Fertilizer_Used_Tons'] = np.round(df['Farm_Area_Acres'] * df['Crop_Type'].map(fert_multiplier) * np.random.uniform(0.05, 0.15, num_rows), 2)

# Financial Health Metrics
df['Current_LTV_Ratio'] = np.round(np.random.beta(5, 2, num_rows) * 0.95, 2) # Skewed towards 0.6 - 0.8
df['Months_Since_Delinquency'] = np.random.choice(
    [-1] + list(range(1, 48)), # -1 means never delinquent
    num_rows, 
    p=[0.7] + [0.3/47]*47
)

# 5. Generate the Target Variable (Stress Flag) for your ML Model
# A farmer is likely to default if LTV is high, Fertilizer needs are high, and they have recent delinquencies.
base_risk = (df['Current_LTV_Ratio'] * 0.4) + \
            ((df['Crop_Type'].map(fert_multiplier) / 2.0) * 0.4) + \
            (np.where(df['Months_Since_Delinquency'] != -1, 12/(df['Months_Since_Delinquency']+1), 0) * 0.2)

# Add some randomness to simulate real-world noise
df['Stress_Probability'] = np.clip(base_risk + np.random.normal(0, 0.1, num_rows), 0, 1)

# Set the binary target for the ML model to predict (1 = Offer Intervention, 0 = Healthy)
df['Requires_Intervention'] = (df['Stress_Probability'] > 0.65).astype(int)

# 6. Save the output
out_path = os.path.join(os.path.dirname(__file__), "../data/agriculture-and-farming-dataset/synthetic_farm_borrowers.csv")
df.to_csv(out_path, index=False)
print("Data generated successfully!")