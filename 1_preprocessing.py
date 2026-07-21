"""
Step 1: Data Preprocessing
Real Estate Investment Advisor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Load ───────────────────────────────────────────────────────────────────

def load_data(path='data/india_housing_prices.csv'):
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print(f"Duplicates: {df.duplicated().sum()}")
    return df


# ─── Clean ──────────────────────────────────────────────────────────────────

def clean_data(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicates")

    # Drop rows with zero/negative price or size
    for col in ['Price_in_Lakhs', 'Size_in_SqFt']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df['Price_in_Lakhs'] > 0) & (df['Size_in_SqFt'] > 0)]

    # Auto-impute: numeric → median, text → mode
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ─── Outlier removal ────────────────────────────────────────────────────────

def remove_outliers_iqr(df, cols):
    before = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"Removed {before - len(df)} outlier rows")
    return df.reset_index(drop=True)


# ─── Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df):
    # Age of property
    df['Year_Built'] = pd.to_numeric(df['Year_Built'], errors='coerce').fillna(2000)
    df['Age_of_Property'] = (2025 - df['Year_Built']).clip(lower=0)

    # Price per sqft
    df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
    df['Size_in_SqFt']   = pd.to_numeric(df['Size_in_SqFt'], errors='coerce')
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt']

    # Amenity score
    df['Amenity_Score'] = df['Amenities'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) and str(x).strip() not in ['nan',''] else 0
    ) if 'Amenities' in df.columns else 0

    # Infrastructure score — safely convert each column
    infra_cols = ['Nearby_Schools', 'Nearby_Hospitals', 'Public_Transport_Accessibility']
    for c in infra_cols:
        if c not in df.columns:
            continue
        if df[c].dtype == object:
            mapped  = df[c].map({'Yes': 1, 'No': 0})
            numeric = pd.to_numeric(df[c], errors='coerce')
            df[c]   = mapped.where(mapped.notna(), numeric).fillna(0)
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    present = [c for c in infra_cols if c in df.columns]
    df['Infra_Score'] = df[present].sum(axis=1) if present else 0

    # BHK numeric
    if 'BHK' in df.columns:
        df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce').fillna(2)

    return df


# ─── Target Variables ───────────────────────────────────────────────────────

def create_targets(df):
    growth_rates = {
        'Mumbai': 0.10, 'Bangalore': 0.09, 'Delhi': 0.09,
        'Hyderabad': 0.08, 'Chennai': 0.07, 'Pune': 0.08,
        'Kolkata': 0.07, 'Ahmedabad': 0.07
    }
    df['city_growth'] = df['City'].map(growth_rates).fillna(0.075)
    df['Future_Price_5Y'] = (df['Price_in_Lakhs'] * (1 + df['city_growth']) ** 5).round(2)

    city_med_price = df.groupby('City')['Price_in_Lakhs'].transform('median')
    city_med_ppsf  = df.groupby('City')['Price_per_SqFt'].transform('median')
    df['Good_Investment'] = (
        (df['Price_in_Lakhs'] <= city_med_price) &
        (df['Price_per_SqFt']  <= city_med_ppsf)
    ).astype(int)

    dist = df['Good_Investment'].value_counts(normalize=True).round(3) * 100
    print(f"\nGood_Investment distribution:\n{dist.to_string()}")
    return df


# ─── Encode & Scale ─────────────────────────────────────────────────────────

def encode_and_scale(df):
    # Drop columns not useful for ML
    drop_cols = ['ID', 'Locality', 'Amenities', 'Year_Built', 'city_growth']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Ordinal: Furnished_Status
    furnished_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Fully Furnished': 2}
    if 'Furnished_Status' in df.columns:
        df['Furnished_Status'] = df['Furnished_Status'].map(furnished_map).fillna(0).astype(int)

    # One-hot encode known nominal columns
    ohe_cols = [c for c in ['Property_Type', 'Facing', 'Owner_Type',
                              'Availability_Status', 'Security'] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Label encode high-cardinality text (State, City)
    for col in ['State', 'City']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Handle ALL remaining object columns
    for col in df.select_dtypes(include='object').columns:
        mapped  = df[col].map({'Yes': 1, 'No': 0})
        numeric = pd.to_numeric(df[col], errors='coerce')
        if mapped.notna().mean() > 0.8:
            df[col] = mapped.fillna(0).astype(int)
        elif numeric.notna().mean() > 0.5:
            df[col] = numeric.fillna(0)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Convert booleans to int
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    # Scale all numeric columns except targets
    exclude = {'Good_Investment', 'Future_Price_5Y', 'Price_in_Lakhs'}
    scale_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    print(f"Final dataset shape: {df.shape}")
    print(f"Columns scaled: {len(scale_cols)}")
    return df, scaler


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = load_data()
    df = clean_data(df)
    df = remove_outliers_iqr(df, ['Price_in_Lakhs', 'Size_in_SqFt'])
    df = engineer_features(df)
    df = create_targets(df)
    df, scaler = encode_and_scale(df)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/processed.csv', index=False)
    print(f"\nSaved → data/processed.csv")
    print(f"Target ranges:")
    print(f"  Future_Price_5Y : {df['Future_Price_5Y'].min():.1f} – {df['Future_Price_5Y'].max():.1f} lakhs")
    print(f"  Good_Investment : {df['Good_Investment'].sum()} positive out of {len(df)} total")
