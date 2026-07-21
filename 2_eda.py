"""
Step 2: Exploratory Data Analysis (EDA) — all 20 questions
Real Estate Investment Advisor
Run BEFORE preprocessing (uses raw data for interpretable plots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.0)
os.makedirs('eda_plots', exist_ok=True)

RAW_PATH = 'data/india_housing_prices.csv'


def load_raw():
    df = pd.read_csv(RAW_PATH)
    # Ensure derived columns exist
    df['Age_of_Property'] = 2025 - df['Year_Built']
    df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt'].replace(0, np.nan)
    df['Amenity_Score'] = df['Amenities'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) and str(x) != 'nan' else 0)
    infra = [c for c in ['Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility'] if c in df.columns]
    for _c in infra:
        if df[_c].dtype == object:
            _mapped  = df[_c].map({'Yes': 1, 'No': 0})
            _numeric = pd.to_numeric(df[_c], errors='coerce')
            df[_c]   = _mapped.where(_mapped.notna(), _numeric).fillna(0)
        else:
            df[_c] = pd.to_numeric(df[_c], errors='coerce').fillna(0)
    df['Infra_Score'] = df[infra].astype(float).sum(axis=1)

    # Good Investment label
    city_med = df.groupby('City')['Price_in_Lakhs'].transform('median')
    city_ppsf = df.groupby('City')['Price_per_SqFt'].transform('median')
    df['Good_Investment'] = ((df['Price_in_Lakhs'] <= city_med) & (df['Price_per_SqFt'] <= city_ppsf)).astype(int)
    return df


def save(name):
    plt.tight_layout()
    plt.savefig(f'eda_plots/{name}.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved eda_plots/{name}.png")


# ════════════════════════════════════════════════════════════════════════════
# Q1–5  Price & Size Analysis
# ════════════════════════════════════════════════════════════════════════════

def q1_price_distribution(df):
    """Q1: Distribution of property prices"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Price_in_Lakhs'].plot(kind='hist', bins=60, ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Distribution of price (₹ lakhs)')
    axes[0].set_xlabel('Price (₹ lakhs)')
    df['Price_in_Lakhs'].plot(kind='box', ax=axes[1], color='steelblue')
    axes[1].set_title('Box plot — price')
    save('q1_price_distribution')


def q2_size_distribution(df):
    """Q2: Distribution of property sizes"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Size_in_SqFt'].plot(kind='hist', bins=60, ax=axes[0], color='teal', edgecolor='white')
    axes[0].set_title('Distribution of size (sq ft)')
    axes[0].set_xlabel('Size (sq ft)')
    df['Size_in_SqFt'].plot(kind='box', ax=axes[1], color='teal')
    axes[1].set_title('Box plot — size')
    save('q2_size_distribution')


def q3_ppsf_by_type(df):
    """Q3: Price per sq ft by property type"""
    order = df.groupby('Property_Type')['Price_per_SqFt'].median().sort_values(ascending=False).index
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt', order=order, palette='Blues_d')
    plt.title('Price per sq ft by property type')
    plt.xlabel('Property type')
    plt.ylabel('Price / sq ft (₹ lakhs)')
    plt.xticks(rotation=30, ha='right')
    save('q3_ppsf_by_type')


def q4_size_vs_price(df):
    """Q4: Relationship between property size and price"""
    sample = df.sample(min(3000, len(df)), random_state=42)
    plt.figure(figsize=(8, 5))
    plt.scatter(sample['Size_in_SqFt'], sample['Price_in_Lakhs'], alpha=0.3, s=12, color='steelblue')
    m, b = np.polyfit(sample['Size_in_SqFt'], sample['Price_in_Lakhs'], 1)
    xs = np.linspace(sample['Size_in_SqFt'].min(), sample['Size_in_SqFt'].max(), 200)
    plt.plot(xs, m * xs + b, color='crimson', lw=1.5, label='Trend')
    corr = sample['Size_in_SqFt'].corr(sample['Price_in_Lakhs'])
    plt.title(f'Size vs price  (Pearson r = {corr:.2f})')
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price (₹ lakhs)')
    plt.legend()
    save('q4_size_vs_price')


def q5_outliers_ppsf(df):
    """Q5: Outliers in price per sq ft"""
    Q1 = df['Price_per_SqFt'].quantile(0.25)
    Q3 = df['Price_per_SqFt'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['Price_per_SqFt'] < Q1 - 1.5*IQR) | (df['Price_per_SqFt'] > Q3 + 1.5*IQR)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['Price_per_SqFt'].plot(kind='hist', bins=60, ax=axes[0], color='darkorange', edgecolor='white')
    axes[0].set_title(f'Price/sqft distribution  ({len(outliers)} outliers)')
    df['Price_per_SqFt'].plot(kind='box', ax=axes[1], color='darkorange')
    axes[1].set_title('Box plot — price/sqft')
    save('q5_outliers_ppsf')
    print(f"  Q5: {len(outliers)} outlier rows in Price_per_SqFt")


# ════════════════════════════════════════════════════════════════════════════
# Q6–10  Location Analysis
# ════════════════════════════════════════════════════════════════════════════

def q6_ppsf_by_state(df):
    """Q6: Average price per sq ft by state"""
    state_avg = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 5))
    state_avg.plot(kind='bar', color='steelblue')
    plt.title('Avg price per sq ft by state')
    plt.ylabel('Avg price/sqft (₹ lakhs)')
    plt.xticks(rotation=45, ha='right')
    save('q6_ppsf_by_state')


def q7_avg_price_by_city(df):
    """Q7: Average property price by city (top 15)"""
    top = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    top.plot(kind='bar', color='teal')
    plt.title('Avg property price by city (top 15)')
    plt.ylabel('Avg price (₹ lakhs)')
    plt.xticks(rotation=45, ha='right')
    save('q7_avg_price_by_city')


def q8_median_age_by_locality(df):
    """Q8: Median age of properties by locality (top 15)"""
    top = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 5))
    top.plot(kind='barh', color='slateblue')
    plt.title('Median property age by locality (top 15 oldest)')
    plt.xlabel('Median age (years)')
    save('q8_median_age_locality')


def q9_bhk_by_city(df):
    """Q9: BHK distribution across top 8 cities"""
    top_cities = df['City'].value_counts().head(8).index
    sub = df[df['City'].isin(top_cities)]
    bhk_city = sub.groupby(['City','BHK']).size().unstack(fill_value=0)
    bhk_city.plot(kind='bar', stacked=True, figsize=(12,5), colormap='tab10')
    plt.title('BHK distribution across top 8 cities')
    plt.ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='BHK', bbox_to_anchor=(1.01,1))
    save('q9_bhk_by_city')


def q10_price_trends_localities(df):
    """Q10: Price trends for top 5 most expensive localities"""
    top5 = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5).index
    sub = df[df['Locality'].isin(top5)]
    plt.figure(figsize=(10, 5))
    for loc in top5:
        d = sub[sub['Locality'] == loc].sort_values('Year_Built')
        if len(d) > 1:
            plt.plot(d['Year_Built'], d['Price_in_Lakhs'], marker='o', markersize=3, label=loc, alpha=0.7)
    plt.title('Price trend by year built — top 5 expensive localities')
    plt.xlabel('Year built')
    plt.ylabel('Price (₹ lakhs)')
    plt.legend(fontsize=8)
    save('q10_price_trends_localities')


# ════════════════════════════════════════════════════════════════════════════
# Q11–15  Feature Relationships
# ════════════════════════════════════════════════════════════════════════════

def q11_correlation_heatmap(df):
    """Q11: Correlation between numeric features"""
    num = df.select_dtypes(include='number').drop(
        columns=['ID','Year_Built','Good_Investment'], errors='ignore')
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature correlation heatmap')
    save('q11_correlation_heatmap')


def q12_schools_vs_ppsf(df):
    """Q12: Nearby schools vs price per sq ft"""
    plt.figure(figsize=(8, 5))
    avg = df.groupby('Nearby_Schools')['Price_per_SqFt'].mean()
    avg.plot(kind='bar', color='mediumseagreen')
    plt.title('Avg price/sqft by number of nearby schools')
    plt.xlabel('Nearby schools')
    plt.ylabel('Avg price/sqft (₹ lakhs)')
    save('q12_schools_vs_ppsf')


def q13_hospitals_vs_ppsf(df):
    """Q13: Nearby hospitals vs price per sq ft"""
    plt.figure(figsize=(8, 5))
    avg = df.groupby('Nearby_Hospitals')['Price_per_SqFt'].mean()
    avg.plot(kind='bar', color='coral')
    plt.title('Avg price/sqft by number of nearby hospitals')
    plt.xlabel('Nearby hospitals')
    plt.ylabel('Avg price/sqft (₹ lakhs)')
    save('q13_hospitals_vs_ppsf')


def q14_price_by_furnished(df):
    """Q14: Price by furnished status"""
    order = ['Unfurnished','Semi-Furnished','Fully Furnished']
    order = [o for o in order if o in df['Furnished_Status'].unique()]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Furnished_Status', y='Price_in_Lakhs', order=order, palette='Set2')
    plt.title('Property price by furnished status')
    plt.xlabel('Furnished status')
    plt.ylabel('Price (₹ lakhs)')
    save('q14_price_by_furnished')


def q15_ppsf_by_facing(df):
    """Q15: Price per sq ft by facing direction"""
    avg = df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    avg.plot(kind='bar', color='orchid')
    plt.title('Avg price/sqft by facing direction')
    plt.ylabel('Avg price/sqft (₹ lakhs)')
    plt.xticks(rotation=30, ha='right')
    save('q15_ppsf_by_facing')


# ════════════════════════════════════════════════════════════════════════════
# Q16–20  Investment / Amenities / Ownership
# ════════════════════════════════════════════════════════════════════════════

def q16_owner_type_count(df):
    """Q16: Properties by owner type"""
    plt.figure(figsize=(7, 5))
    df['Owner_Type'].value_counts().plot(kind='bar', color='darkcyan')
    plt.title('Properties by owner type')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    save('q16_owner_type_count')


def q17_availability_status(df):
    """Q17: Properties by availability status"""
    plt.figure(figsize=(7, 5))
    df['Availability_Status'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                   startangle=140, colormap='Set3')
    plt.title('Availability status distribution')
    plt.ylabel('')
    save('q17_availability_status')


def q18_parking_vs_price(df):
    """Q18: Parking space vs property price"""
    plt.figure(figsize=(8, 5))
    avg = df.groupby('Parking_Space')['Price_in_Lakhs'].mean()
    avg.plot(kind='bar', color='goldenrod')
    plt.title('Avg price by number of parking spaces')
    plt.xlabel('Parking spaces')
    plt.ylabel('Avg price (₹ lakhs)')
    save('q18_parking_vs_price')


def q19_amenities_vs_ppsf(df):
    """Q19: Amenity score vs price per sq ft"""
    plt.figure(figsize=(9, 5))
    avg = df.groupby('Amenity_Score')['Price_per_SqFt'].mean()
    avg.plot(kind='bar', color='slateblue')
    plt.title('Avg price/sqft by amenity count')
    plt.xlabel('Number of amenities')
    plt.ylabel('Avg price/sqft (₹ lakhs)')
    save('q19_amenities_vs_ppsf')


def q20_transport_vs_investment(df):
    """Q20: Public transport accessibility vs investment potential"""
    pivot = df.groupby(['Public_Transport_Accessibility','Good_Investment']).size().unstack(fill_value=0)
    pivot.columns = ['Not good investment','Good investment']
    pivot.plot(kind='bar', stacked=True, figsize=(9, 5), color=['#d73027','#1a9850'])
    plt.title('Public transport accessibility vs investment classification')
    plt.xlabel('Public transport accessibility score')
    plt.ylabel('Number of properties')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    save('q20_transport_vs_investment')


# ─── Run all ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading raw data...")
    df = load_raw()
    fns = [
        q1_price_distribution, q2_size_distribution, q3_ppsf_by_type,
        q4_size_vs_price, q5_outliers_ppsf,
        q6_ppsf_by_state, q7_avg_price_by_city, q8_median_age_by_locality,
        q9_bhk_by_city, q10_price_trends_localities,
        q11_correlation_heatmap, q12_schools_vs_ppsf, q13_hospitals_vs_ppsf,
        q14_price_by_furnished, q15_ppsf_by_facing,
        q16_owner_type_count, q17_availability_status, q18_parking_vs_price,
        q19_amenities_vs_ppsf, q20_transport_vs_investment
    ]
    for i, fn in enumerate(fns, 1):
        print(f"[{i:02d}/20] {fn.__doc__}")
        try:
            fn(df)
        except Exception as e:
            print(f"  WARNING: {e}")

    print("\nAll EDA plots saved to eda_plots/")
