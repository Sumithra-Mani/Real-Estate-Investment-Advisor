"""
Streamlit App: Real Estate Investment Advisor
Predicting Property Profitability & Future Value

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load models ────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    clf = reg = None
    try:
        clf = pickle.load(open('models/best_classifier.pkl','rb'))
    except FileNotFoundError:
        st.warning("Classifier not found. Run 3_train_models.py first.")
    try:
        reg = pickle.load(open('models/best_regressor.pkl','rb'))
    except FileNotFoundError:
        st.warning("Regressor not found. Run 3_train_models.py first.")
    return clf, reg


@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed.csv')
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_raw():
    try:
        df = pd.read_csv('data/india_housing_prices.csv')
        df['Age_of_Property'] = 2025 - df['Year_Built']
        df['Price_per_SqFt'] = df['Price_in_Lakhs'] / df['Size_in_SqFt'].replace(0, np.nan)
        df['Amenity_Score'] = df['Amenities'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) and str(x) != 'nan' else 0)
        city_med  = df.groupby('City')['Price_in_Lakhs'].transform('median')
        city_ppsf = df.groupby('City')['Price_per_SqFt'].transform('median')
        df['Good_Investment'] = ((df['Price_in_Lakhs'] <= city_med) & (df['Price_per_SqFt'] <= city_ppsf)).astype(int)
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_feature_cols():
    try:
        return pd.read_csv('models/feature_columns.csv', header=None)[0].tolist()
    except:
        return None


clf, reg = load_models()
df_proc  = load_data()
df_raw   = load_raw()
feat_cols = load_feature_cols()

# ─── Sidebar navigation ─────────────────────────────────────────────────────

st.sidebar.title("🏠 Real Estate Advisor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🔮 Predict Investment",
    "📊 EDA Insights",
    "🔍 Property Explorer",
    "📈 Model Performance"
])

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT INVESTMENT
# ════════════════════════════════════════════════════════════════════════════

if page == "🔮 Predict Investment":
    st.title("🔮 Investment Prediction")
    st.markdown("Enter property details to get an investment recommendation and 5-year price forecast.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property details")
        city = st.selectbox("City", ['Mumbai','Bangalore','Delhi','Hyderabad','Chennai','Pune','Kolkata','Ahmedabad','Other'])
        property_type = st.selectbox("Property type", ['Apartment','Villa','House','Studio','Penthouse'])
        bhk = st.slider("BHK", 1, 6, 2)
        size_sqft = st.number_input("Size (sq ft)", min_value=200, max_value=10000, value=1000, step=50)
        price_lakhs = st.number_input("Price (₹ lakhs)", min_value=5.0, max_value=50000.0, value=80.0, step=5.0)
        year_built = st.slider("Year built", 1970, 2024, 2010)

    with col2:
        st.subheader("Location & amenities")
        nearby_schools    = st.slider("Nearby schools", 0, 10, 3)
        nearby_hospitals  = st.slider("Nearby hospitals", 0, 10, 2)
        public_transport  = st.slider("Public transport score", 0, 10, 5)
        parking_space     = st.slider("Parking spaces", 0, 5, 1)
        amenity_count     = st.slider("Number of amenities (gym, pool, etc.)", 0, 8, 2)
        furnished_status  = st.selectbox("Furnished status", ['Unfurnished','Semi-Furnished','Fully Furnished'])
        floor_no          = st.number_input("Floor number", 0, 50, 3)
        total_floors      = st.number_input("Total floors in building", 1, 60, 10)

    st.markdown("---")

    if st.button("🔍 Analyse this property", use_container_width=True):

        # ── Derive features ────────────────────────────────────────────────
        age = 2025 - year_built
        ppsf = price_lakhs / size_sqft if size_sqft > 0 else 0
        infra_score = nearby_schools + nearby_hospitals + public_transport
        furnished_map = {'Unfurnished':0,'Semi-Furnished':1,'Fully Furnished':2}
        furn_enc = furnished_map[furnished_status]

        growth_rates = {'Mumbai':0.10,'Bangalore':0.09,'Delhi':0.09,'Hyderabad':0.08,
                        'Chennai':0.07,'Pune':0.08,'Kolkata':0.07,'Ahmedabad':0.07,'Other':0.075}
        g = growth_rates[city]
        future_price = price_lakhs * (1 + g) ** 5

        input_dict = {
            'BHK': bhk, 'Size_in_SqFt': size_sqft, 'Price_in_Lakhs': price_lakhs,
            'Price_per_SqFt': ppsf, 'Floor_No': floor_no, 'Total_Floors': total_floors,
            'Age_of_Property': age, 'Nearby_Schools': nearby_schools,
            'Nearby_Hospitals': nearby_hospitals, 'Public_Transport_Accessibility': public_transport,
            'Parking_Space': parking_space, 'Furnished_Status': furn_enc,
            'Amenity_Score': amenity_count, 'Infra_Score': infra_score,
            'Future_Price_5Y': future_price
        }

        input_df = pd.DataFrame([input_dict])

        # Align to trained feature columns
        if feat_cols:
            for fc in feat_cols:
                if fc not in input_df.columns:
                    input_df[fc] = 0
            input_df = input_df.reindex(columns=feat_cols, fill_value=0)

        # ── Results ────────────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)

        # Classification
        if clf and feat_cols:
            pred_cls  = clf.predict(input_df)[0]
            prob_good = clf.predict_proba(input_df)[0][1] if hasattr(clf,'predict_proba') else pred_cls
            with r1:
                if pred_cls == 1:
                    st.success(f"### ✅ Good Investment\nConfidence: **{prob_good*100:.1f}%**")
                else:
                    st.error(f"### ❌ Not a Good Investment\nConfidence: **{(1-prob_good)*100:.1f}%**")
        else:
            with r1:
                st.info("Train models to see classification result")

        # Regression
        if reg and feat_cols:
            pred_price = reg.predict(input_df)[0]
            growth_pct = ((pred_price - price_lakhs) / price_lakhs) * 100
            with r2:
                st.info(f"### 📈 5-Year Price Forecast\n**₹ {pred_price:,.1f} lakhs**\nGrowth: +{growth_pct:.1f}%")
        else:
            # Rule-based fallback
            growth_pct = ((future_price - price_lakhs) / price_lakhs) * 100
            with r2:
                st.info(f"### 📈 5-Year Price Forecast\n**₹ {future_price:,.1f} lakhs** (formula-based)\nGrowth: +{growth_pct:.1f}%")

        with r3:
            roi = ((future_price - price_lakhs) / price_lakhs) * 100
            st.metric("Estimated ROI (5 yrs)", f"{roi:.1f}%")
            st.metric("Price per sq ft", f"₹ {ppsf:.2f} L/sqft")
            st.metric("Infrastructure score", f"{infra_score}/30")

        # ── Price projection chart ─────────────────────────────────────────
        st.subheader("Price projection over 5 years")
        years = list(range(0, 6))
        prices = [price_lakhs * (1 + g)**y for y in years]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(years, prices, marker='o', color='steelblue', lw=2)
        ax.fill_between(years, prices, alpha=0.15, color='steelblue')
        ax.set_xlabel('Years from now')
        ax.set_ylabel('Price (₹ lakhs)')
        ax.set_title(f'Projected price — {city} ({g*100:.0f}% annual growth)')
        for y, p in zip(years, prices):
            ax.annotate(f'₹{p:.0f}L', (y, p), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        st.pyplot(fig)
        plt.close()

        # ── Feature summary ───────────────────────────────────────────────
        with st.expander("Property summary"):
            summary = pd.DataFrame({
                'Feature': ['City','Property type','BHK','Size','Price','Year built','Age',
                            'Price/sqft','Furnished','Infra score','Amenities'],
                'Value': [city, property_type, bhk, f'{size_sqft} sqft', f'₹{price_lakhs}L',
                          year_built, f'{age} yrs', f'₹{ppsf:.2f}L/sqft', furnished_status,
                          infra_score, amenity_count]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA INSIGHTS
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA Insights":
    st.title("📊 EDA Insights")

    if df_raw is None:
        st.error("Raw data not found. Place india_housing_prices.csv in the data/ folder.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Price & Size", "Location", "Correlations", "Investment"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_raw['Price_in_Lakhs'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
            ax.set_title('Price distribution (₹ lakhs)')
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_raw['Size_in_SqFt'].hist(bins=50, ax=ax, color='teal', edgecolor='white')
            ax.set_title('Size distribution (sq ft)')
            st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        order = df_raw.groupby('Property_Type')['Price_per_SqFt'].median().sort_values(ascending=False).index
        df_raw.boxplot(column='Price_per_SqFt', by='Property_Type', ax=ax, figsize=(10,4))
        ax.set_title('Price per sq ft by property type')
        plt.suptitle('')
        st.pyplot(fig); plt.close()

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 4))
            city_avg = df_raw.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(12)
            city_avg.plot(kind='barh', ax=ax, color='coral')
            ax.set_title('Avg price by city (₹ lakhs)')
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 4))
            state_avg = df_raw.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).head(12)
            state_avg.plot(kind='barh', ax=ax, color='orchid')
            ax.set_title('Avg price/sqft by state')
            st.pyplot(fig); plt.close()

        if 'BHK' in df_raw.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            top_cities = df_raw['City'].value_counts().head(6).index
            sub = df_raw[df_raw['City'].isin(top_cities)]
            bhk_city = sub.groupby(['City','BHK']).size().unstack(fill_value=0)
            bhk_city.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
            ax.set_title('BHK distribution in top 6 cities')
            plt.xticks(rotation=30, ha='right')
            st.pyplot(fig); plt.close()

    with tab3:
        num = df_raw.select_dtypes(include='number').drop(
            columns=['ID','Year_Built'], errors='ignore').dropna()
        if len(num.columns) > 1:
            corr = num.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                        ax=ax, linewidths=0.5, vmin=-1, vmax=1)
            ax.set_title('Feature correlation heatmap')
            st.pyplot(fig); plt.close()

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_raw['Good_Investment'].value_counts().rename({0:'Not good',1:'Good'}).plot(
                kind='bar', ax=ax, color=['#d73027','#1a9850'])
            ax.set_title('Good investment distribution')
            ax.set_xticklabels(['Not good','Good'], rotation=0)
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            df_raw['Furnished_Status'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_title('Furnished status')
            ax.set_ylabel('')
            st.pyplot(fig); plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        pivot = df_raw.groupby(['Public_Transport_Accessibility','Good_Investment']).size().unstack(fill_value=0)
        pivot.columns = ['Not good','Good investment']
        pivot.plot(kind='bar', stacked=True, ax=ax, color=['#d73027','#1a9850'])
        ax.set_title('Public transport vs investment classification')
        plt.xticks(rotation=0)
        st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PROPERTY EXPLORER
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Property Explorer":
    st.title("🔍 Property Explorer")

    if df_raw is None:
        st.error("Raw data not found.")
        st.stop()

    st.markdown("Filter and explore properties from the dataset.")

    col1, col2, col3 = st.columns(3)
    with col1:
        cities = ['All'] + sorted(df_raw['City'].dropna().unique().tolist())
        sel_city = st.selectbox("City", cities)
    with col2:
        price_min, price_max = float(df_raw['Price_in_Lakhs'].min()), float(df_raw['Price_in_Lakhs'].max())
        price_range = st.slider("Price range (₹ lakhs)", price_min, price_max, (price_min, price_max*0.3))
    with col3:
        bhk_options = ['All'] + sorted(df_raw['BHK'].dropna().unique().tolist()) if 'BHK' in df_raw.columns else ['All']
        sel_bhk = st.selectbox("BHK", bhk_options)

    filtered = df_raw.copy()
    if sel_city != 'All':
        filtered = filtered[filtered['City'] == sel_city]
    filtered = filtered[(filtered['Price_in_Lakhs'] >= price_range[0]) &
                        (filtered['Price_in_Lakhs'] <= price_range[1])]
    if sel_bhk != 'All' and 'BHK' in filtered.columns:
        filtered = filtered[filtered['BHK'] == sel_bhk]

    st.markdown(f"**{len(filtered):,} properties match your filters**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg price", f"₹ {filtered['Price_in_Lakhs'].mean():,.1f} L")
    col2.metric("Median price/sqft", f"₹ {filtered['Price_per_SqFt'].median():,.2f}")
    col3.metric("Avg size", f"{filtered['Size_in_SqFt'].mean():,.0f} sqft")
    col4.metric("Good investments", f"{filtered['Good_Investment'].mean()*100:.1f}%")

    show_cols = [c for c in ['City','Locality','Property_Type','BHK','Size_in_SqFt',
                              'Price_in_Lakhs','Price_per_SqFt','Furnished_Status',
                              'Good_Investment'] if c in filtered.columns]
    st.dataframe(filtered[show_cols].head(200), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    # Feature importance
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Classifier — top features")
        fi_path = 'models/clf_feature_importance.csv'
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path, index_col=0, header=None)
            fi.columns = ['Importance']
            fig, ax = plt.subplots(figsize=(6, 6))
            fi.head(15).sort_values('Importance').plot(kind='barh', ax=ax, color='steelblue', legend=False)
            ax.set_title('Top 15 features — classifier')
            st.pyplot(fig); plt.close()
        else:
            st.info("Train models first to see feature importance.")

    with c2:
        st.subheader("Regressor — top features")
        fi_path2 = 'models/reg_feature_importance.csv'
        if os.path.exists(fi_path2):
            fi2 = pd.read_csv(fi_path2, index_col=0, header=None)
            fi2.columns = ['Importance']
            fig, ax = plt.subplots(figsize=(6, 6))
            fi2.head(15).sort_values('Importance').plot(kind='barh', ax=ax, color='coral', legend=False)
            ax.set_title('Top 15 features — regressor')
            st.pyplot(fig); plt.close()
        else:
            st.info("Train models first to see feature importance.")

    st.subheader("MLflow experiments")
    st.markdown("""
    Run `mlflow ui` in your terminal (from the project folder) to open the MLflow dashboard and compare all model runs.

    ```bash
    mlflow ui
    # → open http://localhost:5000
    ```
    """)

    st.subheader("Model files")
    for f in ['models/best_classifier.pkl','models/best_regressor.pkl',
              'models/clf_feature_importance.csv','models/reg_feature_importance.csv']:
        exists = "✅" if os.path.exists(f) else "❌"
        st.markdown(f"{exists}  `{f}`")


# ─── Footer ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.caption("Real Estate Investment Advisor\nML Capstone Project")
