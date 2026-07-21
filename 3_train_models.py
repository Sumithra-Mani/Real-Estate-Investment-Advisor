"""
Step 3 & 4: Model Training + MLflow Tracking
Real Estate Investment Advisor
"""
import os, warnings, pickle
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, mean_squared_error, mean_absolute_error, r2_score)

os.makedirs('models', exist_ok=True)
mlflow.set_tracking_uri('sqlite:///mlflow.db')


# ─── Prepare data (mirrors debug3.py exactly which gives R²=0.998) ──────────

def prepare_data():
    df = pd.read_csv('data/india_housing_prices.csv')

    df['Price_in_Lakhs'] = pd.to_numeric(df['Price_in_Lakhs'], errors='coerce')
    df['Size_in_SqFt']   = pd.to_numeric(df['Size_in_SqFt'], errors='coerce')
    df = df[(df['Price_in_Lakhs'] > 0) & (df['Size_in_SqFt'] > 0)].copy()

    df['Age_of_Property'] = (2025 - pd.to_numeric(df['Year_Built'], errors='coerce').fillna(2000)).clip(lower=0)
    df['Price_per_SqFt']  = df['Price_in_Lakhs'] / df['Size_in_SqFt']
    df['BHK']             = pd.to_numeric(df['BHK'], errors='coerce').fillna(2)

    for col in ['Floor_No','Total_Floors','Parking_Space']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    for c in ['Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility']:
        if c not in df.columns: continue
        if df[c].dtype == object:
            mapped  = df[c].map({'Yes':1,'No':0})
            numeric = pd.to_numeric(df[c], errors='coerce')
            df[c]   = mapped.where(mapped.notna(), numeric).fillna(0)
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Infra_Score']   = df[['Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility']].sum(axis=1)
    df['Amenity_Score'] = df['Amenities'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) and str(x).strip() not in ['nan',''] else 0
    ) if 'Amenities' in df.columns else 0

    # ── Targets ──────────────────────────────────────────────────────────────
    growth_rates = {'Mumbai':0.10,'Bangalore':0.09,'Delhi':0.09,
                    'Hyderabad':0.08,'Chennai':0.07,'Pune':0.08,
                    'Kolkata':0.07,'Ahmedabad':0.07}
    df['city_growth']    = df['City'].map(growth_rates).fillna(0.075)
    df['Future_Price_5Y'] = (df['Price_in_Lakhs'] * (1 + df['city_growth']) ** 5).round(2)

    city_med  = df.groupby('City')['Price_in_Lakhs'].transform('median')
    city_ppsf = df.groupby('City')['Price_per_SqFt'].transform('median')
    df['Good_Investment'] = ((df['Price_in_Lakhs'] <= city_med) &
                             (df['Price_per_SqFt']  <= city_ppsf)).astype(int)

    # ── Encode ───────────────────────────────────────────────────────────────
    df.drop(columns=[c for c in ['ID','Locality','Amenities','Year_Built','city_growth']
                     if c in df.columns], inplace=True)

    furnished_map = {'Unfurnished':0,'Semi-Furnished':1,'Fully Furnished':2}
    if 'Furnished_Status' in df.columns:
        df['Furnished_Status'] = df['Furnished_Status'].map(furnished_map).fillna(0).astype(int)

    ohe_cols = [c for c in ['Property_Type','Facing','Owner_Type',
                             'Availability_Status','Security'] if c in df.columns]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    for col in ['State','City']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    for col in df.select_dtypes(include='object').columns:
        mapped  = df[col].map({'Yes':1,'No':0})
        numeric = pd.to_numeric(df[col], errors='coerce')
        if mapped.notna().mean() > 0.8:
            df[col] = mapped.fillna(0).astype(int)
        elif numeric.notna().mean() > 0.5:
            df[col] = numeric.fillna(0)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    df = df.fillna(0)

    print(f"Data ready: {df.shape}")
    print(f"All numeric: {df.select_dtypes(exclude='number').shape[1] == 0}")
    print(f"Good_Investment in df: {'Good_Investment' in df.columns}")
    print(f"Class balance: {df['Good_Investment'].mean()*100:.1f}%")
    print(f"Future_Price_5Y range: {df['Future_Price_5Y'].min():.1f} – {df['Future_Price_5Y'].max():.1f}")

    return df


# ─── Classification ──────────────────────────────────────────────────────────

CLASSIFIERS = {
    'Logistic Regression':  LogisticRegression(max_iter=500, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    'K-Nearest Neighbours': KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

def train_classifiers(X_train, X_test, y_train, y_test, feat_names):
    print("\n" + "="*60)
    print("CLASSIFICATION MODELS — Target: Good_Investment")
    print("="*60)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    results = {}
    mlflow.set_experiment('Classification_GoodInvestment')
    for name, model in CLASSIFIERS.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            y_prob = model.predict_proba(Xte)[:,1] if hasattr(model,'predict_proba') else y_pred
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            auc  = roc_auc_score(y_test, y_prob)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'roc_auc':auc})
            mlflow.sklearn.log_model(model, name=name.replace(' ','_'))
            results[name] = {'model':model,'acc':acc,'f1':f1,'auc':auc}
            print(f"  {name:<28} Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
    best_name  = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']
    print(f"\n  Best classifier: {best_name}  (AUC={results[best_name]['auc']:.3f})")
    pickle.dump({'model':best_model,'scaler':scaler}, open('models/best_classifier.pkl','wb'))
    print("  Saved → models/best_classifier.pkl")
    if hasattr(best_model,'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=feat_names).sort_values(ascending=False)
        fi.head(20).to_csv('models/clf_feature_importance.csv')
        print("  Feature importance → models/clf_feature_importance.csv")
    return results, best_name


# ─── Regression ──────────────────────────────────────────────────────────────

REGRESSORS = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression':  Ridge(alpha=1.0),
    'Lasso Regression':  Lasso(alpha=0.1, max_iter=2000),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
}

def train_regressors(X_train, X_test, y_train, y_test, feat_names):
    print("\n" + "="*60)
    print("REGRESSION MODELS — Target: Future_Price_5Y")
    print("="*60)
    # Raw values — no scaling — Price_in_Lakhs dominates and must not be normalised
    Xtr = np.array(X_train)
    Xte = np.array(X_test)
    ytr = np.array(y_train)
    yte = np.array(y_test)
    print(f"  Sanity check R² (LinearReg): ", end='', flush=True)
    _q = LinearRegression().fit(Xtr, ytr)
    print(f"{r2_score(yte, _q.predict(Xte)):.4f}")
    results = {}
    mlflow.set_experiment('Regression_FuturePrice5Y')
    for name, model in REGRESSORS.items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, ytr)
            y_pred = model.predict(Xte)
            rmse = np.sqrt(mean_squared_error(yte, y_pred))
            mae  = mean_absolute_error(yte, y_pred)
            r2   = r2_score(yte, y_pred)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({'rmse':rmse,'mae':mae,'r2':r2})
            mlflow.sklearn.log_model(model, name=name.replace(' ','_'))
            results[name] = {'model':model,'rmse':rmse,'mae':mae,'r2':r2}
            print(f"  {name:<28} RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")
    best_name  = min(results, key=lambda k: results[k]['rmse'])
    best_model = results[best_name]['model']
    print(f"\n  Best regressor: {best_name}  (RMSE={results[best_name]['rmse']:.2f})")
    pickle.dump({'model':best_model}, open('models/best_regressor.pkl','wb'))
    print("  Saved → models/best_regressor.pkl")
    if hasattr(best_model,'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=feat_names).sort_values(ascending=False)
        fi.head(20).to_csv('models/reg_feature_importance.csv')
        print("  Feature importance → models/reg_feature_importance.csv")
    elif hasattr(best_model,'coef_'):
        fi = pd.Series(np.abs(best_model.coef_), index=feat_names).sort_values(ascending=False)
        fi.head(20).to_csv('models/reg_feature_importance.csv')
        print("  Feature coefficients → models/reg_feature_importance.csv")
    return results, best_name


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    df = prepare_data()

    TARGET_CLS = 'Good_Investment'
    TARGET_REG = 'Future_Price_5Y'
    feat_cols  = [c for c in df.columns if c not in [TARGET_CLS, TARGET_REG]]

    print(f"\nFeature columns ({len(feat_cols)}): {feat_cols}")
    print(f"Price_in_Lakhs in features: {'Price_in_Lakhs' in feat_cols}")

    X     = df[feat_cols]   # keep as DataFrame to preserve column names
    y_cls = df[TARGET_CLS]
    y_reg = df[TARGET_REG]

    pd.Series(feat_cols).to_csv('models/feature_columns.csv', index=False, header=False)

    X_tr, X_te, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    # Regression split must use SAME X split — reuse X_tr/X_te indices
    yr_tr = y_reg.iloc[X_tr.index]
    yr_te = y_reg.iloc[X_te.index]

    train_classifiers(X_tr, X_te, yc_tr, yc_te, feat_cols)
    train_regressors(X_tr, X_te, yr_tr, yr_te, feat_cols)

    print("\nAll done!")
    print("View MLflow:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
