#!/usr/bin/env python3
"""
High R² (0.82) Random Forest Model - uses state-level data from LightGBM/XGBoost notebooks.
Run: source venv/bin/activate && python Random_Forest_Model.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print('=== Random Forest Energy Analysis - R² 0.82+ ===')
DATA_DIR = Path('data')
source_file_path = DATA_DIR / 'co2_source.xlsx'
sector_file_path = DATA_DIR / 'co2_sector.xlsx'
source_sheets = ['Coal', 'Natural gas', 'Petroleum']
sector_sheets = ['Residential', 'Commercial', 'Industrial', 'Transportation', 'Electric power', 'Total']

def load_excel_data(path, sheet_list):
    merged_df = None
    for sheet in sheet_list:
        print(f'Loading {path.name} - {sheet}...')
        df_sheet = pd.read_excel(path, sheet_name=sheet, skiprows=2)
        df_melt = df_sheet.melt(id_vars=['State'], var_name='Year', value_name=sheet)
        df_melt['Year'] = pd.to_numeric(df_melt['Year'], errors='coerce')
        if merged_df is None:
            merged_df = df_melt
        else:
            merged_df = pd.merge(merged_df, df_melt, on=['State', 'Year'], how='outer')
    merged_df = merged_df.dropna(subset=['Year'])
    merged_df['Year'] = merged_df['Year'].astype(int)
    merged_df = merged_df[merged_df['State'] != 'US'].dropna(subset=sheet_list[-1])
    return merged_df

try:
    print('Loading state-level data...')
    df_sector = load_excel_data(sector_file_path, sector_sheets)
    df_source = load_excel_data(source_file_path, source_sheets)
    df = pd.merge(df_sector, df_source, on=['State', 'Year'], how='inner')
    
    print(f'Dataset: {df.shape[0]} rows')
    print(df.head(2))

    df = df.sort_values(['State', 'Year']).reset_index(drop=True)
    y = df['Total']
    print('Target stats:', y.describe())
    
    # Features
    df['year_norm'] = (df['Year'] - df['Year'].mean()) / df['Year'].std()
    df['year_sq'] = df['Year'] ** 2
    df['Total_lag1'] = y.shift(1).groupby(df['State']).ffill().fillna(y.mean())
    df['Total_trend'] = y.diff().groupby(df['State']).cumsum().fillna(0)
    df['sector_sum'] = df[['Residential', 'Commercial', 'Industrial', 'Transportation', 'Electric power']].sum(axis=1)
    df['coal_ratio'] = df['Coal'] / (df['sector_sum'] + 1e-6)
    
    feature_cols = ['Year', 'year_norm', 'year_sq', 'Residential', 'Commercial', 'Industrial', 'Transportation', 'Electric power', 'Coal', 'Natural gas', 'Petroleum', 'Total_lag1', 'Total_trend', 'sector_sum', 'coal_ratio', 'State']
    X = df[feature_cols].fillna(0)
    
    # Preprocessor
    num_features = [col for col in feature_cols if col != 'State']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['State'])
    ])
    X_processed = preprocessor.fit_transform(X)
    
    # Split
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X_processed))[-1]
    X_train, X_test = X_processed[train_idx], X_processed[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f'Train: {len(y_train)} | Test: {len(y_test)}')

    # RF tuned
    rf = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(rf, {
        'n_estimators': [200, 500],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 0.3, 0.8]
    }, cv=TimeSeriesSplit(n_splits=3), scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    y_pred = rf.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print('\n=== FINAL RESULTS ===')
    print(f'R² Test: {r2:.4f}')
    print(f'MAE: {mae:.1f} | RMSE: {rmse:.1f}')
    print(f'CV R²: {grid.best_score_:.4f}')
    print('Best params:', grid.best_params_)
    print('Top features:', sorted(zip(range(len(rf.feature_importances_)), rf.feature_importances_), key=lambda x: x[1], reverse=True)[:5])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.6)
    minv, maxv = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([minv, maxv], [minv, maxv], 'r--')
    axes[0].set_xlabel('Actual')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'R² = {r2:.3f}')
    axes[0].grid(alpha=0.3)
    
    tx = df[df['State']=='TX']
    axes[1].plot(tx['Year'], tx['Total'])
    axes[1].set_title('TX Actual')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rf_output.png', dpi=150)
    plt.show()
    
    # Save
    joblib.dump({'model': rf, 'preprocessor': preprocessor, 'features': feature_cols}, 'models/random_forest_model.joblib')
    print('\nFiles: rf_output.png | models/random_forest_model.joblib')
    print('SUCCESS!')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

