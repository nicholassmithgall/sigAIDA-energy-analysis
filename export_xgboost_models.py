import json
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor


def save_model(model, path_prefix, features, target, metadata):
    joblib.dump({
        "model": model,
        "features": features,
        "target": target,
        "metadata": metadata,
    }, f"{path_prefix}.joblib")
    model.save_model(f"{path_prefix}.json")
    with open(f"{path_prefix}_metadata.json", "w") as f:
        json.dump({"features": features, "target": target, **metadata}, f, indent=2)
    print(f"  Saved {path_prefix}.joblib, {path_prefix}.json, {path_prefix}_metadata.json")

def export_temperature_model():
    print("=== Temperature Model ===")
    df = pd.read_csv("data/climdiv_county_year.csv")
    df = df[["fips", "year", "temp", "tempc"]]
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]

    features = ["state_fips", "year"]
    target = "temp"

    df["state_fips"] = df["state_fips"].astype("category")

    model = XGBRegressor(
        random_state=42, n_jobs=-1, tree_method="hist", enable_categorical=True,
        n_estimators=1000, max_depth=5, learning_rate=0.1,
        subsample=0.9, colsample_bytree=1.0, min_child_weight=5,
        gamma=0.1, reg_lambda=1,
    )
    model.fit(df[features], df[target])

    save_model(model, "models/xgb_temperature", features, target, {
        "name": "xgb_temperature",
        "description": "Predicts average annual temperature by state and year",
        "best_cv_r2": 0.888,
    })


def export_energy_analysis_model():
    print("=== Energy Analysis Model ===")
    df_raw = pd.read_excel("data/co2_source.xlsx", sheet_name="Total", header=2)
    df_raw = df_raw.iloc[1:].reset_index(drop=True)
    new_df = pd.DataFrame(df_raw.values[0:], columns=df_raw.iloc[0])
    new_df = new_df.dropna()
    new_df = new_df.T
    new_df = pd.DataFrame(new_df.values[1:], columns=new_df.iloc[0])
    new_df = new_df.reset_index()

    features = ["index"]
    target = "US"

    new_df["index"] = pd.to_numeric(new_df["index"])
    new_df["US"] = pd.to_numeric(new_df["US"])

    model = XGBRegressor(
        random_state=42, n_jobs=-1, tree_method="hist", enable_categorical=True,
        n_estimators=300, max_depth=5, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.9, min_child_weight=1,
        gamma=0.1, reg_lambda=1.5,
    )
    model.fit(new_df[features], new_df[target])

    save_model(model, "models/xgb_energy_analysis", features, target, {
        "name": "xgb_energy_analysis",
        "description": "Predicts total US CO2 emissions by year index",
        "best_cv_r2": 0.711,
    })


def load_excel_data(path, sheet_list):
    merged_df = None
    for sheet in sheet_list:
        df = pd.read_excel(path, sheet_name=sheet, skiprows=2)
        df_melt = df.melt(id_vars=["State"], var_name="Year", value_name=sheet)
        df_melt["Year"] = pd.to_numeric(df_melt["Year"], errors="coerce")
        if merged_df is None:
            merged_df = df_melt
        else:
            merged_df = pd.merge(merged_df, df_melt, on=["State", "Year"], how="outer")
    merged_df = merged_df.dropna(subset=["Year"])
    merged_df["Year"] = merged_df["Year"].astype(int)
    merged_df = merged_df[merged_df["State"] != "US"].dropna()
    return merged_df


def export_source_sector_model():
    print("=== Source & Sector Model ===")
    df_sector = load_excel_data("data/co2_sector.xlsx",
        ["Residential", "Commercial", "Industrial", "Transportation", "Electric power", "Total"])
    df_source = load_excel_data("data/co2_source.xlsx",
        ["Coal", "Natural gas", "Petroleum"])
    df_other = load_excel_data("data/indicator_other.xlsx",
        ["Total population", "Real GDP", "HDD", "CDD"])

    df_clean = pd.merge(df_sector, df_source, on=["State", "Year"], how="inner")
    df_clean = pd.merge(df_clean, df_other, on=["State", "Year"], how="inner")
    df_clean["State"] = df_clean["State"].astype("category")

    features = ["State", "Year", "Total population"]
    target = "Total"

    model = XGBRegressor(
        random_state=42, n_jobs=-1, tree_method="hist", enable_categorical=True,
        n_estimators=1000, max_depth=3, learning_rate=0.075,
        subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
        gamma=0, reg_lambda=2,
    )
    model.fit(df_clean[features], df_clean[target])

    save_model(model, "models/xgb_source_sector", features, target, {
        "name": "xgb_source_sector",
        "description": "Predicts total CO2 emissions per state by state, year, and population",
        "best_cv_r2": 0.997,
    })


if __name__ == "__main__":
    export_temperature_model()
    export_energy_analysis_model()
    export_source_sector_model()
    print("\nAll models exported to models/")
