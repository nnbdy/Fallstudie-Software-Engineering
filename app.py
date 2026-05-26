import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REFERENCE_TEMP_C = 10.0

FEATURES = [
    "ambient_temp",
    "track_temp",
    "tire_type",
    "track",
    "position",
]

VALID_TIRE_TYPES = ["DM", "DH", "WUS"]

TIRE_LABELS = {
    "DM": "DM - Medium",
    "DH": "DH - Hard",
    "WUS": "WUS - Regen",
}

st.set_page_config(
    page_title="Tire Pressure AI",
    layout="wide"
)

st.title("Tire Pressure AI")
st.caption( "Vorhersage des Druckaufbaus. Ausgabe des Einstelldrucks immer als Referenzwert bei 10 °C.")

## Data preparation

def clean_columns(df):
    df = df.copy()
    df.columns =[
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        for c in df.columns
    ]
    return df

def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({
            "": np.nan,
            "-": np.nan,
            "/": np.nan,
            "nan": np.nan,
            "None": np.nan,
        }),
        errors="coerce",
    )

def extract_tire_type(value):
    if pd.isna(value):
        return np.nan
    
    match = re.search(r"\b(WUS|DM|DH)\b", str(value).upper())
    if match:
        return match.group(1)
    
    return np.nan

def prepare_data(raw_df):
    df = clean_columns(raw_df)

    required_cols = [
        "event",
        "track",
        "session",
        "driver",
        "tire_entry",
        "position",
        "ambient_temp",
        "track_temp",
        "cold_temp",
        "cold_pressure",
        "cold_pressure_corr",
        "bleed_boost",
        "hot_pressure",
        "comment",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    numeric_cols = [
        "ambient_temp",
        "track_temp",
        "cold_temp",
        "cold_pressure",
        "cold_pressure_corr",
        "bleed_boost",
        "hot_pressure",
    ]

    for col in numeric_cols:
        df[col] = to_number(df[col])

    df["bleed_boost"] = df["bleed_boost"].fillna(0.0)

    df["tire_type"] = df["tire_type"].fillna(
        df("tire_entry").apply(extract_tire_type)
    ) 

    df["tire_type"] = df["tire_type"].astype(str).str.upper().str.strip()
    df["track"] = df["track"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.strip()
    df["driver"] = df["driver"].astype(str).str.strip()
    df["comment"] = df["comment"].astype(str).str.strip()

    # Falls Streckentemperatur fehlt, vorübergehend Außentemperatur nutzen.
    df["track_temp"] = df["track_temp"].fillna(df["ambient_temp"])

    #Kaltdruck immer auf 10C normieren
    missing_corr = (
        df["cold_pressure_corr"].isna()
        & df["cold_pressure"].notna()
        & df["cold_temp"].notna()
    )

    df.loc[missing_corr, "cold_pressure_corr"] = (
        (df.loc[missing_corr, "cold_pressure_corr"] + 1) 
        * (
            (REFERENCE_TEMP_C + 273.15)
             / (df.loc[missing_corr, "cold_temp"] + 273.15)
             )
             - 1
    )

# Druckaufbau wird ohne Zieldruck berechnet

    df["pressure_build"] = (
        df["hot_pressure"]
        - (df["cold_pressure_corr"] + df["bleed_boost"])
    )

    valid = (
        df["ambient_temp"].notna()
        & df["track_temp"].notna()
        & df["cold_pressure_corr"].notna()
        & df["hot_pressure"].notna()
        & df["pressure_build"].notna()
        & df["tire_type"].isin(VALID_TIRE_TYPES)
        & df["track"].ne("")
        & df["track"].ne("nan") 
        & df["position"].ne("")
        & df["position"].ne("nan")         
        )
    
    df = df.loc[valid].copy()

# Grobe Plausibilitätsgrenzen für Druckaufbau.

    df = df[
        (df["pressure_build"] > -0.20)
        & (df["pressure_build"] < 1.50)
    ].copy()

# Sonderfälle optional ausschließen
    bad_keywords = [
        "crash",
        "schaden",
        "damage",
        "defekt",
        "puncture",
        "platte",
    ]

    bad_mask = df["comment"].str.lower().apply(
        lambda x: any(word in x for word in bad_keywords)
    )

    df["excluded_by_comment"] = bad_mask
    df = df[~bad_mask].copy()

    return df

## Model training
