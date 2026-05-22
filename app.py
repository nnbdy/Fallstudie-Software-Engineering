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

st.set_page_config(
    page_title="Tire Pressure AI",
    layout="wide"
)

st.title("Tire Pressure AI")
st.caption( "Vorhersage des Druckaufbaus. Ausgabe des Einstelldrucks immer als Referenzwert bei 10 °C.")


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
        "blled_boost",
        "hot_pressure",
        "comment",
    ]

