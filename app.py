import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from excel_parser import load_training_data

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

CAR_MODELS = [
    "BMW",
    "Supra",
    "Porsche",
]

PORSCHE_REAR_OFFSET = -0.05

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
    
    match = re.search(r"^\s*(WUS|DM|DH)", str(value).upper())
    if match:
        return match.group(1)
    
    return np.nan
 
   # Liest NS- oder GP-Runden aus Texten wie: '6NS', '1GP 3NS', '7-8 NS', 'Form + 6NS'

def extract_laps_by_type(laps_raw, lap_type: str) -> str:
    if pd.isna(laps_raw):
        return ""
    
    normalized = str(laps_raw).upper()
    normalized = normalized.replace("\n", " ")

    match = re.search(
        rf"(\d+(?:\s*-\s*\d+)?)\s*{lap_type}\b",
        normalized,
    )

    if not match:
        return ""
    
    return re.sub(
        r"\s+",
        "",
        match.group(1),
    )


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(raw_df)

    required_cols = [
        "event",
        "track",
        "session",
        "driver",
        "tire_entry",
        "tire_type",
        "position",
        "ambient_temp",
        "track_temp",
        "cold_temp",
        "cold_pressure",
        "cold_pressure_corr",
        "bleed_boost",
        "hot_pressure",
        "laps_raw",
        "comment",
        "source_sheet",
        "source_sheet_order",
        "source_stint_row",
    ]

    for column in required_cols:
            if column not in df.columns:
                df[column] = np.nan

    numeric_cols = [
            "ambient_temp",
            "track_temp",
            "cold_temp",
            "cold_pressure",
            "cold_pressure_corr",
            "bleed_boost",
            "hot_pressure",
            "source_sheet_order",
            "source_stint_row",
        ]

    for column in numeric_cols:
            df[column] = to_number(df[column])

    df["bleed_boost"] = df["bleed_boost"].fillna(0.0)

    df["tire_type"] = df["tire_type"].replace("", np.nan)
    df["tire_type"] = df["tire_type"].fillna(
        df["tire_entry"].apply(extract_tire_type)
    )

    df["tire_type"] = df["tire_type"].astype(str).str.upper().str.strip()
    df["track"] = df["track"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.strip()
    df["driver"] = df["driver"].fillna("").astype(str).str.strip()
    df["comment"] = df["comment"].fillna("").astype(str).str.strip()
    df["laps_raw"] = df["laps_raw"].fillna("").astype(str).str.strip()
    df["source_sheet"] = df["source_sheet"].fillna("").astype(str).str.strip()

    # Falls Streckentemperatur fehlt, vorübergehend Außentemperatur nutzen.
    df["track_temp_was_missing"] = df["track_temp"].isna()
    df["track_temp"] = df["track_temp"].fillna(df["ambient_temp"])

    #Kaltdruck immer auf 10C normieren
    missing_corr = (
        df["cold_pressure_corr"].isna()
        & df["cold_pressure"].notna()
        & df["cold_temp"].notna()
    )

    df.loc[missing_corr, "cold_pressure_corr"] = (
        (df.loc[missing_corr, "cold_pressure"] + 1) 
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
def train_model(df):
    if len(df) < 4:
        raise ValueError(
            "Es sind weniger als vier gültige Trainingszeilen vorhanden."
        )
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["tire_type", "track", "position"],
            ),
            (
                "numeric",
                "passthrough",
                ["ambient_temp", "track_temp"],
            )
        ]
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=0.1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x = df[FEATURES]
    y = df["pressure_build"]

    if len(df) >= 30:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
        )

        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        mae = mean_absolute_error(y_test, prediction)
    else:
        pipeline.fit(x, y)
        mae = None

    return pipeline, mae

def calculate_driver_offset(
        df: pd.DataFrame,
        model,
        driver: str,
        tire_type: str,
        track: str,
        position: str,
        ambient_temp: float,
        track_temp: float,
        min_count: int = 5,
):
    if driver == "Ohne Fahrerfilter":
        return 0.0, 0, "Kein Fahrerfilter"
    
    data = df.copy()
    data["base_prediction"] = model.predict(data[FEATURES])
    data["residual"] = data["pressure_build"] - data["base_prediction"]

    filters = [
        (
            "Fahrer + Reifen + Strecke + Position + Temperaturbereich",
            (
                (data["driver"] == driver)
                & (data["tire_type"] == tire_type)
                & (data["track"] == track)
                & (data["position"] == position)
                & (data["ambient_temp"].between(ambient_temp - 5, ambient_temp + 5))
                & (data["track_temp"].between(track_temp - 8, track_temp + 8))
            ),
        ),
        (
            "Fahrer + Reifen + Strecke + Position",
            (
                (data["driver"] == driver)
                & (data["tire_type"] == tire_type)
                & (data["track"] == track)
                & (data["position"] == position)
            ),
        ),
        (
            "Fahrer + Reifen + Position",
            (
                (data["driver"] == driver)
                & (data["tire_type"] == tire_type)
                & (data["position"] == position)               
            ),
        ),
    ]

    for label, mask in filters:
        subset = data.loc[mask]

        if len(subset) >= min_count:
            offset = float(subset["residual"].mean())

            # Fahrer nur als kleiner Korrekturfaktor
            offset = float(np.clip(offset, -0.05, 0.05))

            return offset, len(subset), label
        
    return 0.0, 0, "Zu wenig Fahrerr-Daten, Offset ignoriert"

def build_recommendation(
    df: pd.DataFrame,
    model,
    track: str,
    ambient_temp: float,
    track_temp: float,
    tire_type: str,
    driver: str,
    target_pressure: float,
    car_model: str,
) -> pd.DataFrame:
    raw_results = []

    preferred_position_order = [
        "V_L",
        "V_R",
        "H_L",
        "H_R",
    ]

    available_positions = set(
        df["position"].dropna().unique()
    )

    positions = [
        position
        for position in preferred_position_order
        if position in available_positions
    ]      

    for position in positions:
        input_row = pd.DataFrame(
            [
                {
                    "ambient_temp": ambient_temp,
                    "track_temp": track_temp,
                    "tire_type": tire_type,
                    "track": track,
                    "position": position,
                }
            ]
        )

        base_build = float(model.predict(input_row)[0])

        driver_offset, driver_count, offset_source = calculate_driver_offset(
            df=df,
            model=model,
            driver=driver,
            tire_type=tire_type,
            track=track,
            position=position,
            ambient_temp=ambient_temp,
            track_temp=track_temp,
        )

        final_build = base_build + driver_offset

        # Hier wird der Zieldruck eingerechnet
        # Ausgabe immer bei 10C
        cold_pressure_10C_raw = target_pressure - final_build

        similar_data = df[
            (df["track"] == track)
            & (df["tire_type"] == tire_type)
            & (df["position"] == position)
            & (df["ambient_temp"].between(ambient_temp - 3, ambient_temp + 3))
            & (df["track_temp"].between(track_temp - 5, track_temp + 5))
        ]

        raw_results.append(
            {
                "Position": position,
                "Auto": car_model,
                "Reifen": tire_type,
                "Außentemp": round(ambient_temp, 1),
                "Streckentemp": round(track_temp, 1),
                "Basis-Druckaufbau": round(base_build, 3),
                "Fahrer-Offset": round(driver_offset, 3),
                "Finaler Druckaufbau": round(final_build, 3),
                "Zieldruck": round(target_pressure, 3),
                "Modell-Empfehlung @10°C": round(cold_pressure_10C_raw, 3),
                "Einstelldruck @10°C": round(cold_pressure_10C_raw, 3),
                "Ähnliche Daten": len(similar_data),
                "Auto-Korrektur": 0.0,
                "Fahrer-Daten genutzt": driver_count,
                "Offset-Quelle": offset_source,
            }
        )

    result_df = pd.DataFrame(raw_results)

    # Porsche-Regel anwenden
    # H_L = V_L - 0,05
    # T_R = V_R - 0,05

    if car_model == "Porsche":
        pressure_by_position = dict(
            zip(
                result_df["Position"],
                result_df["Einstelldruck @10°C"],
            )
        )

        if "V_L" in pressure_by_position and "H_L" in pressure_by_position:
            new_H_L = pressure_by_position["V_L"] - 0.05
            mask_H_L = result_df["Position"] == "H_L"

            old_H_L = result_df.loc[
                mask_H_L,
                "Einstelldruck @10°C"
            ].iloc[0]

            result_df.loc[
                mask_H_L,
                "Einstelldruck @10°C"
            ] = round(new_H_L, 3)

            result_df.loc[
                mask_H_L,
                "Auto-Korrektur"
            ] = round(new_H_L - old_H_L, 3)

            result_df.loc[
                mask_H_L,
                "Finaler Druckaufbau"
            ] = round(target_pressure - new_H_L, 3)

        if "V_R" in pressure_by_position and "H_R" in pressure_by_position:
            new_H_R = pressure_by_position["V_R"] - 0.05

            mask_H_R = result_df["Position"] == "H_R"

            old_H_R = result_df.loc[
                mask_H_R,
                "Einstelldruck @10°C"
            ].iloc[0]

            result_df.loc[
                mask_H_R,
                "Einstelldruck @10°C"
            ] = round(new_H_R, 3)

            result_df.loc[
                mask_H_R,
                "Auto-Korrektur"
            ] = round(new_H_R - old_H_R, 3)

            result_df.loc[
                mask_H_R,
                "Finaler Druckaufbau"
            ] = round(target_pressure - new_H_R, 3)

    return result_df

# Gibt historische Einträge im Bereich Streckentemperatur ±5 °C zurück.
def build_history_lookup(
    df: pd.DataFrame,
    reference_track_temp: float,
    selected_track: str,
    selected_tire_type: str,
    selected_driver: str,
    filter_same_track: bool,
    filter_same_tire_type: bool,
    filter_same_driver: bool,
) -> pd.DataFrame:
    
    history_df = df[
        df["track_temp"].between(
            reference_track_temp - 5,
            reference_track_temp + 5,
        )
    ].copy()

    if filter_same_track:
        history_df = history_df[
            history_df["track"] == selected_track
        ]

    if filter_same_tire_type:
        history_df = history_df[
            history_df["tire_type"] == selected_tire_type
        ]

    if (
        filter_same_driver
        and selected_driver != "Ohne Fahrerfilter"
    ):
        history_df = history_df[
            history_df["driver"] == selected_driver
        ]

    if history_df.empty:
        return pd.DataFrame()
    
    history_df["NS-Runden"] = history_df["laps_raw"].apply(
        lambda value: extract_laps_by_type(
            value,
            "NS",
        )
    )

    history_df["GP-Runden"] = history_df["laps_raw"].apply(
        lambda value: extract_laps_by_type(
            value,
            "GP",
        )
    )

    history_df["Temperaturabweichung °C"] = (
        history_df["track_temp"]
        - reference_track_temp
    ).round(1)

    history_df["Sortierung Temperatur"] = (
        history_df["Temperaturabweichung °C"].abs()
    )

    history_df["Streckentemp geschätzt"] = history_df[
        "track_temp_was_missing"
    ].map(
        {
            True: "ja",
            False: "nein",
        }
    )

    # Diese Werte sind für alle vier Reifen eines Stints identisch.
    index_columns = [
        "source_sheet",
        "source_sheet_order",
        "source_stint_row",
        "event",
        "session",
        "driver",
        "tire_entry",
        "tire_type",
        "track",
        "track_temp",
        "Temperaturabweichung °C",
        "Streckentemp geschätzt",
        "laps_raw",
        "NS-Runden",
        "GP-Runden",
        "comment",
    ]

    # Die vier Reifenpositionen werden nebeneinander angeordnet.
    wide_df = history_df.pivot_table(
        index=index_columns,
        columns="position",
        values=[
            "cold_pressure_corr",
            "bleed_boost",
            "hot_pressure",
        ],
        aggfunc="first",
    ).reset_index()

    # Mehrstufige Pandas-Spaltennamen vereinfachen.
    metric_names = {
        "cold_pressure_corr": "Einstelldruck @10°C",
        "bleed_boost": "Bleed/Boost",
        "hot_pressure": "Gemessener Heißdruck",
    }

    flattened_columns = []

    for column in wide_df.columns:
        if not isinstance(column, tuple):
            flattened_columns.append(column)
            continue

        metric, position = column

        if position == "":
            flattened_columns.append(metric)
        else:
            flattened_columns.append(
                f"{position} {metric_names.get(metric, metric)}"
            )

    wide_df.columns = flattened_columns

    wide_df = wide_df.rename(
        columns={
            "event": "Event",
            "session": "Session",
            "driver": "Fahrer",
            "tire_entry": "Reifensatz",
            "tire_type": "Reifenart",
            "track": "Strecke",
            "track_temp": "Streckentemp °C",
            "laps_raw": "Runden Original",
            "comment": "Kommentar",            
        }
    )
# Gewünschte Reihenfolge der Reifenpositionen
    position_order = [
        "V_L",
        "V_R",
        "H_L",
        "H_R",
    ]

    position_columns = []

    for position in position_order:
        position_columns.append(
            f"{position} Einstelldruck @10°C"
        )

    for position in position_order:
        position_columns.append(
            f"{position} Gemessener Heißdruck"
        )

    for position in position_order:
        position_columns.append(
            f"{position} Bleed/Boost"
        )


# Falls bei einem Stint ein Messwert fehlt, bleibt die Spalte leer
    for column in position_columns:
        if column not in wide_df.columns:
            wide_df[column] = np.nan

# Zahlen runden
    for column in position_columns:
        wide_df[column] = pd.to_numeric(
            wide_df[column],
            errors="coerce",
        ).round(3)

    wide_df["Sortierung Temperatur"] = (
        wide_df["Temperaturabweichung °C"].abs()
    )

    wide_df = wide_df.sort_values(
        by=[
            "source_sheet_order",
            "Sortierung Temperatur",
            "source_stint_row",
            "Reifensatz",            
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ]
    )

    return wide_df[
        [
            "Event",
            "Session",
            "Fahrer",
            "Reifensatz",
            "Reifenart",
            "Streckentemp °C",
            "Temperaturabweichung °C",
            "Runden Original",
            "NS-Runden",
            "GP-Runden",
            *position_columns,
            "Kommentar",
        ]
    ]

## Uplaod Excel

uploaded_file = st.file_uploader(
    "Excel-Datei hochladen",
    type=["xlsx"],
)

if uploaded_file is None:
    st.info("Bitte Excel-Datei hochloaden")
    st.stop()

try:
    parsed_df = load_training_data(uploaded_file.getvalue())
except Exception as error:
    st.error("Die Excel- Datei konnte nicht eingelesem werden")
    st.exception(error)
    st.stop()

if parsed_df.empty:
    st.error(
        "Es konnten keine Trainingsdaten aus der Excel extrahiert werden"
    )
    st.stop()

with st.expander("Automatisch extrahierte Rohdaten anzeigen"):
    st.dataframe(parsed_df, use_container_width=True)

df = prepare_data(parsed_df)

# Datencheck

st.subheader("Datencheck")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Trainingszeilen", len(df))
col2.metric("Reifenarten", df["tire_type"].nunique() if len(df) else 0)
col3.metric("Strecken", df["track"].nunique() if len(df) else 0)
col4.metric("Positionen", df["position"].nunique() if len(df) else 0)

if df.empty:
    st.error(
        "Keine gültigen Trainingsdaten gefunden"
    )
    st.stop()

missing_track_temp_count = int(df["track_temp_was_missing"].sum())

if missing_track_temp_count:
    st.warning(
        f"Bei {missing_track_temp_count} Trainingszeilen fehlte die Streckentemperatur. Dort wurde ersatzweise die Außentemperatur genutzt"
    )

if len(df) < 20:
    st.warning(
        "Es sind wenig Trainingsdaten vorhanden"
        "Die Empfehlung ist nur eingeschränkt belastbar"
    )

with st.expander("Bereinigte Trainingsdatem anzeigen"):
    st.dataframe(df, use_container_width=True)

## Modell trainieren
try:
    model, mae = train_model(df)
except ValueError as error:
    st.error(str(error))
    st.stop()

if mae is not None:
    st.success(f"Modell trainiert. Testfehler: {mae:.3f} bar")
else:
    st.success("Modell trainiert. Für einen Testfehler sind zu wenig Daten vorhanden")

## Interface

st.divider()
st.subheader("Recommendation Interface")

left, right = st.columns([1, 2])

with left:
    st.markdown("### Aktuelle Bedingungen")

    track_options = sorted(df["track"].dropna().unique())

    track = st.selectbox(
        "Strecke / Event",
        track_options,
    )

    car_model = st.selectbox(
        "Auto",
        CAR_MODELS,
    )

    ambient_temp = st.number_input(
        "Aktuelle Außentemperatur °C",
        value=18.0,
        step=1.0,
    )

    track_temp = st.number_input(
        "Aktuelle Streckentemperatur °C",
        value=26.0,
        step=1.0,
    )

    available_tire_types = [
        tire_type
        for tire_type in VALID_TIRE_TYPES
        if tire_type in set(df["tire_type"].unique())
    ]

    tire_type = st.selectbox(
        "Reifenart",
        available_tire_types,
        format_func=lambda value: TIRE_LABELS.get(value, value),
    )

    drivers = ["Ohne Fahrerfilter"] + sorted(
        [
            driver_name
            for driver_name in df["driver"].dropna().unique()
            if driver_name and str(driver_name).lower() != "nan"
        ]
    )

    driver = st.selectbox(
        "Fahrer optional",
        drivers,
    )

    target_pressure = st.number_input(
        "Aktueller Zieldruck",
        value=1.77,
        step=0.01,
        format="%.2f",
    )

    calculate = st.button(
        "Empfehlung berechnen",
        type="primary",
        use_container_width=True,
    )

# Empfehlung anzeigen

if calculate:
    st.session_state["recommendation"] = build_recommendation(
        df=df,
        model=model,
        track=track,
        ambient_temp=ambient_temp,
        track_temp=track_temp,
        tire_type=tire_type,
        driver=driver,
        target_pressure=target_pressure,
        car_model=car_model,       
    )

with right:
    st.markdown("### Empfohlener Einstelldruck @10°C")

    if "recommendation" not in st.session_state:
        st.info("Bitte links die aktuellen Bedingungen eingeben und auf „Empfehlung berechnen“ klicken")
    else:
        result_df = st.session_state["recommendation"]

        if result_df.empty:
            st.warning("Für diese Auswahl konnten keine Empfehlungen berechnet werden")

        else:
            metric_columns = st.columns(len(result_df))

            for index, (_, row) in enumerate(result_df.iterrows()):
                with metric_columns[index]:
                    st.metric(
                        label=str(row["Position"]),
                        value=f'{row["Einstelldruck @10°C"]:.2f} bar',
                        delta=f'Druckaufbau {row["Finaler Druckaufbau"]:.2f} bar',
                    )
            st.markdown("### Details")

            display_columns = [
                "Position",
                "Auto",
                "Reifen",
                "Außentemp",
                "Streckentemp",
                "Zieldruck",
                "Basis-Druckaufbau",
                "Fahrer-Offset",
                "Finaler Druckaufbau",
                "Einstelldruck @10°C",
                "Ähnliche Daten",
            ]  

            st.dataframe(
                result_df[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Empfehlng als CSV herunterladen",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="tire_pressure_recommendation.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
# Historische Vergleichswerte

st.divider()
st.subheader("Historische Einträge im Temperaturbereich")

show_history = st.checkbox(
    "Historische Einträge mit Streckentemperatur ±5 °C anzeigen",    
)

if show_history:
    st.caption(
        f"Angezeigter Temperaturbereich: "
        f"{track_temp - 5:.1f} °C bis {track_temp + 5:.1f} °C"
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        filter_same_track = st.checkbox(
            "Nur gewählte Strecke",
            value=True,
        )

    with filter_col2:
        filter_same_tire_type = st.checkbox(
            "Nur gewählte Reifenart",
            value=True,
        )

    with filter_col3:
        filter_same_driver = st.checkbox(
            "Nur gewählte Fahrer",
            value=False,
            disabled=(
                driver == "Ohne Fahrerfilter"
            ),
        )

    history_df = build_history_lookup(
        df=df,
        reference_track_temp=track_temp,
        selected_track=track,
        selected_tire_type=tire_type,
        selected_driver=driver,
        filter_same_track=filter_same_track,
        filter_same_tire_type=filter_same_tire_type,
        filter_same_driver=filter_same_driver,
    )

    if history_df.empty:
        st.warning(
            "Für diesen Temperaturbereich wurden keine historischen Einträge gefunden."        
        )
    
    else:
        st.success(
            f"{len(history_df)} historische Einträge gefunden"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Historische Einträge als CSV herunterladen",
            data=history_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="historische_reifendruckwerte.csv",
            mime="text/csv",
            use_container_width=True,
        )

## Datenqualität

st.divider()
st.subheader("Datenqualität")

quality = (
    df.groupby(["tire_type", "track", "position"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

st.dataframe(
    quality,
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
    **Finale Formel**
    `Einstelldruck @10°C = eingegebener Zieldruck - vorhergesagter Druckaufbau`
    """
)