import re
from io import BytesIO

import numpy as np
import pandas as pd

# TRACK_BY_SHEET = {
#     "24h Race": "NBR",
#     "24H Q": "NBR",
#     "Qualifiers 2": "NBR",
#     "Qualifiers 1": "NBR",
#     "NLS3 25": "NBR",
#     "NLS2 25": "NBR",
#     "NLS1 25": "NBR",
#     "NLS6 24": "NBR",
#     "NLS5 24": "NBR",
# }

VALID_TIRE_TYPES = {"DM", "DH", "WUS"}

POSITION_SPECS = [
    ("V_L", 0, 4, 9, 14, 18),
    ("V_R", 0, 5, 10, 15, 19),
    ("H_L", 1, 4, 9, 14, 18),
    ("H_R", 1, 5, 10, 15, 19),
]

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def normalize_marker(value) -> str:
    return clean_text(value).upper().replace(" ", "")

def cell(frame: pd.DataFrame, row: int, col: int):
    if row < 0 or row >= len(frame.index):
        return np.nan
    if col < 0 or col >= len(frame.columns):
        return np.nan
    return frame.iat[row, col]

def first_number(*values):
    for value in values:
        if pd.isna(value):
            continue

        parsed = pd.to_numeric(
            str(value).replace(",", ".").replace(" ", ""),
            errors="coerce",
        )

        if not pd.isna(parsed):
            return float(parsed)

    return np.nan
    
def extract_tire_type(value):
    if pd.isna(value):
        return np.nan
    
    # Erkennt sowohl "DM 120" als auch "DM112"
    match = re.search(r"^\s*(WUS|DM|DH)", str(value).upper())

    if match:
        return match.group(1)
    
    return np.nan

def extract_car_model(raw: pd.DataFrame, sheet_name: str) -> str:
    car_text = ""

    # Sucht oben im Sheet nach "Auto:" und nimmt den ersten nicht-leeren Wert rechts davon.
    for row in range(min(5, len(raw.index))):
        for col in range(len(raw.columns)):
            if normalize_marker(cell(raw, row, col)).rstrip(":") == "AUTO":
                for next_col in range(
                    col + 1,
                    min(col + 6, len(raw.columns)),
                ):
                    car_text = clean_text(
                        cell(raw, row, next_col)
                    )

                    if car_text:
                        break

                break

        if car_text:
            break

    # Fallback: Falls kein Auto-Feld gefunden wurde, Sheetnamen verwenden.
    if not car_text:
        car_text = sheet_name

    normalized = (
        car_text
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )

    if "SUPRA" in normalized:
        return "Supra"

    if (
        "PORSCHE" in normalized
        or "922" in normalized
        or "992" in normalized
        or "911" in normalized
    ):
        return "Porsche"

    if "BMW" in normalized or "M4" in normalized:
        return "BMW M4"

    return "BMW M4"

def parse_block_sheet(raw: pd.DataFrame, sheet_name: str, source_sheet_order: int,) -> pd.DataFrame:
    rows = []

    event = clean_text(cell(raw, 0, 1)) or sheet_name
    track = "NBR"
    car_model = extract_car_model(raw, sheet_name)

    current_driver = ""
    current_comment = ""

    for row in range(max(0, len(raw.index) - 1)):
        first_col = clean_text(cell(raw, row, 0))

        if first_col.lower() == "driver:":
            current_driver = clean_text(cell(raw, row, 1))
            current_comment = clean_text(cell(raw, row, 16))
            continue

        tire_entry = first_col
        session = clean_text(cell(raw, row, 2))
        tire_type = extract_tire_type(tire_entry)

        is_a_row = normalize_marker(cell(raw, row, 6)) == "A:"
        is_t_row_below = normalize_marker(cell(raw, row + 1, 6)) == "T:"

        if (
            not session
            or tire_type not in VALID_TIRE_TYPES
            or not is_a_row
            or not is_t_row_below
        ):
            continue

        ambient_temp = first_number(
            cell(raw, row, 21),
            cell(raw, row, 7),
        )

        track_temp = first_number(
            cell(raw, row + 1, 21),
            cell(raw, row + 1, 7),
        )

        cold_temp = first_number(
            cell(raw, row, 7),
        )

        for (
            position,
            row_offset,
            cold_col,
            corr_col,
            bleed_col,
            hot_col,
        ) in POSITION_SPECS:
            source_row = row + row_offset

            rows.append(
                {
                    "event": event,
                    "track": track,
                    "session": session,
                    "driver": current_driver,
                    "car_model": car_model,
                    "tire_entry": tire_entry,
                    "tire_type": tire_type,
                    "position": position,
                    "ambient_temp": ambient_temp,
                    "track_temp": track_temp,
                    "cold_temp": cold_temp,
                    "cold_pressure": cell(raw, source_row, cold_col),
                    "cold_pressure_corr": cell(raw, source_row, corr_col),
                    "bleed_boost": cell(raw, source_row, bleed_col),
                    "hot_pressure": cell(raw, source_row, hot_col),
                    "laps_raw": clean_text(cell(raw, row, 17)),
                    "comment": current_comment,
                    "source_sheet": sheet_name,
                    "source_sheet_order": source_sheet_order,
                    "source_excel_row": source_row + 1,
                    "source_stint_row": row + 1,              
                }
            )

    return pd.DataFrame(rows)

def load_training_data(file_bytes: bytes) -> pd.DataFrame:
    excel_file = pd.ExcelFile(BytesIO(file_bytes))

    frames = []

    for source_sheet_order, sheet_name in enumerate(
        excel_file.sheet_names
    ):
        raw = pd.read_excel(
            excel_file,
            sheet_name=sheet_name,
            header=None,
        )

        parsed = parse_block_sheet(
            raw=raw,
            sheet_name=sheet_name,
            source_sheet_order=source_sheet_order,
        )

        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )
