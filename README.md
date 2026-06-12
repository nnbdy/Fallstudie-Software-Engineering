# Fallstudie-Software-Engineering
Eine KI, die am Ende für jeden Fahrer je nach Strecken und Außentemperatur einen Einstelldruck ausgibt

# Dateien
app.py: Streamlit-Oberfläche und Modell
excel_parser.py: automatischer Import der vorhandenen Rennblätter
requirements.txt: Python-Pakete

# Unterstützte Reifenarten:

| Code  | Bedeutung   |
| ----- | ----------- |
| `DM`  | Medium      |
| `DH`  | Hard        |
| `WUS` | Regenreifen |

# Erkannte Reifenpositionen:

| Code  | Bedeutung          |
| ----- | ------------------ |
| `V_L` | Vorderachse links  |
| `V_R` | Vorderachse rechts |
| `H_L` | Hinterachse links  |
| `H_R` | Hinterachse rechts |

## Interface-Eingaben

Im Interface werden folgende Werte eingegeben:

| Eingabe            | Bedeutung                                 |
| ------------------ | ----------------------------------------- |
| Strecke            | Strecke oder Event                        |
| Außentemperatur    | aktuelle Lufttemperatur                   |
| Streckentemperatur | aktuelle Streckentemperatur |
| Reifenart          | `DM`, `DH` oder `WUS`                     |
| Fahrer             | optionaler Fahrerfilter                   |
| Fahrzeug           | `BMW`, `Supra` oder `Porsche`             |
| Zieldruck          | gewünschter Druck nach dem Stint          |

## Berechnungslogik

### 1. Normierung des Kaltdrucks auf 10 °C

Damit historische Werte vergleichbar sind, wird der Kaltdruck auf 10 °C normiert:

Kaltdruck @10°C =
(Kaltdruck + 1)
× ((10 + 273,15) / (Temperatur bei Kaltdruckmessung + 273,15))
- 1

Falls die Excel-Datei bereits einen korrigierten Kaltdruck enthält, wird dieser Wert übernommen

### 2. Historischer Druckaufbau

Das Modell lernt nicht direkt den Einstelldruck, sondern den Druckaufbau während eines Stints:

Druckaufbau =
Heißdruck
- (Kaltdruck @10°C + Bleed/Boost)

### 3. Relevante Einflussfaktoren

Das Modell verwendet folgende Faktoren:

| Faktor             | Relevanz      | Verwendung                                            |
| ------------------ | ------------- | ----------------------------------------------------- |
| Außentemperatur    | sehr hoch     | beeinflusst den Druckaufbau                           |
| Streckentemperatur | sehr hoch     | beeinflusst die Erwärmung des Reifens                 |
| Reifenart          | sehr hoch     | trennt `DM`, `DH` und `WUS`                           |
| Strecke            | hoch          | berücksichtigt streckenspezifische Belastungen        |
| Reifenposition     | hoch          | erzeugt getrennte Empfehlungen je Position            |
| Fahrer             | optional      | wird nur als kleiner Korrekturfaktor verwendet        |
| Zieldruck          | Endberechnung | wird nicht trainiert, sondern im Interface eingegeben |
| Fahrzeug           | Endberechnung | aktiviert fahrzeugspezifische Regeln                  |

Als Modell wird ein `RandomForestRegressor` verwendet

### 4. Fahrer-Offset

Falls ausreichend historische Daten für einen Fahrer vorhanden sind, wird ein kleiner Offset ergänzt:

Finaler Druckaufbau =
Basis-Druckaufbau
+ Fahrer-Offset

Der Fahrer-Offset ist begrenzt auf:
-0,05 bar bis +0,05 bar

Wenn nicht genügend passende Fahrer-Daten vorhanden sind, wird kein Fahrer-Offset angewendet

### 5. Einstelldruck berechnen

Der Zielddruck wird erst nach der Modellvorhersage verwendet:

Einstelldruck @10°C =
Zielruck
- finaler Druckaufbau

## Fahrzeuglogik

### BMW und Supra

Für BMW und Supra wird die Modell-Empfehlung unverändert übernommen

### Porsche

Für den Porsche werden die hinteren Werte fest von den vorderen Werten abgeleitet:

H_L = V_L - 0,05 bar
H_R = V_R - 0,05 bar
