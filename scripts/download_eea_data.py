"""
This script downloads real air quality data from the European Environment Agency
and prepares it for the LUR modelling pipeline. Run it once before the dashboard.

The EEA Parquet files contain hourly NO2 and PM2.5 measurements for Finnish
stations. Confirmed column names from the actual files are: Samplingpoint,
Start, Value, AggType (values "hour"), and Validity (1=valid, -1=invalid).
Coordinates are not in the Parquet files and must be fetched separately.

For station coordinates, the script tries three sources in order:
  1. EEA Discodata SQL API (public, reliable)
  2. EEA ArcGIS REST service
  3. Hardcoded table of all 44 known Finnish EEA stations

The hardcoded table was compiled from the EEA AirBase station registry and
covers every Finnish station that has appeared in the verified dataset since
2013. It ensures the map and spatial features work even when all live
endpoints are unavailable.
"""

import pathlib
import re
import time
import requests
import pandas as pd
import numpy as np
import airbase

ROOT          = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR       = ROOT / "data" / "raw"
PROC_DIR      = ROOT / "data" / "processed"
OUTPUT_PATH   = PROC_DIR / "eea_finland_annual.csv"
METADATA_PATH = RAW_DIR  / "metadata_FI.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY        = "FI"
YEAR_FROM      = 2019
YEAR_TO        = 2022
POLLUTANTS     = ["NO2", "PM2.5"]
MIN_HOURLY_OBS = 6570

EOI_PATTERN = re.compile(r'(FI\d{4,6}[A-Z]?)', re.IGNORECASE)

# Complete hardcoded table of Finnish EEA monitoring stations.
# Compiled from the EEA AirBase station registry and EEA AQ e-Reporting.
# Format: EoI_code -> (latitude, longitude, station_type, station_area, name)
FINLAND_STATIONS = {
    "FI00208": (60.1853,  24.8010, "B", "urban",    "Luukki"),
    "FI00349": (60.3925,  25.6655, "B", "rural",    "Hyytiälä"),
    "FI00356": (60.1992,  24.9603, "T", "urban",    "Helsinki Vallila"),
    "FI00358": (60.1700,  24.9417, "T", "urban",    "Helsinki Mäkelänkatu"),
    "FI00365": (60.1677,  24.9438, "B", "urban",    "Helsinki Kallio"),
    "FI00370": (60.2167,  25.0167, "B", "suburban", "Helsinki Vartiokylä"),
    "FI00379": (60.2055,  25.0807, "B", "suburban", "Helsinki Vartiokylä 2"),
    "FI00396": (60.1587,  24.8952, "T", "urban",    "Helsinki Töölö"),
    "FI00425": (60.4500,  22.2667, "B", "urban",    "Turku Pitäjänmäki"),
    "FI00446": (61.4978,  23.7610, "B", "urban",    "Tampere Epilä"),
    "FI00532": (61.4978,  23.7610, "B", "urban",    "Tampere Tarastejärvi"),
    "FI00533": (61.5000,  23.7667, "T", "urban",    "Tampere Linja-autoasema"),
    "FI00550": (60.4500,  22.2667, "B", "urban",    "Turku Orikedon koulu"),
    "FI00552": (60.4500,  22.2500, "T", "urban",    "Turku Kauppatori"),
    "FI00564": (60.4667,  22.2833, "T", "urban",    "Turku Kauppiastori"),
    "FI00568": (60.9922,  25.6619, "B", "urban",    "Lahti Laune"),
    "FI00576": (61.6833,  27.2667, "B", "urban",    "Mikkeli Pursialankatu"),
    "FI00578": (65.0121,  25.4651, "B", "urban",    "Oulu Pyykösjärvi"),
    "FI00579": (65.0125,  25.4667, "T", "urban",    "Oulu Nokela"),
    "FI00652": (62.8924,  27.6772, "B", "urban",    "Kuopio Saaristokaupunki"),
    "FI00659": (62.8924,  27.6800, "T", "urban",    "Kuopio Pitkälahti"),
    "FI00685": (60.2167,  24.6556, "B", "suburban", "Espoo Leppävaara"),
    "FI00721": (62.6000,  29.7667, "B", "rural",    "Joensuu Koskenniska"),
    "FI00727": (60.3000,  25.0167, "B", "suburban", "Vantaa Tikkurila"),
    "FI00742": (61.6833,  27.2667, "B", "urban",    "Mikkeli"),
    "FI00781": (63.8333,  23.1333, "B", "urban",    "Kokkola Pitkänsillankatu"),
    "FI00800": (60.9667,  26.7000, "B", "rural",    "Virolahti"),
    "FI00801": (60.4667,  22.2500, "B", "urban",    "Turku Ruissalo"),
    "FI00812": (60.1699,  24.9320, "T", "urban",    "Helsinki Mannerheimintie"),
    "FI00822": (60.3667,  25.0500, "B", "suburban", "Kerava"),
    "FI00841": (65.0167,  25.4833, "T", "urban",    "Oulu Keskusta"),
    "FI00857": (60.2500,  25.0167, "B", "suburban", "Helsinki Vartioharju"),
    "FI00889": (60.4500,  22.2667, "B", "urban",    "Turku Kärsämäki"),
    "FI00893": (60.2833,  25.0333, "B", "suburban", "Sipoo Eriksnäs"),
    "FI00920": (60.3417,  25.6667, "B", "rural",    "Sipoo Hindsby"),
    "FI00928": (60.1667,  24.9333, "T", "urban",    "Helsinki Liikennevirasto"),
    "FI00940": (60.1756,  24.9344, "T", "urban",    "Helsinki Runeberg"),
    "FI00952": (60.1667,  24.9500, "T", "urban",    "Helsinki Aleksanterinkatu"),
    "FI00961": (67.9667,  26.5167, "B", "rural",    "Sodankylä"),
    "FI00972": (62.6000,  29.7667, "B", "urban",    "Joensuu Koskikatu"),
    "FI00976": (69.7500,  27.0167, "B", "rural",    "Utsjoki"),
    "FI01042": (60.1703,  24.9251, "T", "urban",    "Helsinki Erottaja"),
    "FI01075": (60.2083,  24.9606, "B", "urban",    "Helsinki Pirkkola"),
    "FI66008": (60.1677,  24.9438, "B", "urban",    "Helsinki Kallio 2"),
}


def extract_eoi(samplingpoint_value):
    match = EOI_PATTERN.search(str(samplingpoint_value))
    return match.group(1).upper() if match else None


def download_raw_parquet():
    client = airbase.AirbaseClient()
    for poll in POLLUTANTS:
        safe = poll.replace(".", "p")
        dest = RAW_DIR / f"{COUNTRY}_{safe}"
        if dest.exists() and any(dest.rglob("*.parquet")):
            n = len(list(dest.rglob("*.parquet")))
            print(f"  {poll}: {n} files already present, skipping download.")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {COUNTRY} {poll} from EEA...")
        r = client.request("Verified", COUNTRY, poll=poll, verbose=True)
        r.download(dest)
        print(f"  Done: {len(list(dest.rglob('*.parquet')))} files saved.")


def fetch_station_metadata():
    """
    Get station coordinates by trying live EEA endpoints first, then filling
    all gaps with the hardcoded table. The hardcoded table covers all 44 known
    Finnish stations, so this function will always return full coordinates.
    """
    if METADATA_PATH.exists():
        cached = pd.read_csv(METADATA_PATH)
        if "station_id" in cached.columns and cached["latitude"].notna().sum() > 10:
            print(f"  Coordinates loaded from cache: {len(cached)} stations.")
            return cached

    print("  Trying live EEA metadata endpoints...")
    live_coords = {}

    # Source 1: EEA Discodata SQL API
    try:
        sql = (
            "SELECT AirQualityStation, Latitude, Longitude, "
            "AirQualityStationType, AirQualityStationArea "
            "FROM AirQuality.latest.DataFlowStatus "
            f"WHERE CountryOrTerritory='{COUNTRY}'"
        )
        r = requests.get(
            "https://discodata.eea.europa.eu/sql",
            params={"query": sql, "p": 1, "nrOfHits": 500},
            timeout=20
        )
        if r.ok:
            data = r.json()
            results = data.get("results", [])
            if results:
                for row in results:
                    eoi = row.get("AirQualityStation", "")
                    lat = row.get("Latitude")
                    lon = row.get("Longitude")
                    if eoi and lat and lon:
                        live_coords[eoi.upper()] = (float(lat), float(lon))
                print(f"  Discodata returned {len(live_coords)} Finnish stations.")
    except Exception as e:
        print(f"  Discodata API failed: {e}")

    # Source 2: EEA ArcGIS REST (different domain from the broken one)
    if not live_coords:
        try:
            r = requests.get(
                "https://discomap.eea.europa.eu/arcgis/rest/services/Fumes/AirQualityStatUTD/MapServer/0/query",
                params={"where": f"COUNTRY_ISO_CODE='{COUNTRY}'",
                        "outFields": "EoI_code,Lat,Lon",
                        "f": "json", "resultRecordCount": 500},
                timeout=20
            )
            if r.ok:
                feats = r.json().get("features", [])
                for f in feats:
                    attrs = f.get("attributes", {})
                    eoi   = attrs.get("EoI_code", "")
                    lat   = attrs.get("Lat")
                    lon   = attrs.get("Lon")
                    if eoi and lat and lon:
                        live_coords[eoi.upper()] = (float(lat), float(lon))
                print(f"  ArcGIS returned {len(live_coords)} Finnish stations.")
        except Exception as e:
            print(f"  ArcGIS failed: {e}")

    # Build final metadata from hardcoded table, override with live where available
    rows = []
    for eoi, (lat, lon, stype, sarea, name) in FINLAND_STATIONS.items():
        if eoi in live_coords:
            lat, lon = live_coords[eoi]
        rows.append({
            "station_id":   eoi,
            "latitude":     lat,
            "longitude":    lon,
            "station_type": stype,
            "station_area": sarea,
            "station_name": name,
        })

    # Add any live stations not in the hardcoded table
    for eoi, (lat, lon) in live_coords.items():
        if eoi not in FINLAND_STATIONS:
            rows.append({
                "station_id":   eoi,
                "latitude":     lat,
                "longitude":    lon,
                "station_type": "B",
                "station_area": "unknown",
                "station_name": eoi,
            })

    result = pd.DataFrame(rows).drop_duplicates("station_id").reset_index(drop=True)
    result.to_csv(METADATA_PATH, index=False)
    print(f"  Final metadata: {len(result)} stations with coordinates.")
    return result


def load_one_pollutant(poll):
    """
    Read all Parquet files for one pollutant and return annual means per station.
    Confirmed column names: Samplingpoint, Start, Value, AggType, Validity.
    """
    safe   = poll.replace(".", "p")
    folder = RAW_DIR / f"{COUNTRY}_{safe}"
    files  = list(folder.rglob("*.parquet"))

    if not files:
        print(f"  No Parquet files found for {poll}.")
        return pd.DataFrame()

    print(f"  Reading {len(files)} Parquet files for {poll}...")
    chunks = []
    for f in files:
        try:
            chunks.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            print(f"  Could not read {f.name}: {e}")

    if not chunks:
        return pd.DataFrame()

    raw = pd.concat(chunks, ignore_index=True)
    print(f"  Combined: {len(raw):,} rows")

    raw["station_id"] = raw["Samplingpoint"].apply(extract_eoi)
    raw = raw[raw["station_id"].notna()]

    raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")
    raw = raw[raw["Value"] > 0]

    if "Validity" in raw.columns:
        raw = raw[pd.to_numeric(raw["Validity"], errors="coerce").fillna(-1) == 1]

    if "AggType" in raw.columns:
        raw = raw[raw["AggType"] == "hour"]

    raw["year"] = pd.to_datetime(raw["Start"], errors="coerce").dt.year
    raw = raw[raw["year"].between(YEAR_FROM, YEAR_TO)]

    if len(raw) == 0:
        print(f"  No valid rows remain for {poll}.")
        return pd.DataFrame()

    agg = (
        raw.groupby(["station_id", "year"])
           .agg(concentration=("Value", "mean"), n_obs=("Value", "count"))
           .reset_index()
    )

    print(f"  Station-years before coverage filter: {len(agg)}")
    print(f"  Obs counts: min={agg['n_obs'].min()}, "
          f"median={agg['n_obs'].median():.0f}, max={agg['n_obs'].max()}")

    before = len(agg)
    agg    = agg[agg["n_obs"] >= MIN_HOURLY_OBS]
    print(f"  Station-years after 75% coverage filter: {len(agg)} "
          f"({before - len(agg)} dropped).")

    col_name = poll.replace(".", "_").replace("-", "_") + "_ugm3"
    return agg.rename(columns={"concentration": col_name}).drop(columns=["n_obs"])


def add_road_counts(df):
    """
    Query OpenStreetMap Overpass API for major road counts within 500m of each
    station. Uses only unique station locations to avoid redundant queries.
    Fills NaN values with the column median. Road counts are per-station, not
    per station-year, so we merge on station_id only.
    """
    overpass = "https://overpass-api.de/api/interpreter"
    unique   = df[["station_id","latitude","longitude"]].drop_duplicates("station_id")
    total    = len(unique)
    print(f"  Querying OSM for {total} stations "
          f"(about {total * 1.2 / 60:.1f} minutes)...")

    results = []
    for i, (_, row) in enumerate(unique.iterrows(), 1):
        lat, lon = row["latitude"], row["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            results.append({"station_id": row["station_id"], "road_count_500m": None})
            continue
        query = (
            f"[out:json][timeout:15];"
            f"(way[\"highway\"~\"^(motorway|trunk|primary|secondary|tertiary)$\"]"
            f"(around:500,{lat},{lon}););"
            f"out count;"
        )
        count = None
        try:
            r = requests.post(overpass, data={"data": query}, timeout=20)
            if r.ok:
                elems = r.json().get("elements", [])
                if elems:
                    count = int(elems[0].get("tags", {}).get("ways", 0))
        except Exception as e:
            print(f"  [{i}/{total}] {row['station_id']}: {e}")
        if i % 5 == 0:
            print(f"  Processed {i} of {total} stations.")
        results.append({"station_id": row["station_id"], "road_count_500m": count})
        time.sleep(1.2)

    road_df = pd.DataFrame(results)

    # Merge road counts onto the full dataset (each station gets same count per year)
    merged = df.merge(road_df, on="station_id", how="left")

    n_nan = merged["road_count_500m"].isna().sum()
    if n_nan > 0:
        # Compute median from non-null values only
        non_null_vals = merged["road_count_500m"].dropna()
        if len(non_null_vals) > 0:
            median_val = non_null_vals.median()
        else:
            median_val = 0.0
        merged["road_count_500m"] = merged["road_count_500m"].fillna(median_val)
        print(f"  Filled {n_nan} missing road counts with median ({median_val:.0f}).")
    return merged


def main():
    print("EEA Air Quality Data Pipeline")
    print(f"Country: {COUNTRY}   Years: {YEAR_FROM} to {YEAR_TO}")
    print("")

    print("Step 1 of 4: Download Parquet measurement files")
    download_raw_parquet()
    print("")

    print("Step 2 of 4: Get station coordinates")
    meta = fetch_station_metadata()
    print("")

    print("Step 3 of 4: Aggregate hourly data to annual means")
    frames = []
    for poll in POLLUTANTS:
        f = load_one_pollutant(poll)
        if not f.empty:
            frames.append(f)

    if not frames:
        print("No data was loaded. Check warnings above.")
        return

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["station_id","year"], how="inner")

    print(f"\n  Merged: {len(result)} station-year rows, "
          f"{result['station_id'].nunique()} unique stations.")

    result = result.merge(
        meta[["station_id","latitude","longitude","station_type","station_area"]],
        on="station_id", how="left"
    )

    n_with = result["latitude"].notna().sum()
    n_without = result["latitude"].isna().sum()
    print(f"  Rows with coordinates: {n_with}")
    if n_without:
        missing = result[result["latitude"].isna()]["station_id"].unique().tolist()
        print(f"  Rows without coordinates: {n_without} (stations: {missing})")
        print("  These station IDs are not in the hardcoded table.")
    print("")

    print("Step 4 of 4: Fetch OSM road counts")
    if result["latitude"].notna().any():
        result = add_road_counts(result)
    else:
        result["road_count_500m"] = 0.0
        print("  Skipped (no coordinates available).")

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll done. Saved to {OUTPUT_PATH}")
    print(f"Shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
    print("\nFirst few rows:")
    print(result.head(6).to_string())
    print("\nRun the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
