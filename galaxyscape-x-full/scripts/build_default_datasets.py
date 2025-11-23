#!/usr/bin/env python3
"""
Build Default Datasets for PolarisCapital Analytics
===================================================

This script downloads real astronomy and finance data from public sources and
produces the canonical default CSVs that the dashboards rely on when the user
has not uploaded their own files.

Astronomy data source:
    HYG Database (Hipparcos + Tycho + Gaia composite catalog)

Finance data source:
    Yahoo Finance historical prices via yfinance

Outputs:
    data/raw/astronomy/default_astronomy_dataset.csv
    data/raw/finance/default_finance_dataset.csv
    data/data_index.json  (updated with default references)

Run:
    python scripts/build_default_datasets.py
"""

from __future__ import annotations

import argparse
import json
import math
import ssl
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ASTRO_DIR = RAW_DIR / "astronomy"
FIN_DIR = RAW_DIR / "finance"
ASTRO_DEFAULT = ASTRO_DIR / "default_astronomy_dataset.csv"
FIN_DEFAULT = FIN_DIR / "default_finance_dataset.csv"
DATA_INDEX = DATA_DIR / "data_index.json"


ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DIR, ASTRO_DIR, FIN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Astronomy Dataset Builders
# ---------------------------------------------------------------------------

HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/v3/hyg_v38.csv.gz"


def download_astronomy_catalog() -> pd.DataFrame:
    """Download stellar parameters from the public HYG catalog (Hipparcos+Gaia)."""
    response = requests.get(HYG_URL, timeout=60, verify=False)
    response.raise_for_status()
    df = pd.read_csv(BytesIO(response.content), compression="gzip")
    if df.empty:
        raise RuntimeError("HYG catalog returned zero rows")
    return df


def _spectral_class(temp: float) -> str:
    """Classify star by temperature similar to Morgan–Keenan system."""
    if math.isnan(temp):
        return "Unknown"
    if temp >= 30000:
        return "O"
    if temp >= 10000:
        return "B"
    if temp >= 7500:
        return "A"
    if temp >= 6000:
        return "F"
    if temp >= 5200:
        return "G"
    if temp >= 3700:
        return "K"
    return "M"


def build_astronomy_default(limit: int = 4000) -> Tuple[Path, int]:
    """Create a single astronomy CSV with all required dashboard features."""
    print("🔭 Downloading HYG stellar catalog…")
    df_raw = download_astronomy_catalog()

    rename_map = {
        "id": "star_id",
        "proper": "star_name",
        "ra": "ra",
        "dec": "dec",
        "dist": "distance_pc",
        "mag": "apparent_magnitude",
        "absmag": "absolute_magnitude",
        "ci": "color_index",
        "lum": "luminosity",
        "spect": "spectral_class",
        "var": "variability_type",
        "var_min": "var_min",
        "var_max": "var_max",
    }
    df = df_raw.rename(columns=rename_map)
    df["star_id"] = np.arange(1, len(df) + 1)
    df["star_name"] = df["star_name"].fillna(df["bf"]).fillna(df["gl"]).fillna(df["hip"]).fillna("Unnamed Star")

    df = df.dropna(subset=["ra", "dec"]).copy()

    # Limit dataset to keep file size reasonable while still representative
    if limit and len(df) > limit:
        df = df.head(limit)

    required_cols = [
        "star_name",
        "temperature",
        "mass",
        "radius",
        "luminosity",
        "stellar_age",
        "ra",
        "dec",
    ]

    # Derive additional astrophysical properties from available columns
    df["name"] = df["star_name"]
    df["color_index"] = pd.to_numeric(df["color_index"], errors="coerce")
    df["apparent_magnitude"] = pd.to_numeric(df["apparent_magnitude"], errors="coerce")
    df["absolute_magnitude"] = pd.to_numeric(df["absolute_magnitude"], errors="coerce")
    df["luminosity"] = pd.to_numeric(df.get("luminosity"), errors="coerce")

    # If luminosity missing, compute from absolute magnitude (Sun absolute magnitude ≈ 4.83)
    missing_lum = df["luminosity"].isna()
    if missing_lum.any():
        df.loc[missing_lum, "luminosity"] = 10 ** ((4.83 - df.loc[missing_lum, "absolute_magnitude"]) / 2.5)

    # Approximate temperature via Ballesteros' formula using B-V color index
    def bv_to_temp(bv: float) -> float:
        if pd.isna(bv):
            return np.nan
        return 4600 * ((1 / (0.92 * bv + 1.7)) + (1 / (0.92 * bv + 0.62)))

    df["temperature"] = df["color_index"].apply(bv_to_temp)
    df["temperature"] = df["temperature"].fillna(df["temperature"].median(skipna=True))

    # Estimate mass and radius from luminosity (rough astrophysical scaling)
    df["mass"] = (df["luminosity"].clip(lower=1e-3)) ** (1 / 3.5)
    df["radius"] = np.sqrt(df["luminosity"].clip(lower=1e-6)) * ((5778 / df["temperature"]) ** 2)

    # Estimate stellar age inversely related to mass (massive stars live shorter)
    df["stellar_age"] = (10 * (df["mass"].clip(lower=0.1) ** -2.5)).clip(0.01, 15)

    # Approximate metallicity by mapping spectral class (purely heuristic but deterministic)
    spectral_to_metallicity = {
        "O": -0.3,
        "B": -0.2,
        "A": -0.1,
        "F": -0.05,
        "G": 0.0,
        "K": 0.1,
        "M": 0.2,
    }
    df["metallicity"] = df["spectral_class"].str.upper().str[0].map(spectral_to_metallicity).fillna(0.0)

    # Rotation period heuristic from mass
    df["rotation_period"] = (24 * (df["mass"].clip(lower=0.1) ** -0.5)).clip(0.1, 90)
    df["period"] = df["rotation_period"]

    df["brightness"] = df["luminosity"]
    df["magnitude"] = df["apparent_magnitude"]

    # Spectral class + habitability heuristic (temperature + radius sweet spot)
    df["spectral_class"] = df["temperature"].apply(lambda t: _spectral_class(float(t)) if pd.notna(t) else "Unknown")
    hab_temp = np.exp(-((df["temperature"] - 5778) ** 2) / (2 * 800 ** 2))
    hab_radius = np.exp(-((df["radius"] - 1) ** 2) / (2 * 0.5 ** 2))
    df["habitability_index"] = (0.7 * hab_temp + 0.3 * hab_radius).fillna(0).round(3)

    # Prepare features for clustering/anomaly detection
    feature_cols = ["temperature", "mass", "radius", "luminosity", "metallicity", "rotation_period"]
    feature_frame = df[feature_cols].copy()
    feature_frame = feature_frame.fillna(feature_frame.median())
    feature_frame = (feature_frame - feature_frame.mean()) / feature_frame.std(ddof=0)
    feature_frame = feature_frame.fillna(0)

    # Clustering
    n_clusters = min(6, max(3, len(df) // 400))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(feature_frame)
    except Exception:
        df["cluster"] = 0

    # Anomaly detection
    try:
        iso = IsolationForest(contamination=0.035, random_state=42)
        df["anomaly_score"] = iso.fit(feature_frame).decision_function(feature_frame)
        df["is_anomaly"] = (iso.predict(feature_frame) == -1).astype(int)
    except Exception:
        df["anomaly_score"] = 0.0
        df["is_anomaly"] = 0

    # 2D embedding for sky network
    try:
        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(feature_frame)
        df["embedding_x"] = embedding[:, 0]
        df["embedding_y"] = embedding[:, 1]
    except Exception:
        df["embedding_x"] = 0.0
        df["embedding_y"] = 0.0

    # Derived flux for light curve visuals
    df["flux"] = (10 ** (-0.4 * (df["magnitude"] - df["magnitude"].min()))).round(6)

    # Ensure all required columns exist
    required = set(required_cols + [
        "name",
        "rotation_period",
        "period",
        "color_index",
        "magnitude",
        "cluster",
        "anomaly_score",
        "embedding_x",
        "embedding_y",
        "habitability_index",
        "flux",
        "spectral_class",
    ])
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["dataset_source"] = "HYG Database (Hipparcos/Tycho/Gaia)"
    df["record_type"] = "stellar"

    ASTRO_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ASTRO_DEFAULT, index=False)
    print(f"✅ Saved astronomy default dataset ({len(df):,} rows) -> {ASTRO_DEFAULT}")
    return ASTRO_DEFAULT, len(df)


# ---------------------------------------------------------------------------
# Finance Dataset Builders
# ---------------------------------------------------------------------------

TICKER_META: Dict[str, Dict[str, str]] = {
    "AAPL": {"sector": "Technology", "channel": "App Store", "region": "North America", "country": "USA", "lat": 37.3349, "lon": -122.0090},
    "MSFT": {"sector": "Technology", "channel": "Cloud / Azure", "region": "North America", "country": "USA", "lat": 47.6427, "lon": -122.1313},
    "GOOGL": {"sector": "Communication Services", "channel": "Search Ads", "region": "North America", "country": "USA", "lat": 37.4220, "lon": -122.0841},
    "AMZN": {"sector": "Consumer Discretionary", "channel": "E-Commerce", "region": "North America", "country": "USA", "lat": 47.6221, "lon": -122.3366},
    "TSLA": {"sector": "Consumer Discretionary", "channel": "Direct-to-Consumer", "region": "North America", "country": "USA", "lat": 37.3947, "lon": -122.1503},
    "JNJ": {"sector": "Healthcare", "channel": "Healthcare Retail", "region": "North America", "country": "USA", "lat": 40.4981, "lon": -74.4469},
    "JPM": {"sector": "Financials", "channel": "Wholesale Banking", "region": "North America", "country": "USA", "lat": 40.7550, "lon": -73.9840},
    "BAC": {"sector": "Financials", "channel": "Retail Banking", "region": "North America", "country": "USA", "lat": 35.2271, "lon": -80.8431},
    "XOM": {"sector": "Energy", "channel": "Energy Distribution", "region": "North America", "country": "USA", "lat": 29.7360, "lon": -95.5170},
    "WMT": {"sector": "Consumer Staples", "channel": "Omni-Retail", "region": "North America", "country": "USA", "lat": 36.3654, "lon": -94.2196},
    "SAP": {"sector": "Technology", "channel": "Enterprise SaaS", "region": "Europe", "country": "Germany", "lat": 49.2937, "lon": 8.6411},
    "BHP": {"sector": "Materials", "channel": "Industrial Supply", "region": "Australia", "country": "Australia", "lat": -37.8200, "lon": 144.9600},
}

STOOQ_SYMBOLS = {
    "AAPL": "aapl.us",
    "MSFT": "msft.us",
    "GOOGL": "googl.us",
    "AMZN": "amzn.us",
    "TSLA": "tsla.us",
    "JNJ": "jnj.us",
    "JPM": "jpm.us",
    "BAC": "bac.us",
    "XOM": "xom.us",
    "WMT": "wmt.us",
    "SAP": "sap.us",
    "BHP": "bhp.us",
}

def download_price_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download historical OHLCV data for tickers via Stooq daily CSV export."""
    frames = []
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    for ticker in tickers:
        print(f"  ⬇️  {ticker}")
        symbol = STOOQ_SYMBOLS.get(ticker, f"{ticker.lower()}.us")
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        resp = requests.get(url, timeout=60, verify=False)
        if resp.status_code != 200 or "Date,Open,High" not in resp.text:
            print(f"    ⚠️  Stooq returned status {resp.status_code} for {ticker}")
            continue
        df_ticker = pd.read_csv(StringIO(resp.text))
        if df_ticker.empty:
            print(f"    ⚠️  No data returned for {ticker}")
            continue
        df_ticker["Date"] = pd.to_datetime(df_ticker["Date"])
        df_ticker = df_ticker[(df_ticker["Date"] >= start_dt) & (df_ticker["Date"] <= end_dt)]
        if df_ticker.empty:
            print(f"    ⚠️  No data in requested window for {ticker}")
            continue
        df_ticker["Ticker"] = ticker
        frames.append(df_ticker)

    if not frames:
        raise RuntimeError("Failed to download any finance data")

    df = pd.concat(frames, ignore_index=True)
    df.rename(
        columns={
            "Adj Close": "AdjClose",
        },
        inplace=True,
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _compute_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns, volatility, Sharpe, drawdown, marketing proxies, etc."""
    df = df.sort_values(["Ticker", "Date"]).copy()

    # Map metadata
    meta_df = pd.DataFrame.from_dict(TICKER_META, orient="index")
    meta_df["Ticker"] = meta_df.index
    df = df.merge(meta_df, on="Ticker", how="left")
    df["sector"] = df["sector"].fillna("Unknown")

    group = df.groupby("Ticker", group_keys=False)
    df["returns"] = group["Close"].pct_change().fillna(0)
    df["rolling_vol_21"] = (group["returns"].rolling(window=21).std().reset_index(level=0, drop=True) * np.sqrt(252)).bfill()
    df["rolling_return_21"] = group["returns"].rolling(window=21).mean().reset_index(level=0, drop=True)
    df["rolling_sharpe_21"] = np.where(df["rolling_vol_21"] > 0, df["rolling_return_21"] / df["rolling_vol_21"], 0)

    # Max drawdown per ticker
    cum = (1 + df["returns"]).groupby(df["Ticker"]).cumprod()
    rolling_max = cum.groupby(df["Ticker"]).cummax()
    drawdown = (cum - rolling_max) / rolling_max
    df["max_drawdown"] = drawdown.groupby(df["Ticker"]).cummin().fillna(0)

    # Value at Risk (historical)
    df["value_at_risk_95"] = group["returns"].rolling(window=60).quantile(0.05).reset_index(level=0, drop=True)

    # Risk score scaled 0-100 (volatility + drawdown + VaR)
    vol_norm = (df["rolling_vol_21"] - df["rolling_vol_21"].min()) / (df["rolling_vol_21"].max() - df["rolling_vol_21"].min() + 1e-9)
    dd_norm = (df["max_drawdown"] - df["max_drawdown"].min()) / (df["max_drawdown"].max() - df["max_drawdown"].min() + 1e-9)
    var_norm = (df["value_at_risk_95"] - df["value_at_risk_95"].min()) / (df["value_at_risk_95"].max() - df["value_at_risk_95"].min() + 1e-9)
    df["risk_score"] = ((0.5 * vol_norm + 0.3 * (-dd_norm) + 0.2 * (-var_norm)) * 100).clip(0, 100).fillna(0)

    # Marketing-style metrics derived from real signals
    df["roi"] = (df["returns"] * 100).round(3)
    df["cost"] = (df["Close"] * 12).round(2)
    df["revenue"] = (df["cost"] * (1 + df["returns"].clip(lower=-0.5))).round(2)
    df["impressions"] = (df["Volume"] / 1_000).round().astype(int)
    df["conversion_rate"] = np.clip((df["returns"] * 50) + 5, 0.5, 40).round(2)
    df["channel"] = df["channel"].fillna("Digital")
    df["campaign"] = df["Ticker"] + "-" + df["channel"].str.replace(" ", "_")

    # Geographic coordinates (fallback to 0,0)
    df["geo_lat"] = df["lat"].fillna(0.0)
    df["geo_lon"] = df["lon"].fillna(0.0)

    # Compliance/Audit style signals
    df["audit_flag"] = np.where(df["risk_score"] > 75, "Review", "OK")
    df["risk_band"] = pd.cut(
        df["risk_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["Low", "Moderate", "High", "Critical"],
    )

    return df


def build_finance_default(years: int = 2) -> Tuple[Path, int]:
    """Create finance CSV with comprehensive metrics derived from real prices."""
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=365 * years)

    print("💹 Downloading Yahoo Finance history…")
    df_prices = download_price_history(list(TICKER_META.keys()), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
    df_enriched = _compute_finance_features(df_prices)

    # Date ranges for filters
    df_enriched["date"] = df_enriched["Date"].dt.strftime("%Y-%m-%d")
    df_enriched.rename(columns={"Ticker": "ticker", "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"}, inplace=True)

    FIN_DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(FIN_DEFAULT, index=False)
    print(f"✅ Saved finance default dataset ({len(df_enriched):,} rows) -> {FIN_DEFAULT}")
    return FIN_DEFAULT, len(df_enriched)


# ---------------------------------------------------------------------------
# Data Index Utilities
# ---------------------------------------------------------------------------

def update_data_index() -> None:
    """Ensure data_index.json references the default datasets."""
    if DATA_INDEX.exists():
        with open(DATA_INDEX, "r") as fh:
            data = json.load(fh)
    else:
        data = {}

    defaults = {
        "astronomy": str(ASTRO_DEFAULT.relative_to(PROJECT_ROOT)),
        "finance": str(FIN_DEFAULT.relative_to(PROJECT_ROOT)),
    }
    data.setdefault("defaults", defaults)
    data["defaults"] = defaults

    with open(DATA_INDEX, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"🗂  Updated data index -> {DATA_INDEX}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PolarisCapital default datasets")
    parser.add_argument("--astronomy", action="store_true", help="Only rebuild astronomy dataset")
    parser.add_argument("--finance", action="store_true", help="Only rebuild finance dataset")
    return parser.parse_args()


def main() -> None:
    _ensure_directories()
    args = parse_args()

    run_astronomy = args.astronomy or not (args.astronomy or args.finance)
    run_finance = args.finance or not (args.astronomy or args.finance)

    if run_astronomy:
        build_astronomy_default()
    if run_finance:
        build_finance_default()

    update_data_index()
    print("Done ✅")


if __name__ == "__main__":
    main()

