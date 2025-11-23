"""
Data Context Engine
Analyzes uploaded CSVs to detect schema, infer domain, and compute dashboard capability.
This keeps dashboards from rendering empty visuals by describing what is missing and
what can be shown with the columns that exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

# Canonical feature names mapped to common aliases across domains
# Canonical feature aliases (teaches the engine how to map messy CSV headers)
# Example: "bp_rp" -> "color_index", "close" -> "price"
FEATURE_ALIASES: Dict[str, Set[str]] = {
    "id": {"id", "star_id", "object_id", "source_id"},
    "age": {"age", "stellar_age", "age_gyr", "star_age"},
    "temperature": {"temperature", "teff", "temp"},
    "mass": {"mass", "stellar_mass"},
    "rotation_period": {"rotation_period", "rot_period", "period", "rotation"},
    "color_index": {"color_index", "bp_rp", "bp-rp", "color", "colorindex"},
    "magnitude": {"magnitude", "mag", "g_mag", "bp_mag"},
    "ra": {"ra", "right_ascension"},
    "dec": {"dec", "declination"},
    "cluster": {"cluster", "cluster_id"},
    "anomaly_score": {"anomaly_score", "is_anomaly"},
    "price": {"price", "close", "close_price", "adj_close", "last"},
    "returns": {"returns", "return", "pct_change", "change_pct"},
    "volume": {"volume", "vol"},
    "sector": {"sector", "industry"},
    "date": {"date", "timestamp", "datetime"},
    "roi": {"roi", "return_on_investment"},
    "risk": {"risk", "risk_score", "volatility"},
    "ticker": {"ticker", "symbol"},
}

# Minimal field requirements per dashboard so we can compute availability
ASTRO_REQUIREMENTS: Dict[str, Set[str]] = {
    "overview": {"age"},
    "star-age": {"age"},
    "sky-network": {"cluster", "temperature"},
    "star-explorer": {"rotation_period", "color_index", "mass"},
    "sky-map": {"ra", "dec"},
    "light-curve": {"magnitude", "rotation_period"},
    "clusters": {"cluster"},
    "anomalies": {"temperature", "mass"},
    "ml-models": {"age", "temperature", "mass", "color_index"},
}

FIN_REQUIREMENTS: Dict[str, Set[str]] = {
    "risk": {"returns", "price", "date"},
    "streaming": {"returns", "price", "date"},
    "correlation": {"returns", "price"},
    "portfolio": {"price", "ticker", "sector"},
    "compliance": set(),
    "stock-explorer": {"price", "ticker"},
    "future-outcomes": {"price", "returns"},
    "marketing-analytics": {"roi"},
    "ml-models": {"returns", "price"},
    "game-theory": {"returns", "price"},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_DATASETS = {
    "astronomy": _project_root() / "data" / "raw" / "astronomy" / "default_astronomy_dataset.csv",
    "finance": _project_root() / "data" / "raw" / "finance" / "default_finance_dataset.csv",
}


def _candidate_paths(domain: Optional[str] = None) -> List[Path]:
    """Return ordered candidate CSV paths for the given domain."""
    base = _project_root()
    candidates: List[Path] = []
    domain_names = [domain] if domain else ["astronomy", "finance"]
    for d in domain_names:
        default_path = DEFAULT_DATASETS.get(d)
        if default_path and default_path.exists():
            candidates.append(default_path)
        uploads = base / "uploads" / d
        raw = base / "data" / "raw" / d
        for folder in [uploads, raw]:
            if folder.exists():
                for file in sorted(folder.glob("*.csv"), reverse=True):
                    if default_path and file == default_path:
                        continue
                    candidates.append(file)
    return candidates


def _load_preview(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, nrows=2000)
    except Exception:
        return pd.DataFrame()


def get_dataset_schema(file_path: Optional[str] = None, domain: Optional[str] = None) -> Dict:
    """Return a lightweight schema description (columns, dtypes, row count)."""
    path: Optional[Path] = Path(file_path) if file_path else None
    if path is None or not path.exists():
        candidates = _candidate_paths(domain)
        path = candidates[0] if candidates else None

    if path is None or not path.exists():
        return {
            "path": None,
            "columns": [],
            "dtypes": {},
            "row_count": 0,
        }

    df = _load_preview(path)
    return {
        "path": str(path),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "row_count": len(df),
    }


def infer_domain(columns: List[str]) -> str:
    """Infer domain by matching known feature markers."""
    lowered = {c.lower() for c in columns}
    astro_hits = lowered & {
        "bp_rp",
        "g_mag",
        "ra",
        "dec",
        "rotation_period",
        "stellar_age",
        "metallicity",
        "luminosity",
    }
    finance_hits = lowered & {
        "ticker",
        "close",
        "price",
        "volume",
        "sector",
        "returns",
        "beta",
    }
    if len(astro_hits) > len(finance_hits):
        return "astronomy"
    if len(finance_hits) > len(astro_hits):
        return "finance"
    return "unknown"


def auto_feature_mapping(columns: List[str]) -> Dict[str, str]:
    """Map raw column names to canonical semantic features."""
    mapping: Dict[str, str] = {}
    for raw in columns:
        lowered = raw.lower()
        for canonical, aliases in FEATURE_ALIASES.items():
            if lowered in aliases:
                mapping[canonical] = raw
                break
    return mapping


def _availability(
    available_features: Set[str], requirements: Dict[str, Set[str]]
) -> Tuple[Dict[str, Dict], float]:
    availability: Dict[str, Dict] = {}
    all_required: Set[str] = set()
    for needed in requirements.values():
        all_required |= needed

    coverage = (len(available_features & all_required) / len(all_required)) if all_required else 1.0

    for dashboard, needed in requirements.items():
        missing = sorted(list(needed - available_features))
        availability[dashboard] = {
            "available": len(missing) == 0,
            "coverage": 1 - len(missing) / len(needed) if needed else 1.0,
            "missing": missing,
        }
    return availability, coverage


def get_metric_availability(schema: Dict, domain: Optional[str] = None) -> Dict[str, Dict]:
    """Return per-dashboard availability with missing fields."""
    columns = schema.get("columns", [])
    mapped = auto_feature_mapping(columns)
    available_features = set(mapped.keys())
    reqs = ASTRO_REQUIREMENTS if domain == "astronomy" else FIN_REQUIREMENTS
    availability, _ = _availability(available_features, reqs)
    return availability


def get_dataset_context(domain: Optional[str] = None, file_path: Optional[str] = None) -> Dict:
    """
    Build a unified dataset context used by the frontend to decide what to render.
    Think of this as a "what can we show?" summary: schema, feature mappings,
    capability mode (full/partial/explanation), and missing columns per dashboard.
    """
    schema = get_dataset_schema(file_path, domain)
    detected_domain = domain or infer_domain(schema.get("columns", []))
    feature_map = auto_feature_mapping(schema.get("columns", []))

    requirements = ASTRO_REQUIREMENTS if detected_domain == "astronomy" else FIN_REQUIREMENTS
    availability, coverage = _availability(set(feature_map.keys()), requirements)

    if coverage >= 0.7:
        capability_mode = "full"
    elif coverage >= 0.4:
        capability_mode = "partial"
    else:
        capability_mode = "explanation"

    return {
        "schema": schema,
        "domain": detected_domain,
        "featureMapping": feature_map,
        "metricAvailability": availability,
        "capabilityMode": capability_mode,
        "missingByDashboard": {k: v.get("missing", []) for k, v in availability.items()},
        "coverage": coverage,
    }


def _scan_files_for_domain(domain: str) -> List[Path]:
    return _candidate_paths(domain)


def summarize_available_files(domain: str) -> List[Dict]:
    files = _scan_files_for_domain(domain)
    summaries = []
    for path in files:
        df = _load_preview(path)
        if df.empty:
            continue
        summaries.append({
            "path": str(path),
            "name": path.name,
            "columns": df.columns.tolist(),
            "rows": len(df)
        })
    return summaries


def pick_best_file(domain: str, dashboard: Optional[str] = None) -> Optional[Path]:
    """Pick the CSV whose columns best match the dashboard requirements."""
    reqs = ASTRO_REQUIREMENTS if domain == "astronomy" else FIN_REQUIREMENTS
    needed = reqs.get(dashboard or "", set())
    candidates = _scan_files_for_domain(domain)
    best_path = None
    best_score = -1
    for path in candidates:
        df = _load_preview(path)
        if df.empty:
            continue
        mapped = auto_feature_mapping(df.columns.tolist())
        score = len(set(mapped.keys()) & needed)
        coverage = score / len(needed) if needed else 1.0
        # prefer higher coverage, then more rows
        if coverage > best_score or (coverage == best_score and len(df) > 0):
            best_score = coverage
            best_path = path
        if coverage >= 1.0:
            break
    if best_path:
        return best_path
    return candidates[0] if candidates else None


def get_dashboard_dataset(domain: str, dashboard: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Return best-fit dataframe for a dashboard based on available files.
    Learning note: this is the decision point that stops blank charts—if the best
    file lacks required columns, capability mode drops to partial/explanation."""
    path = pick_best_file(domain, dashboard)
    if not path:
        return None
    try:
        return _load_preview(path)
    except Exception:
        return None
