"""
GitHub Pages Build Helper
=========================

Copies the production-ready static frontend into the `docs/` directory so it can
be published via GitHub Pages (`Settings > Pages > Build from /docs`).

Usage:
    python scripts/build_github_pages.py [--api-base https://YOUR_BACKEND]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_SRC = PROJECT_ROOT / "frontend" / "static"
DOCS_DIR = PROJECT_ROOT / "docs"


def copy_static_assets() -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    shutil.copytree(STATIC_SRC, DOCS_DIR)
    (DOCS_DIR / ".nojekyll").write_text("")


def write_api_base(api_base: str | None) -> None:
    config_path = DOCS_DIR / "js" / "github_pages_config.js"
    if not config_path.exists():
        return
    if api_base:
        api_base = api_base.rstrip("/")
        config_path.write_text(
            "window.GALAXYSCAPE_API_BASE = '{}';\n".format(api_base),
            encoding="utf-8",
        )


def write_docs_readme(api_base: str | None) -> None:
    readme_path = DOCS_DIR / "README.md"
    doc = [
        "# PolarisCapital Analytics - GitHub Pages Build",
        "",
        "This directory is auto-generated from `frontend/static/` using",
        "`python scripts/build_github_pages.py`.",
        "",
        "## Deploy Steps",
        "1. Run `python scripts/build_github_pages.py [--api-base https://your-backend]`.",
        "2. Commit the updated `docs/` directory.",
        "3. Push to `main` and enable GitHub Pages (Settings → Pages → Build from `docs/`).",
        "",
        "When hosting on GitHub Pages, set `--api-base` to the publicly reachable Flask",
        "backend so the dashboards can load live data.",
    ]
    if api_base:
        doc += [
            "",
            f"Current API base baked into this build: `{api_base}`."
        ]
    readme_path.write_text("\n".join(doc) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare docs/ for GitHub Pages.")
    parser.add_argument(
        "--api-base",
        help="Public base URL of the backend (e.g. https://polaris-api.example.com)",
    )
    args = parser.parse_args()

    copy_static_assets()
    write_api_base(args.api_base)
    write_docs_readme(args.api_base)
    print(f"GitHub Pages bundle ready under: {DOCS_DIR}")


if __name__ == "__main__":
    main()

