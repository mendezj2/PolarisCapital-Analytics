# PolarisCapital Analytics - GitHub Pages Build

This directory is auto-generated from `frontend/static/` using
`python scripts/build_github_pages.py`.

## Deploy Steps
1. Run `python scripts/build_github_pages.py [--api-base https://your-backend]`.
2. Commit the updated `docs/` directory.
3. Push to `main` and enable GitHub Pages (Settings → Pages → Build from `docs/`).

When hosting on GitHub Pages, set `--api-base` to the publicly reachable Flask
backend so the dashboards can load live data.
