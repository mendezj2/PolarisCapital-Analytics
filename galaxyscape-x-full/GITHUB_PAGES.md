# Deploying to GitHub Pages

1. **Build the static site**

   ```
   python3 scripts/build_github_pages.py --api-base https://YOUR-BACKEND-URL
   ```

   The command copies `frontend/static/` into `docs/`, drops a `.nojekyll` marker,
   and (optionally) bakes the backend base URL into `docs/js/github_pages_config.js`.

2. **Commit & push**

   ```
   git add docs
   git commit -m "Publish GitHub Pages bundle"
   git push origin main
   ```

3. **Enable Pages**

   In GitHub → *Settings → Pages*, set the source to `Deploy from a branch`,
   branch `main`, directory `/docs`. Within a minute the dashboard will be
   available at `https://<username>.github.io/<repo>/`.

4. **Update API base when backend changes**

   Re-run the build command with the new backend URL and push the refreshed `docs/`
   folder to keep the hosted dashboard in sync.

