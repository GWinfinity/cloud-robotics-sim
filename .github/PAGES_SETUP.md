# GitHub Pages Setup Guide

## Enable GitHub Pages

GitHub Pages is currently **disabled**. You need to enable it manually:

### Step 1: Go to Settings

1. Open your repository: https://github.com/GWinfinity/cloud-robotics-sim
2. Click the **Settings** tab

### Step 2: Navigate to Pages

1. In the left sidebar, scroll down and click **Pages**

### Step 3: Configure Source

Under **Build and deployment**:

1. **Source**: Select **Deploy from a branch**
2. **Branch**: Select **gh-pages** / **/(root)**
3. Click **Save**

![GitHub Pages Settings](https://docs.github.com/assets/images/help/pages/pages-choose-publish-source-drop-down.png)

### Step 4: Wait for Deployment

After enabling:
1. The workflow will automatically create a `gh-pages` branch
2. GitHub will deploy the site
3. You'll see a green checkmark when ready
4. Visit: `https://gwinfinity.github.io/cloud-robotics-sim/`

---

## Alternative: Use GitHub Actions (New Way)

If you prefer the new GitHub Actions deployment:

1. Go to **Settings** → **Pages**
2. **Source**: Select **GitHub Actions**
3. Click **Save**

Then use the `docs-gh-actions.yml` workflow (not included by default).

---

## Troubleshooting

### "404 Not Found" Error

The Pages site hasn't been deployed yet. Wait 1-2 minutes after the workflow completes.

### "Failed to create deployment" Error

This means GitHub Pages is not enabled. Follow Step 1-3 above.

### Workflow Failed

Check the workflow logs:
1. Go to **Actions** tab
2. Click on the failed workflow
3. Check the error message

---

## Current Workflow

The repository uses `peaceiris/actions-gh-pages` which:
1. Builds Sphinx documentation
2. Creates a `gh-pages` branch
3. Pushes built HTML to that branch
4. GitHub Pages serves from `gh-pages` branch

You need to enable Pages **once**, then it will work automatically.
