# GitHub Actions Setup Guide

## Fixing "git exit code 128" Error

This error means the workflow doesn't have permission to push changes. Here's how to fix it:

### Step 1: Enable Workflow Permissions

1. Go to your repository on GitHub
2. Click **Settings** tab
3. In the left sidebar, click **Actions** → **General**
4. Scroll to **Workflow permissions**
5. Select: **"Read and write permissions"**
6. ✅ Check: **"Allow GitHub Actions to create and approve pull requests"**
7. Click **Save**

![Workflow Permissions](https://docs.github.com/assets/cb-60049/images/help/settings/actions-workflow-permissions-repository.png)

### Step 2: Disable Branch Protection (for auto-fix)

If you have branch protection rules on `main`, you need to allow GitHub Actions to bypass them:

1. Go to **Settings** → **Branches**
2. Find your branch protection rule for `main`
3. Click **Edit**
4. Scroll to **Restrict who can push to matching branches**
5. Uncheck it OR add `github-actions[bot]` to the allowed users
6. Alternatively, check **"Allow force pushes"** → **"Specify who can force push"** → add `github-actions[bot]`

### Step 3: Alternative - Use Personal Access Token

If the above doesn't work, create a PAT:

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Select scopes:
   - ✅ `repo` (Full control of private repositories)
   - ✅ `workflow` (Update GitHub Action workflows)
4. Generate and copy the token
5. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
6. Click **New repository secret**
7. Name: `PAT`
8. Value: Paste your token
9. Update the workflow files to use `${{ secrets.PAT }}` instead of `${{ secrets.GITHUB_TOKEN }}`

## Available Auto-Fix Options

### Option 1: Direct Commit (`auto-fix.yml`)

**Pros:**
- Simple and reliable
- No PR review needed
- Fast turnaround

**Cons:**
- Changes go directly to main
- Less visibility

### Option 2: Pull Request (`auto-fix-pr.yml`)

**Pros:**
- Code review before merging
- Clear audit trail
- Safer for production

**Cons:**
- Requires PR creation permissions
- Manual merge step

## Testing Auto-Fix

### Manual Trigger

You can manually run auto-fix:

1. Go to **Actions** tab
2. Select **Auto Fix** or **Auto Fix with PR**
3. Click **Run workflow**

### Check Logs

If auto-fix fails:

1. Go to **Actions** tab
2. Click on the failed workflow run
3. Expand the failed step
4. Look for error messages

Common errors:
- `exit code 128`: Permission issue (see Step 1 above)
- `exit code 1`: No changes to commit
- `Resource not accessible`: Token permissions

## Disabling Auto-Fix

To disable:

1. Go to **Actions** tab
2. Click on **Auto Fix** workflow
3. Click **...** menu → **Disable workflow**

Or delete the workflow files from `.github/workflows/`.

## Troubleshooting Checklist

- [ ] Workflow permissions set to "Read and write"
- [ ] "Allow GitHub Actions to create PRs" is checked
- [ ] Branch protection allows GitHub Actions to push
- [ ] Workflow file has correct `permissions:` block
- [ ] Token has necessary scopes
