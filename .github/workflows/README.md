# GitHub Actions Workflows

## Workflows Overview

| Workflow | File | Purpose | Trigger |
|----------|------|---------|---------|
| CI | `ci.yml` | Run tests on Python 3.10/3.11/3.12 | Push, PR |
| Documentation | `docs.yml` | Build and deploy docs | Push, PR |
| Auto Fix | `auto-fix.yml` | Auto-fix lint issues and create PR | CI failure |
| Auto Fix (Simple) | `auto-fix-simple.yml` | Direct commit fix (fallback) | CI failure |

## Setup Instructions

### Enable Workflow Permissions

For auto-fix workflows to work, you need to enable permissions:

1. Go to **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select:
   - ✅ **Read and write permissions**
   - ✅ **Allow GitHub Actions to create and approve pull requests**

![Workflow Permissions](https://docs.github.com/assets/images/help/settings/actions-workflow-permissions.png)

### Create PR Token (if needed)

If auto-fix still fails to create PRs, create a Personal Access Token:

1. Go to **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **Generate new token**
3. Select scopes:
   - ✅ `repo` (full control of private repositories)
   - ✅ `workflow` (update GitHub Action workflows)
4. Copy the token
5. Go to your repository → **Settings** → **Secrets and variables** → **Actions**
6. Click **New repository secret**
7. Name: `PAT_TOKEN`
8. Value: Paste your token
9. Update `auto-fix.yml` to use `${{ secrets.PAT_TOKEN }}` instead of `GITHUB_TOKEN`

## Auto Fix Behavior

### Option 1: Pull Request (Recommended)

The `auto-fix.yml` workflow:
1. Detects CI failure
2. Applies fixes (trailing whitespace, ruff, black)
3. Creates a new branch
4. Creates a Pull Request
5. You review and merge

**Benefits:**
- Code review before changes
- No direct commits to main
- Clear audit trail

### Option 2: Direct Commit

The `auto-fix-simple.yml` workflow:
1. Detects CI failure
2. Applies fixes
3. Commits directly to main
4. CI re-runs automatically

**Benefits:**
- Faster turnaround
- No manual intervention

**Risks:**
- Changes go directly to main
- Less visibility

## Manual Trigger

You can manually trigger auto-fix:

```bash
gh workflow run auto-fix.yml
```

Or via GitHub web interface:
**Actions** → **Auto Fix** → **Run workflow**

## Troubleshooting

### "Resource not accessible by integration" Error

This means the token doesn't have enough permissions. Solutions:

1. Enable "Read and write permissions" in Settings
2. Use a Personal Access Token instead of GITHUB_TOKEN
3. Use the `auto-fix-simple.yml` workflow instead

### PR Creation Fails

Check:
1. Workflow permissions are enabled
2. Token has `pull-requests: write` permission
3. Branch protection rules allow PR creation

### No Changes Detected

If auto-fix runs but no changes are made:
1. Check the lint errors are auto-fixable
2. Some errors (like complex type issues) need manual fixing
3. Check the workflow logs for details

## Disabling Auto Fix

To disable auto-fix:

1. Go to **Actions** → **Auto Fix**
2. Click **...** → **Disable workflow**

Or delete/rename the workflow files.
