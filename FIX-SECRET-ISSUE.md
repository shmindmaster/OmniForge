# Fixing Secret Scanning Issue in OmniForge

## Problem
GitHub push protection is blocking pushes because `.env.local.backup` contains an Azure Storage Account Access Key in commit history.

## Solution Options

### Option 1: Use GitHub UI to Allow Secret (Quickest)
1. Visit: https://github.com/shmindmaster/OmniForge/security/secret-scanning/unblock-secret/35d8AbOr1EaUnFSsCk059SPj3ql
2. Click "Allow secret" (if you have permissions)
3. Push again: `git push origin main`

### Option 2: Remove Secret from Git History (Recommended)

**Step 1: Remove file from working directory**
```powershell
cd H:\Repos\shmindmaster\OmniForge
Remove-Item ".env.local.backup" -Force -ErrorAction SilentlyContinue
```

**Step 2: Add to .gitignore**
```powershell
# Add these lines to .gitignore:
.env.local.backup
*.env.local.backup
.env.backup
*.env.backup
```

**Step 3: Remove from git history**
```powershell
# Remove file from all commits
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch .env.local.backup" --prune-empty --tag-name-filter cat -- --all

# Clean up
git for-each-ref --format="delete %(refname)" refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**Step 4: Force push**
```powershell
git push origin main --force
```

### Option 3: Create Clean Orphan Branch (Nuclear Option)

If filter-branch doesn't work:

```powershell
# Create new orphan branch
git checkout --orphan clean-main

# Add all files except .env.local.backup
git add .
git commit -m "Initial commit - clean history"

# Delete old main and rename
git branch -D main
git branch -m main

# Force push
git push origin main --force
```

## Prevention

1. ✅ Added `.env.local.backup` to `.gitignore`
2. ✅ Never commit backup files
3. ✅ Use `.env.example` for templates
4. ✅ Rotate the exposed secret in Azure

## After Fixing

1. Rotate the Azure Storage Account key that was exposed
2. Update all references to use the new key
3. Verify `.gitignore` includes all backup patterns

