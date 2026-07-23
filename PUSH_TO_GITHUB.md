# Push this project to GitHub

Target repository:

```text
https://github.com/dailhse184118-commits/DTA301_IS1904.git
```

## Recommended method for an existing repository

```powershell
git clone https://github.com/dailhse184118-commits/DTA301_IS1904.git
cd DTA301_IS1904
```

Copy the contents of this GitHub-ready package into the cloned folder, then run:

```powershell
git add .
git status
git commit -m "Add full SinD vehicle behavior analysis pipeline"
git push origin main
```

Before committing, confirm that `git status` does not show any raw SinD ZIP
archives or files under `data/raw/`.

## New empty repository method

Open PowerShell in this folder:

```powershell
git init
git branch -M main
git remote add origin https://github.com/dailhse184118-commits/DTA301_IS1904.git
git add .
git commit -m "Add full SinD vehicle behavior analysis pipeline"
git push -u origin main
```

Do not use `git push --force` unless you intentionally want to replace remote
history.
