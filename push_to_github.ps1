$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/dailhse184118-commits/DTA301_IS1904.git"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or is not available in PATH."
}

if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

$origin = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add origin $RepoUrl
} elseif ($origin -ne $RepoUrl) {
    git remote set-url origin $RepoUrl
}

$rawFiles = git status --short | Select-String -Pattern "\.zip$|data/raw/"
if ($rawFiles) {
    Write-Host "Review these possible raw-data files before committing:"
    $rawFiles
    throw "Raw-data safety check failed."
}

git add .
git status
git commit -m "Add full SinD vehicle behavior analysis pipeline"
git push -u origin main
