$ErrorActionPreference = "Stop"

param(
    [switch]$VerifyOnly,
    [string]$Target = ""
)

git fetch --tags --force | Out-Null

if ([string]::IsNullOrWhiteSpace($Target)) {
    $tags = git tag --sort=-creatordate
    $tagList = $tags -split "`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    if ($tagList.Length -lt 2) {
        if ($VerifyOnly) {
            Write-Host "rollback_target=none"
            Write-Host "rollback verification skipped (no release tags found)"
            exit 0
        }
        throw "unable to determine rollback target tag"
    }
    $Target = $tagList[1].Trim()
}

$targetRef = "refs/tags/$Target"
git rev-parse --verify --quiet $targetRef | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "rollback target tag '$Target' does not exist"
}

Write-Host "rollback_target=$Target"

if ($VerifyOnly) {
    Write-Host "rollback verification succeeded"
    exit 0
}

Write-Host "checking out rollback target '$Target'"
git checkout $Target | Out-Null
Write-Host "rollback completed"
