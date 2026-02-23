$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string[]]$CommandParts)
    if (-not $CommandParts -or $CommandParts.Count -eq 0) {
        throw "Invoke-Step requires at least one command part"
    }

    Write-Host "[cli-smoke] $($CommandParts -join ' ')"
    $command = $CommandParts[0]
    $commandArgs = @()
    if ($CommandParts.Count -gt 1) {
        $commandArgs = $CommandParts[1..($CommandParts.Count - 1)]
    }

    $log = New-TemporaryFile
    & $command @commandArgs *> $log
    if ($LASTEXITCODE -ne 0) {
        Get-Content $log
        Remove-Item $log -ErrorAction SilentlyContinue
        throw "Command failed: $($CommandParts -join ' ')"
    }
    Remove-Item $log -ErrorAction SilentlyContinue
}

Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--help")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "examples/mnist.vt")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--bench-infer", "--runs", "1", "--warmup", "0", "--tokens", "4")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--tune-matmul", "--dim", "64", "--runs", "1")

Write-Host "[cli-smoke] all checks passed"
