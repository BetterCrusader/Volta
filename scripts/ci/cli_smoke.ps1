$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string[]]$Args)
    Write-Host "[cli-smoke] $($Args -join ' ')"
    $log = New-TemporaryFile
    & $Args[0] $Args[1..($Args.Length - 1)] *> $log
    if ($LASTEXITCODE -ne 0) {
        Get-Content $log
        Remove-Item $log -ErrorAction SilentlyContinue
        throw "Command failed: $($Args -join ' ')"
    }
    Remove-Item $log -ErrorAction SilentlyContinue
}

Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--help")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "examples/mnist.vt")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--bench-infer", "--runs", "1", "--warmup", "0", "--tokens", "4")
Invoke-Step @("cargo", "run", "--bin", "volta", "--", "--tune-matmul", "--dim", "64", "--runs", "1")

Write-Host "[cli-smoke] all checks passed"
