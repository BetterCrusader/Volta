[CmdletBinding()]
param(
    [string]$Tag = "release-v1.0.0",
    [string]$Remote = "origin"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )
    Write-Host "[cut-v1] $Name"
    & $Action
}

function Assert-CleanWorkingTree {
    git diff --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "[cut-v1] working tree is not clean (unstaged changes)"
    }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "[cut-v1] working tree is not clean (staged changes)"
    }
}

function Invoke-InteropOnnxVerify {
    Invoke-Step "interop onnx verify: contract test" {
        cargo test --features onnx-import interop::contract::tests::contract_compile_rejects_unknown_input_id
    }
    Invoke-Step "interop onnx verify: acceptance tests" {
        cargo test --features onnx-import --test interop_onnx_linear
        cargo test --features onnx-import --test interop_onnx_mlp
        cargo test --features onnx-import --test interop_onnx_parser
        cargo test --features onnx-import --test interop_roundtrip_parity
        cargo test --features onnx-import --test interop_onnx_wave2_contract
        cargo test --features onnx-import --test interop_onnx_wave2_parser
        cargo test --features onnx-import --test interop_onnx_wave2_e2e
    }
}

function Invoke-XlVerify {
    Invoke-Step "xl verify: execution plan cache" { cargo test --test execution_plan_cache -- --nocapture }
    Invoke-Step "xl verify: static memory budget" { cargo test --test xl_static_memory_budget -- --nocapture }
    Invoke-Step "xl verify: gradient checkpointing" { cargo test --test xl_gradient_checkpointing -- --nocapture }
    Invoke-Step "xl verify: backend capability matrix" { cargo test --test backend_capability_matrix -- --nocapture }
    Invoke-Step "xl verify: schedule optimization" { cargo test --test schedule_optimization -- --nocapture }
    Invoke-Step "xl verify: model export" { cargo test --test model_export -- --nocapture }
    Invoke-Step "xl verify: runtime single-path regression" { cargo test --test runtime_single_truth_path -- --nocapture }
    Invoke-Step "xl verify: compile gate" { cargo test --no-run }
}

function Invoke-RollbackVerifyOnly {
    Invoke-Step "rollback verify-only" {
        git fetch --tags --force
        $tags = @(git tag --sort=-creatordate)
        if ($tags.Count -ge 2) {
            $target = $tags[1]
            Write-Host "rollback_target=$target"
            Write-Host "rollback verification succeeded"
        }
        else {
            Write-Host "rollback_target=none"
            Write-Host "rollback verification skipped (no release tags found)"
        }
    }
}

Assert-CleanWorkingTree

$currentBranch = (git branch --show-current).Trim()
if ($currentBranch -ne "main") {
    throw "[cut-v1] must run from main branch (current: $currentBranch)"
}

Invoke-Step "syncing with $Remote/main" {
    git fetch $Remote
    git merge --ff-only "$Remote/main"
}

Invoke-Step "running release checks" {
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test
    cargo test --release
}

Invoke-InteropOnnxVerify
Invoke-XlVerify
Invoke-RollbackVerifyOnly

git rev-parse --verify --quiet "refs/tags/$Tag" | Out-Null
if ($LASTEXITCODE -eq 0) {
    throw "[cut-v1] tag '$Tag' already exists"
}

Invoke-Step "creating tag: $Tag" {
    git tag -a $Tag -m "Volta V1 release"
    git push $Remote $Tag
}

Write-Host "[cut-v1] done"
