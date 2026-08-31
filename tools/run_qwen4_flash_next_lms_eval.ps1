[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Q4', 'Q8')]
    [string]$Quant,

    [int]$ContextLength = 32768,
    [int]$MaxTokens = 512,
    [string]$ResultsDirectory = '..\.model-research\endpoint-eval'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$settings = @{
    Q4 = @{
        ModelKey = 'unsloth/qwen3.8-flash-next'
        Identifier = 'next48-q4'
    }
    Q8 = @{
        ModelKey = 'qwen3.8-flash-next@q8_0'
        Identifier = 'next48-q8'
    }
}
$selected = $settings[$Quant]
$resultRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ResultsDirectory))
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$output = Join-Path $resultRoot ("lmstudio-{0}-{1}.jsonl" -f $Quant.ToLowerInvariant(), $stamp)
$loadedHere = $false

Push-Location $repoRoot
try {
    $processes = @(& lms ps --json | ConvertFrom-Json)
    $collision = $processes | Where-Object { $_.identifier -eq $selected.Identifier }
    if ($collision) {
        throw "LM Studio identifier '$($selected.Identifier)' is already loaded; refusing to take ownership of it."
    }

    Write-Host "Loading only $($selected.ModelKey) as $($selected.Identifier)."
    & lms load $selected.ModelKey `
        --context-length $ContextLength `
        --parallel 1 `
        --identifier $selected.Identifier `
        --yes
    if ($LASTEXITCODE -ne 0) {
        throw "lms load failed with exit code $LASTEXITCODE"
    }
    $loadedHere = $true

    New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
    & uv --cache-dir ..\.uv-cache run --no-sync python `
        tools\qwen4_flash_next_eval.py run `
        --endpoint ("lmstudio-{0}" -f $Quant.ToLowerInvariant()) `
        --base-url http://127.0.0.1:1234 `
        --model $selected.Identifier `
        --env-file ..\.env `
        --max-tokens $MaxTokens `
        --output $output
    if ($LASTEXITCODE -ne 0) {
        throw "endpoint evaluation failed with exit code $LASTEXITCODE"
    }
    Write-Host "Result: $output"
}
finally {
    if ($loadedHere) {
        Write-Host "Unloading only $($selected.Identifier); linked models are untouched."
        & lms unload $selected.Identifier
    }
    Pop-Location
}
