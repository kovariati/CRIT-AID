[CmdletBinding()]
param(
    [string]$DataRoot = (Join-Path $PSScriptRoot 'data_raw')
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw ('Raw-data directory does not exist: {0}' -f $DataRoot)
}

$env:CRIT_AID_ROOT = $root
$env:CRIT_AID_DATA_ROOT = (Resolve-Path -LiteralPath $DataRoot).Path
$env:PYTHONPATH = (Join-Path $root 'scripts')

& python (Join-Path $root 'scripts\prepare_data_from_raw.py')
if ($LASTEXITCODE -ne 0) {
    throw ('prepare_data_from_raw.py failed with exit code {0}' -f $LASTEXITCODE)
}

& (Join-Path $root 'run_all_from_prepared.ps1')
