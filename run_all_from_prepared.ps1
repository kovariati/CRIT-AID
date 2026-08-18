[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$env:CRIT_AID_ROOT = $root
$env:PYTHONPATH = (Join-Path $root 'scripts')

$scripts = @(
    'run_acs_analysis.py',
    'run_acs_postprocessing.py',
    'run_oulad_analysis.py',
    'run_oulad_bootstrap.py',
    'run_external_domain_analyses.py',
    'run_model_family_sensitivity.py'
)

foreach ($script in $scripts) {
    & python (Join-Path $root ('scripts\{0}' -f $script))
    if ($LASTEXITCODE -ne 0) {
        throw ('{0} failed with exit code {1}' -f $script, $LASTEXITCODE)
    }
}

& (Join-Path $root 'run_postprocessing.ps1')
