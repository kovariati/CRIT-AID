[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$env:CRIT_AID_ROOT = $root
$env:PYTHONPATH = (Join-Path $root 'scripts')

$scripts = @(
    'build_result_registry.py',
    'build_tables_figures.py',
    'build_acs_tables_figures.py',
    'make_protocol_figure.py',
    'run_acs_target_definition_decomposition.py',
    'run_conformal_tradeoff_analysis.py',
    'run_cross_domain_variance_analysis.py',
    'build_diagnostic_result_registry.py'
)

foreach ($script in $scripts) {
    & python (Join-Path $root ('scripts\{0}' -f $script))
    if ($LASTEXITCODE -ne 0) {
        throw ('{0} failed with exit code {1}' -f $script, $LASTEXITCODE)
    }
}

& python (Join-Path $root 'scripts\audit_repository.py') --mode results --skip-checksums
if ($LASTEXITCODE -ne 0) {
    throw ('Result audit failed with exit code {0}' -f $LASTEXITCODE)
}
