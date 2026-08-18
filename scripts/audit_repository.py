from __future__ import annotations
"""Audit the public CRIT-AID GitHub repository and optional large-results asset."""
import argparse, csv, gzip, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE_REQUIRED=[
 'README.md','LICENSE','CITATION.cff','codemeta.json','requirements.txt','environment.yml',
 'manifests/analysis_matrix.csv','manifests/seed_manifest.csv','manifests/raw_file_map.json',
 'manifests/acs_relationship_harmonization.csv','manifests/oulad_feature_dictionary.csv','manifests/heart_provenance.csv',
 'docs/result_registry.json','docs/diagnostic_result_registry.json','docs/RESULTS_GUIDE.md',
 'scripts/prepare_data_from_raw.py','scripts/run_acs_analysis.py','scripts/run_oulad_analysis.py',
 'scripts/run_external_domain_analyses.py','scripts/build_result_registry.py',
 'scripts/run_acs_target_definition_decomposition.py','scripts/run_conformal_tradeoff_analysis.py',
 'scripts/run_model_family_sensitivity.py','scripts/run_cross_domain_variance_analysis.py',
 'scripts/build_diagnostic_result_registry.py']
LARGE_RESULT_REQUIRED=['outputs/acs_risk_selection_curves.csv.gz','outputs/oulad_risk_selection_curves.csv.gz','outputs/acs_primary_target_predictions.csv.gz']
RESULT_EXPECTED_ROWS={'outputs/acs_metrics_complete.csv.gz':720,'outputs/oulad_metrics_complete.csv.gz':460,'outputs/south_metrics_complete.csv.gz':540,'outputs/heart_metrics_complete.csv.gz':480}
FORBIDDEN_DIRS={'manuscript','review_material'}
FORBIDDEN_NAME_TOKENS=('generate_manuscript','response_to_','revision_matrix')

def gzip_csv_rows(path):
 with gzip.open(path,'rt',encoding='utf-8',newline='') as s:
  r=csv.reader(s); next(r,None); return sum(1 for _ in r)
def audit_python(fail):
 scripts=sorted((ROOT/'scripts').glob('*.py'))
 for s in scripts:
  try:
   compile(s.read_text(encoding='utf-8'), str(s), 'exec')
  except SyntaxError as e:
   fail.append(f'Python syntax error in {s.relative_to(ROOT)}: {e}')
 print(f'OK: syntax-checked {len(scripts)} Python scripts')
def audit_policy(fail):
 for d in FORBIDDEN_DIRS:
  if (ROOT/d).exists(): fail.append(f'Non-analysis directory should not be in public repository: {d}/')
 for p in ROOT.rglob('*'):
  if not p.is_file() or '.git' in p.parts: continue
  low=p.name.lower()
  if any(t in low for t in FORBIDDEN_NAME_TOKENS): fail.append(f'Non-analysis publication artifact in repository: {p.relative_to(ROOT)}')
  if p.stat().st_size>50*1024*1024: fail.append(f'File exceeds 50 MiB Git-history threshold: {p.relative_to(ROOT)}')
  if p.suffix.lower() in {'.zip','.7z','.rar'}: fail.append(f'Archive should not be committed to source history: {p.relative_to(ROOT)}')
 data=ROOT/'data_raw'
 if data.exists():
  for p in data.rglob('*'):
   if p.is_file() and p.name!='README.md': fail.append(f'Raw data candidate is tracked: {p.relative_to(ROOT)}')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('source','results'),default='source'); a=ap.parse_args(); fail=[]
 for rel in SOURCE_REQUIRED:
  if not (ROOT/rel).exists(): fail.append(f'Missing required source file: {rel}')
 audit_python(fail); audit_policy(fail)
 matrix=ROOT/'manifests/analysis_matrix.csv'
 if matrix.exists():
  rows=list(csv.DictReader(matrix.open(newline='',encoding='utf-8'))); print(f'OK: analysis matrix rows={len(rows)}')
 if a.mode=='results':
  for rel in LARGE_RESULT_REQUIRED:
   if not (ROOT/rel).exists(): fail.append(f'Missing GitHub Release result asset file: {rel}')
  for rel,n in RESULT_EXPECTED_ROWS.items():
   p=ROOT/rel
   if not p.exists(): fail.append(f'Missing row-audited result: {rel}')
   else:
    obs=gzip_csv_rows(p); print(f'OK: {rel} rows={obs}')
    if obs!=n: fail.append(f'{rel}: expected {n} rows, observed {obs}')
 if fail:
  print('\nAUDIT FAILED'); [print('- '+x) for x in fail]; return 1
 print(f'\nAUDIT PASSED ({a.mode} mode)'); return 0
if __name__=='__main__': sys.exit(main()) 
