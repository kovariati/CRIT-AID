from pathlib import Path
import json, os
import pandas as pd

ROOT=Path(os.environ.get('CRIT_AID_ROOT',Path(__file__).resolve().parents[1])); OUT=ROOT/'outputs'; DOCS=ROOT/'docs'
TITLE='Stable Discrimination Can Hide Reliability Failures in AI Decision Support under Distribution Shift and Changing Target Definitions'

def records(path):
    return pd.read_csv(path).to_dict(orient='records')

def main():
    reg={
      'canonical_title':TITLE,
      'analysis_scope':'posthoc_diagnostics',
      'acs_target_label_change':records(OUT/'acs_target_label_change_summary.csv'),
      'acs_two_factor_metric_decomposition':records(OUT/'acs_two_factor_metric_decomposition.csv'),
      'acs_prevalence_alignment':records(OUT/'acs_prevalence_alignment_summary.csv'),
      'conformal_tradeoff_overall':records(OUT/'conformal_tradeoff_overall_summary.csv'),
      'conformal_tradeoff_by_condition':records(OUT/'conformal_tradeoff_condition_summary.csv'),
      'cross_domain_variance_components':records(OUT/'cross_domain_variance_components.csv'),
      'model_family_delta_effects':records(OUT/'model_family_delta_effects.csv'),
      'model_family_sensitivity_summary':records(OUT/'model_family_sensitivity_summary.csv'),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS/'diagnostic_result_registry.json').write_text(json.dumps(reg,indent=2,ensure_ascii=False)+'\n')
    base_path=DOCS/'result_registry.json'
    if base_path.exists():
        base=json.loads(base_path.read_text())
        base['posthoc_diagnostics']=reg
        base_path.write_text(json.dumps(base,indent=2,ensure_ascii=False)+'\n')
    print('Wrote',DOCS/'diagnostic_result_registry.json')
if __name__=='__main__': main()
