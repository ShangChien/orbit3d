"""Replay a complete frozen pool from an isolated wheel payload on the native runtime."""
import argparse,csv,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import orbit3d


def main():
    p=argparse.ArgumentParser();p.add_argument('--wheel',type=Path,required=True);p.add_argument('--site',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    package=Path(orbit3d.__file__).resolve().parent;assert package==(a.site/'orbit3d').resolve(),'Wrong imported package'
    checked=0
    with zipfile.ZipFile(a.wheel) as wheel:
        for name in wheel.namelist():
            if name.startswith('orbit3d/') and Path(name).suffix in {'.py','.json','.txt'}:
                assert wheel.read(name)==(a.site/name).read_bytes(),name;checked+=1
    cfg=json.loads((a.reference/'run_config.json').read_text());assert cfg['mode']=='predict' and not cfg['generator_config']['grpo_steps']
    a.output.mkdir(parents=True,exist_ok=False)
    cfg['mount_paths'].update(output_dir=str(a.output/'run'),checkpoint_save_dir=str(a.output/'weights'))
    config=a.output/'config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    with (a.output/'pipeline.log').open('w') as log:
        subprocess.run([sys.executable,'-m','orbit3d.cli','--config',str(config)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with np.load(a.reference/'raw_trajectories.npz') as old,np.load(a.output/'run/raw_trajectories.npz') as new:
        assert set(old.files)==set(new.files)
        exact={key:np.array_equal(old[key],new[key]) for key in old.files}
    symbols=json.loads((a.reference/'raw_symbols.json').read_text());actual=json.loads((a.output/'run/raw_symbols.json').read_text())
    def retained(path):
        with path.open() as stream:return {int(r['candidate_id']):r for r in csv.DictReader(stream) if r['status']=='refined'}
    old_rows=retained(a.reference/'candidates.csv');new_rows=retained(a.output/'run/candidates.csv');common=sorted(old_rows.keys()&new_rows.keys())
    prop_diff=max((float(np.abs(np.array(json.loads(old_rows[i]['y_pred']))-json.loads(new_rows[i]['y_pred'])).max()) for i in common),default=None)
    force_diff=max((abs(float(old_rows[i]['final_force'])-float(new_rows[i]['final_force'])) for i in common),default=None)
    old_summary=json.loads((a.reference/'summary.json').read_text());new_summary=json.loads((a.output/'run/summary.json').read_text())
    report={'scope':'Same full raw pool; package source verified against wheel bytes. Admission/export differences are recorded separately from exact generation replay.',
        'wheel_sha256':hashlib.sha256(a.wheel.read_bytes()).hexdigest(),'package_path':str(package),'package_files_verified':checked,
        'reference':str(a.reference),'attempted':len(symbols),'symbols_exact':symbols==actual,
        'tokens_exact':sum(v for k,v in exact.items() if k.startswith('tokens_')),
        'coordinates_exact':sum(v for k,v in exact.items() if k.startswith('coordinates_')),
        'common_retained':len(common),'common_property_max_abs_diff_eV':prop_diff,'common_force_max_abs_diff_eVA':force_diff,
        'summaries':{label:{key:value[key] for key in ['attempted','raw_valid','domain_eligible','retained','qualified','export_prediction_recheck_passed']}
                     for label,value in [('reference',old_summary),('wheel',new_summary)]}}
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    assert all(exact.values()) and symbols==actual,'Raw generation changed'
    assert common and prop_diff<=1e-5 and force_diff<=1e-5,'Common native predictions exceed declared replay tolerance'


if __name__=='__main__':main()
