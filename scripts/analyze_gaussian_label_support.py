"""Empirical label distribution implied by Gaussian request augmentation, not generation quality."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from rdkit import Chem


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=a.data/'train.sdf';mols=list(Chem.SDMolSupplier(str(source),removeHs=False))
    assert mols and all(m is not None for m in mols)
    manifest=json.loads((a.data/'manifest.json').read_text());rows=[]
    for target in ['homo','lumo']:
        y=np.array([float(m.GetProp(target)) for m in mols])
        for q,mu in manifest['single_target_points'][target].items():
            for sigma in [.1,.25,.5]:
                logw=-.5*((y-mu)/sigma)**2;w=np.exp(logw-logw.max());w/=w.sum();mean=float(w@y)
                rows.append({'target':target,'quantile':q,'request_mean_eV':mu,'preference_sigma_eV':sigma,
                    'implied_label_mean_eV':mean,'mean_minus_request_eV':mean-mu,
                    'implied_label_std_eV':float(np.sqrt(w@((y-mean)**2))),
                    'implied_label_target_mae_eV':float(w@np.abs(y-mu)),
                    'effective_source_molecules':float(1/(w@w)),
                    'source_mass_within_fixed_025':float(w@(np.abs(y-mu)<=.25))})
    report={'source_molecules':len(mols),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'definition':'Empirical source prior reweighted by exp(-(y-mu)^2/(2*sigma^2)); normalized within the source training set.',
        'scope':'Conditional label law induced by plain Gaussian request augmentation. Not a generator result, performance bound, calibrated predictive uncertainty or reference for hindsight/contrast objectives.',
        'rows':rows}
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
