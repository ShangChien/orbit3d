"""Empirical interval diagnostics on held-out terminal labels, with no Gaussian assumption."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from orbit3d.oracles.calibration import ResidualCalibration


def interval_metrics(calibration, truth, mean, std):
    lo,hi=calibration.intervals(mean,std,simultaneous=False)
    joint_lo,joint_hi=calibration.intervals(mean,std,simultaneous=True)
    return {'n':len(truth),'marginal_coverage':((truth>=lo)&(truth<=hi)).mean(0).tolist(),
            'marginal_mean_width_eV':(hi-lo).mean(0).tolist(),
            'simultaneous_coverage':float(((truth>=joint_lo)&(truth<=joint_hi)).all(1).mean()),
            'simultaneous_mean_width_eV':(joint_hi-joint_lo).mean(0).tolist()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths={key:a.fit/f'{key}_predictions.npz' for key in ['valid_generated','test_generated','source_test']}
    arrays={key:dict(np.load(path)) for key,path in paths.items()};valid=arrays['valid_generated'];reports={}
    for name,field,scaled in [('baseline_constant','baseline',False),('affine_constant','affine',False),('adapted_constant','prediction',False),('adapted_scaled','prediction',True)]:
        std=valid['std'] if scaled else np.zeros_like(valid[field])
        calibration=ResidualCalibration.fit(['homo','lumo'],valid['truth'],valid[field],std,coverage=.9,std_floor=.05)
        reports[name]={'calibration':asdict(calibration),'evaluation':{}}
        for key in ['test_generated','source_test']:
            rows=arrays[key];std=rows['std'] if scaled else np.zeros_like(rows[field])
            reports[name]['evaluation'][key]=interval_metrics(calibration,rows['truth'],rows[field],std)
    report={'scope':'Descriptive 90% residual intervals. Fit on the 85 generated validation structures; this validation set also selected warm-start epochs, so these are not independent split-conformal guarantees. Generated test and native source test labels are never used to set interval widths. Compare constant residual width with bootstrap-scaled width (floor 0.05 eV), without treating bootstrap spread as a Gaussian predictive distribution. Source and generated labels retain their distinct basis protocols.','reports':reports,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
