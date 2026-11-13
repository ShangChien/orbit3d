"""Freeze response inverses using old DPA scans only, before new-seed evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

TARGETS={'homo':[-6.8,-6.4,-6.1],'lumo':[-2.15,-1.95,-1.75]}

def fit_response(x,y,noise):
    center=float(np.mean(x));scale=float(np.ptp(x));yc=float(np.mean(y));ys=float(np.std(y))
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,(.01,100.))*Matern(1.,(.15,3.),nu=2.5),
        alpha=np.maximum(noise,1e-6)/ys**2,normalize_y=False,n_restarts_optimizer=5,random_state=42)
    gp.fit(((x-center)/scale)[:,None],(y-yc)/ys)
    grid=np.linspace(x.min(),x.max(),2001);mu,std=gp.predict(((grid-center)/scale)[:,None],return_std=True)
    slope,intercept=np.polyfit(x,y,1)
    return {'linear_slope':float(slope),'linear_intercept':float(intercept),'gp_kernel':str(gp.kernel_),
        'grid':grid.tolist(),'gp_mean':(mu*ys+yc).tolist(),'gp_latent_std':(std*ys).tolist(),
        'bounds':[float(x.min()),float(x.max())]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);old=json.loads(a.reference.read_text())
    report={'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        'fit_scope':'Five old input conditions, in-domain DPA predictions, seeds 2026/2027/2028 only. FP is not used for fitting.',
        'validation_seeds':[2031,2032,2033],'candidates_per_pool':64,
        'target_scope':'New absolute target values around the previously observed response range, not new training-label quantiles; excludes a claim of full-range controllability.',
        'inversion':'Linear inverse clipped to old P10-P90 input support. GP minimizes squared posterior-mean mismatch over 2001 equally spaced supported inputs; no extrapolation.',
        'gp_noise':'Sample variance of three seed-specific conditional means divided by 3; GP posterior std concerns the mean response, not molecular spread.',
        'selection':'Same candidates, top4 vs Gaussian sampling without replacement sigma=.25 or .5 eV, plus uniform control; 100 deterministic selection repetitions. No training or coordinate optimization.',
        'properties':{}}
    for prop,index in [('homo',0),('lumo',1)]:
        runs=old['properties'][prop]['runs'];x=[];y=[];noise=[];observations=[]
        for q in ['10','25','50','75','90']:
            cells=[r for r in runs if r['quantile']==q];seedmeans=[]
            for r in cells:
                vals=[v['prediction'][index] for v in r['raw_records'] if 'prediction' in v and v['domain']['in_domain']]
                seedmeans.append(float(np.mean(vals)))
            x.append(cells[0]['scoring_value']);y.append(float(np.mean(seedmeans)));noise.append(float(np.var(seedmeans,ddof=1)/3))
            observations.append({'input':x[-1],'seed_means':seedmeans,'mean':y[-1],'mean_noise_variance':noise[-1]})
        fit=fit_response(np.array(x),np.array(y),np.array(noise));experiments=[]
        for ti,target in enumerate(TARGETS[prop]):
            raw_inverse=(target-fit['linear_intercept'])/fit['linear_slope'];linear=float(np.clip(raw_inverse,*fit['bounds']))
            idx=int(np.argmin(np.square(np.array(fit['gp_mean'])-target)));gp=fit['grid'][idx]
            for arm,u in [('original',target),('linear',linear),('gp',gp)]:
                experiments.append({'key':f't{ti}_{arm}','target_value':target,'input_value':u,'arm':arm,
                    'at_boundary':bool(np.isclose(u,fit['bounds']).any()),'linear_unclipped_input':float(raw_inverse) if arm=='linear' else None})
        plan={'property':prop,'experiments':experiments,'calibration':fit,'observations':observations}
        (a.output/f'{prop}.json').write_text(json.dumps(plan,indent=2)+'\n');report['properties'][prop]=plan
        print(prop,'kernel',fit['gp_kernel'],'linear slope',fit['linear_slope'])
        for e in experiments:print(e['key'],e['target_value'],'->',round(e['input_value'],5),'boundary',e['at_boundary'])
    (a.output/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
