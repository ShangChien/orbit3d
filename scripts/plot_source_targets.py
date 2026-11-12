"""Plot frozen training support, selecting the actual SDF IDs from the split table."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-csv',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads(a.manifest.read_text());ids=set(manifest['selection_ids']['train'])
    with a.train_csv.open() as stream:rows=[r for r in csv.DictReader(stream) if r['identifier'] in ids]
    assert len(rows)==len(ids) and {r['identifier'] for r in rows}==ids
    values=np.array([[float(r[n]) for n in ['homo','lumo']] for r in rows])
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors=['#9aadb8','#5d879e','#227b83','#bc7c46','#c6aaa0']
    support={}
    for i,name in enumerate(['homo','lumo']):
        ax=axes[i];ax.hist(values[:,i],bins=50,color='#c5d6df',edgecolor='white',linewidth=.3)
        support[name]={}
        for (q,point),color in zip(manifest['single_target_points'][name].items(),colors):
            ax.axvline(point,color=color,linewidth=1.5,label=f'P{q}: {point:.2f}')
            support[name][q]={'target_eV':point,'within_0.25_eV':int((np.abs(values[:,i]-point)<=.25).sum())}
        ax.set_xlabel(f'{name.upper()} (eV)');ax.set_ylabel('Training molecules');ax.legend(fontsize=8)
        ax.set_title(f'{name.upper()} point targets')
    plot=axes[2].hexbin(values[:,0],values[:,1],gridsize=35,mincnt=1,norm=LogNorm(),cmap='Blues')
    fig.colorbar(plot,ax=axes[2],label='Training molecules per bin',shrink=.85)
    correlation=float(np.corrcoef(values.T)[0,1])
    axes[2].set(xlabel='HOMO (eV)',ylabel='LUMO (eV)',title=f'Observed joint support (r = {correlation:.2f})')
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'OE62: {len(rows):,} selected training molecules; source geometry / PBE0 vacuum labels',fontsize=12)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf','svg']:fig.savefig(a.output.with_suffix('.'+extension),dpi=180)
    plt.close(fig)
    a.output.with_suffix('.json').write_text(json.dumps({'training_records':len(rows),'pearson_homo_lumo':correlation,
        'support':support,'training_table_sha256':hashlib.sha256(a.train_csv.read_bytes()).hexdigest(),
        'selection_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()
