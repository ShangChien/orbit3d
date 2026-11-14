"""Relocate a completed octree inference run without copying training datasets."""
import argparse,copy,hashlib,json,shutil,tarfile
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def export_bundle(run, output, wheel, requirements, asset_root):
    run,output,wheel,requirements,asset_root=map(lambda p:Path(p).resolve(),[run,output,wheel,requirements,asset_root])
    source=run/'run_config.json';cfg=json.loads(source.read_text())
    if cfg.get('generator_config',{}).get('backend')!='uni3dar_octree':
        raise ValueError('This bundle format exports the explicitly named octree backend')
    paths=cfg['mount_paths'];roles={}
    for role in ['checkpoint_load_path','forcefield_path','property_checkpoint_dir']:
        path=Path(paths[role]);roles[role]=path if path.is_absolute() else asset_root/path
    domain=Path(cfg['oracle_config']['applicability_domain_path'])
    roles['applicability_domain_path']=domain if domain.is_absolute() else asset_root/domain
    for path in [*roles.values(),wheel,requirements]:
        if not path.exists():raise FileNotFoundError(path)
        if path.is_dir() and output.is_relative_to(path):raise ValueError('Bundle output cannot be inside a copied input')
    output.mkdir(parents=True,exist_ok=False)
    relocated={}
    for role,path in roles.items():
        relative=Path('models')/role/path.name
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if path.is_dir():shutil.copytree(path,target)
        else:shutil.copyfile(path,target)
        relocated[role]=str(relative)
    current=copy.deepcopy(cfg);current['mode']='predict';current['generator_config']['grpo_steps']=0
    current['mount_paths']={key:relocated[key] for key in ['checkpoint_load_path','forcefield_path','property_checkpoint_dir']}
    current['mount_paths'].update(output_dir='outputs/prediction',checkpoint_save_dir='runtime/checkpoints')
    current['oracle_config']['applicability_domain_path']=relocated['applicability_domain_path']
    (output/'config.json').write_text(json.dumps(current,indent=2)+'\n')
    artifacts=output/'package';artifacts.mkdir()
    shutil.copyfile(wheel,artifacts/wheel.name);shutil.copyfile(requirements,artifacts/'requirements.lock.txt')
    shutil.copyfile(__file__,artifacts/'export_inference_bundle.py')
    reference=output/'reference_run';reference.mkdir()
    for name in ['run_config.json','candidates.csv','raw_trajectories.npz','raw_symbols.json',
                 'generated.sdf','qualified.sdf','relaxation_traces.json','summary.json',
                 'run_manifest.json','input_manifest.json','source_snapshot.tar.gz']:
        shutil.copyfile(run/name,reference/name)
    readme="""# Relocated inference bundle

The models, fitted scalers, applicability domain, configuration, source wheel and
original replay cohort are contained here. Training datasets and the upstream
pretraining file are not required for inference. This is not an operating-system
image; install the exported locked dependencies in a compatible Python environment.

Run commands from this bundle directory. CUDA configuration is retained from the
original remote run. Use a GPU machine for that configuration.

    python -m pip install --no-deps package/"""
    readme+=wheel.name+"""

    loom --config config.json

For full replay, pass the relocated generator path printed in config.json:

    python -c 'from loom.utils.replay_validation import main; main()' \
      --config config.json --checkpoint GENERATOR_PATH \
      --reference-run reference_run --output verification

The exporter has not changed the original run or checkpoint. The manifest binds
every bundled file by SHA-256. Verification outputs are written separately and
are not part of that original file manifest. Dependency source-build switches
for DeepMD remain DP_ENABLE_TENSORFLOW=0 and DP_ENABLE_PYTORCH=0, as in the repository.
"""
    readme=readme.replace('GENERATOR_PATH',relocated['checkpoint_load_path'])
    readme += '''

The short verification entrypoint is:

    python verify_bundle.py

For a new compatible environment, install the exported versions and hashes with:

    DP_ENABLE_TENSORFLOW=0 DP_ENABLE_PYTORCH=0 python -m pip install --require-hashes --no-binary deepmd-kit -r package/requirements.lock.txt

Then install the bundled source wheel without resolving dependencies again.
'''
    (output/'README.md').write_text(readme)
    (output/'verify_bundle.py').write_text('''import json,sys
from loom.utils.replay_validation import main
configuration=json.load(open('config.json'))
sys.argv=['verify_bundle','--config','config.json','--checkpoint',configuration['mount_paths']['checkpoint_load_path'],
          '--reference-run','reference_run','--output','verification']
main()
''')
    files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'format':'orbit3d_inference_bundle_v1','source_run_configuration_sha256':digest(source),
              'source_wheel_sha256':digest(wheel),'source_requirement_lock_sha256':digest(requirements),
              'relocated_inputs':relocated,'files':files}
    (output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser()
    for name in ['run','output','wheel','requirements','asset-root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--archive',type=Path,help='Optional uncompressed tar destination, typically on persistent storage')
    a=p.parse_args()
    if a.archive and (a.archive.exists() or a.archive.resolve().is_relative_to(a.output.resolve())):
        raise ValueError('Use a fresh archive path outside the bundle directory')
    manifest=export_bundle(a.run,a.output,a.wheel,a.requirements,a.asset_root)
    if a.archive:
        if a.archive.exists():raise FileExistsError(a.archive)
        a.archive.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(a.archive,'w') as archive:archive.add(a.output,arcname=a.output.name)
    print(json.dumps({'files':len(manifest['files']),'generator':manifest['relocated_inputs']['checkpoint_load_path'],
                      'wheel_sha256':manifest['source_wheel_sha256']},indent=2))


if __name__=='__main__':main()
