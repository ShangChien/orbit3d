#!/usr/bin/env python3
"""Package explicitly selected weights unchanged, with source/output checksums."""
import argparse,hashlib,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dpa4-src-dir',type=Path)
    p.add_argument('--dpa4-variants',nargs='*',default=[],choices=['Nano','Mini','Neo','Air','Plus'])
    p.add_argument('--hyff-src-dir',type=Path)
    p.add_argument('--hyff-tiers',nargs='*',default=[],choices=['presto','vivace','moderato','andante','largo'])
    p.add_argument('--unimol-all-h',type=Path)
    p.add_argument('--unimol-no-h',type=Path)
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();selected=[]
    for variants,root in [(a.dpa4_variants,a.dpa4_src_dir),(a.hyff_tiers,a.hyff_src_dir)]:
        if variants and root is None:p.error('Each selected model family requires its explicit source directory')
    for variant in a.dpa4_variants:
        name=f'DPA4-{variant}-OMol25-100M-v20260820.pt'
        selected.append((a.dpa4_src_dir/name,Path('dpa4')/name))
    for tier in a.hyff_tiers:
        source=a.hyff_src_dir/tier
        if not source.is_dir():raise FileNotFoundError(source)
        files=sorted(path for path in source.rglob('*') if path.is_file())
        if not files:raise ValueError(f'Empty HyFF checkpoint: {source}')
        selected.extend((path,Path('hyff/checkpoints')/tier/path.relative_to(source)) for path in files)
    for tag,source in [('all_h',a.unimol_all_h),('no_h',a.unimol_no_h)]:
        if source is not None:selected.append((source,Path('unimol')/f'mol_pre_{tag}_220816.pt'))
    if not selected:p.error('Select at least one model; no private source paths or default model substitutions are used')
    # Resolve all requested inputs before publishing any output.
    for source,_ in selected:
        if not source.is_file():raise FileNotFoundError(source)
    if a.out_dir.exists() and any(a.out_dir.iterdir()):raise FileExistsError('Use a fresh package directory')
    manifest=[]
    for source,relative in selected:
        target=a.out_dir/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        with source.open('rb') as stream:before=hashlib.file_digest(stream,'sha256').hexdigest()
        with target.open('rb') as stream:after=hashlib.file_digest(stream,'sha256').hexdigest()
        if before!=after:raise IOError(f'Copy checksum mismatch: {relative}')
        manifest.append({'source_name':source.name,'packaged_path':str(relative),'bytes':target.stat().st_size,'sha256':after,'transformation':'none'})
    (a.out_dir/'weights_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(row['bytes'] for row in manifest),'output':str(a.out_dir)}))


if __name__=='__main__':main()
