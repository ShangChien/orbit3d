"""Source-geometry Uni-Mol2 property adaptation using the pinned official architecture."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from ..compat.unimol2.unimol2 import UniMol2Model

ARCHITECTURES={'310M':(32,1024,64),'1.1B':(64,1536,96),'tiny':(2,32,4)}


def build_unimol2(size, dropout=0.):
    layers,width,heads=ARCHITECTURES[size]
    args=SimpleNamespace(encoder_layers=layers,encoder_embed_dim=width,encoder_ffn_embed_dim=width,
        encoder_attention_heads=heads,pair_embed_dim=512 if size!='tiny' else 16,
        pair_hidden_dim=64 if size!='tiny' else 4,dropout=dropout,attention_dropout=dropout,
        activation_dropout=0.,pooler_dropout=0.,droppath_prob=0.,mode='train')
    model=UniMol2Model(args);model.register_classification_head('homo_lumo',num_classes=2)
    return model


def load_pretrained(model,path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    state=checkpoint['model']
    required={k:v for k,v in model.state_dict().items() if not k.startswith('classification_heads.')}
    missing=sorted(set(required)-set(state));wrong=[k for k,v in required.items() if k in state and v.shape!=state[k].shape]
    extra=[k for k in state if k not in required and not k.startswith(('classification_heads.','lm_head.'))]
    if missing or wrong or extra:raise ValueError(f'Incompatible Uni-Mol2 pretrained weights: missing={missing}, shape={wrong}, extra={extra}')
    result=model.load_state_dict({k:state[k] for k in required},strict=False)
    assert all(k.startswith('classification_heads.') for k in result.missing_keys) and not result.unexpected_keys
    return {'loaded_tensors':len(required),'parameter_count':sum(p.numel() for p in model.parameters())}


def source_features(mol,max_atoms=80):
    if mol is None or mol.GetNumConformers()!=1 or mol.GetNumAtoms()>max_atoms:
        raise ValueError('Uni-Mol2 requires one source conformer and at most 80 source atoms')
    keep=[i for i,a in enumerate(mol.GetAtoms()) if a.GetAtomicNum()!=1]
    if not keep:raise ValueError('Uni-Mol2 heavy-atom input is empty')
    coords=mol.GetConformer().GetPositions()[keep].astype(np.float32);tokens=np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in keep])
    if not np.isfinite(coords).all() or (tokens>=127).any():raise ValueError('Unsupported source positions or atomic numbers')
    coords-=coords.mean(0,keepdims=True);n=len(keep)
    # Exact official Unimol2FinetuneFeatureDataset values for drop_feat_prob=1.
    pair=torch.empty((n,n,2),dtype=torch.long);pair[...,0]=2;pair[...,1]=130
    return {'src_token':torch.as_tensor(tokens,dtype=torch.long),'src_pos':torch.from_numpy(coords),
        'atom_feat':torch.ones(n,8,dtype=torch.long),'atom_mask':torch.ones(n,dtype=torch.long),
        'edge_feat':torch.ones(n,n,3,dtype=torch.long),'shortest_path':torch.full((n,n),511,dtype=torch.long),
        'degree':torch.ones(n,dtype=torch.long),'pair_type':pair,'attn_bias':torch.zeros(n+1,n+1)}


class SourceUniMol2Dataset(Dataset):
    def __init__(self,mols,targets):
        if len(mols)!=len(targets):raise ValueError('Structure/target count mismatch')
        self.features=[source_features(m) for m in mols];self.targets=np.asarray(targets,dtype=np.float32)
        if not np.isfinite(self.targets).all():raise ValueError('Nonfinite targets')
        self.lengths=np.array([len(f['src_token']) for f in self.features])
    def __len__(self):return len(self.features)
    def __getitem__(self,i):return self.features[i],self.targets[i]


def collate_unimol2(items):
    features,targets=zip(*items);n=(max(len(f['src_token']) for f in features)+4)//4*4-1;b=len(features)
    out={}
    for k,example in features[0].items():
        if k=='attn_bias':
            value=example.new_full((b,n+1,n+1),float('-inf'))
            for i,f in enumerate(features):
                l=len(f['src_token'])+1;value[i,:l,:l]=f[k];value[i,l:,:l]=0
        else:
            pair=k in ['edge_feat','shortest_path','pair_type'];shape=(b,n,n,*example.shape[2:]) if pair else (b,n,*example.shape[1:]);value=example.new_zeros(shape)
            for i,f in enumerate(features):
                l=len(f['src_token'])
                if pair:value[i,:l,:l]=f[k]
                else:value[i,:l]=f[k]
        out[k]=value
    return out,torch.from_numpy(np.stack(targets))


class InternalUniMol2Oracle:
    def __init__(self,model,mean,scale,device='cuda:0'):
        self.model=model.to(device).eval();self.mean=np.asarray(mean);self.scale=np.asarray(scale);self.device=device
        self.target_cols=['homo','lumo'];self.target_units=['eV','eV'];self.has_3d_unimol=True
    @classmethod
    def load_from_dir(cls,directory,device='cuda:0'):
        directory=Path(directory);meta=json.loads((directory/'oracle_meta.json').read_text());model=build_unimol2(meta['model_size'])
        state=torch.load(directory/'model.pt',map_location='cpu',weights_only=True,mmap=True);model.load_state_dict(state,strict=True)
        return cls(model,meta['mean'],meta['scale'],device)
    @torch.no_grad()
    def predict_3d_mols(self,mols,batch_size=8):
        if not mols:return np.empty((0,2)),np.empty((0,2))
        ds=SourceUniMol2Dataset(mols,np.zeros((len(mols),2)));outputs=np.empty((len(mols),2),dtype=np.float32)
        for indices in grouped_batches(ds.lengths,batch_size):
            batch,_=collate_unimol2([ds[i] for i in indices])
            batch={k:v.to(self.device) for k,v in batch.items()}
            value=self.model(batch,classification_head_name='homo_lumo',features_only=True)[0]
            outputs[indices]=value.float().cpu().numpy()
        pred=outputs*self.scale+self.mean
        if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite Uni-Mol2 predictions')
        return pred,np.zeros_like(pred)
    def predict_3d_mols_batched(self,mols,batch_size=8):return self.predict_3d_mols(mols,batch_size)
    def forward_differentiable_3d(self,*args,**kwargs):
        raise NotImplementedError('Official Uni-Mol2 detaches input distances; this evaluator is not a coordinate-gradient oracle')


def grouped_batches(lengths,batch_size,seed=None):
    # The upstream outer-product normalization depends on padded length.
    # Keep each molecule's pad length canonical, independent of its batch mates.
    groups={}
    for i,n in enumerate(lengths):groups.setdefault((int(n)+4)//4*4-1,[]).append(i)
    rng=np.random.default_rng(seed);batches=[]
    for key in sorted(groups):
        ids=groups[key]
        if seed is not None:rng.shuffle(ids)
        batches.extend(ids[start:start+batch_size] for start in range(0,len(ids),batch_size))
    if seed is not None:rng.shuffle(batches)
    return batches
