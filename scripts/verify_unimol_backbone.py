"""Check pretrained CLS parity against the downloaded official Uni-Mol encoder."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import torch
from loom.oracles.uniqsar_oracle import StandaloneUniMolModel,UNIMOL_DICT_SYMBOLS
from loom.oracles.unimol_oracle import load_pretrained_unimol_backbone
p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--official-encoder',required=True);p.add_argument('--dictionary',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(42)
assert Path(a.dictionary).read_text().splitlines()+['[MASK]']==UNIMOL_DICT_SYMBOLS
model=StandaloneUniMolModel(2,dropout=.1).eval();count=load_pretrained_unimol_backbone(model,a.weight)
spec=importlib.util.spec_from_file_location('official',a.official_encoder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ref=module.TransformerEncoderWithPair(encoder_layers=15,embed_dim=512,ffn_embed_dim=2048,attention_heads=64,no_final_head_layer_norm=True).eval();ref.load_state_dict(model.encoder.state_dict(),strict=True)
tokens=torch.tensor([[1,4,6,8,2],[1,4,8,2,0]]);coords=torch.randn(2,5,3);edge=tokens[:,:,None]*31+tokens[:,None,:]
with torch.no_grad():
 x=model.embed_tokens(tokens);bias=model.gbf_proj(model.gbf(torch.cdist(coords,coords),edge)).permute(0,3,1,2).contiguous().view(-1,5,5);pad=tokens.eq(0)
 out=model.encoder(x,attn_mask=bias.clone(),padding_mask=pad);expected=ref(x,attn_mask=bias.clone(),padding_mask=pad)[0]
 error=float((out[:,0]-expected[:,0]).abs().max());assert torch.allclose(out[:,0],expected[:,0],atol=2e-5,rtol=2e-5)
report={'loaded_backbone_tensors':count,'official_cls_max_abs_difference':error,'with_padding':True,'dictionary_exact':True,
 'weight_url':'https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/mol_pre_all_h_220816.pt',
 'encoder_url':'https://raw.githubusercontent.com/deepmodeling/Uni-Mol/main/unimol_tools/unimol_tools/models/transformers.py',
 'sha256':{k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('weight',a.weight),('official_encoder',a.official_encoder),('dictionary',a.dictionary)]}}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)
