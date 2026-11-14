from rdkit import Chem
from rdkit.Chem import AllChem

from orbit3d.oracles.unimol2_task import support_exclusion, summarize_task


def test_all_proposals_and_novel_graphs_are_distinct_metrics():
    rows=[dict(legal=False,evaluated=False),
          dict(legal=True,evaluated=False,support_exclusion='Te'),
          dict(legal=True,evaluated=True,smiles='CC',prediction_eV=[-7.,-1.]),
          dict(legal=True,evaluated=True,smiles='CC',prediction_eV=[-7.,-1.]),
          dict(legal=True,evaluated=True,smiles='CO',prediction_eV=[-6.,-2.])]
    r=summarize_task(rows,'homo',-7.,set(),{'CC'})
    assert r['hit_yield']==2/5 and r['conditional_hit_rate']==2/3
    assert r['unique_source_novel_hits']==1 and r['unique_all_asset_novel_hits']==0
    assert summarize_task(rows,'homo',-6.,set(),{'CC'})['unique_all_asset_novel_hits']==1


def test_nitro_source_is_not_rejected_by_predictor_support():
    mol=Chem.AddHs(Chem.MolFromSmiles('C[N+](=O)[O-]'))
    assert AllChem.EmbedMolecule(mol,randomSeed=42)==0
    assert support_exclusion(mol) is None
    tellurium=Chem.MolFromSmiles('[Te]')
    assert support_exclusion(tellurium)=='unsupported_elements:Te'
