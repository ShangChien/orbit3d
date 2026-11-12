import json
import numpy as np
import pytest
from rdkit import Chem
from orbit3d.oracles.graph_features import RAW_GRAPH,NORMALIZED_GRAPH
from orbit3d.oracles.graph_features import fingerprint_graph
from orbit3d.oracles.fingerprint_oracle import FingerprintOracle
from orbit3d.oracles.applicability import MorganDomain


def test_equivalent_sulfoxide_forms_have_the_same_features_and_domain():
    neutral=Chem.MolFromSmiles('CS(C)=O')
    separated=Chem.MolFromSmiles('C[S+](C)[O-]')
    original=Chem.MolToSmiles(neutral)
    model=FingerprintOracle()
    np.testing.assert_array_equal(model.features([neutral]),model.features([separated]))
    assert Chem.MolToSmiles(neutral)==original
    domain=MorganDomain([original],.99)
    assert domain.score(separated)==1.
    # Charge normalization does not collapse a different chemical formula.
    assert not np.array_equal(model.features([neutral]),model.features([Chem.MolFromSmiles('CSC')]))


def test_legacy_fingerprint_weights_keep_the_original_feature_contract(tmp_path):
    molecules=[Chem.MolFromSmiles(s) for s in ['CS(C)=O','CSC','CCO','CCN']]
    model=FingerprintOracle(graph_normalization=RAW_GRAPH)
    model.fit(molecules,np.array([[-6.,-1.],[-7.,-2.],[-8.,-3.],[-9.,-4.]]),tmp_path,n_models=1,n_trees=4,n_jobs=1)
    meta_path=tmp_path/'metadata.json';meta=json.loads(meta_path.read_text())
    meta['oracle_type']='fingerprint_bootstrap_v1';meta.pop('graph_normalization')
    meta_path.write_text(json.dumps(meta))
    loaded=FingerprintOracle.load_from_dir(tmp_path)
    assert loaded.graph_normalization==RAW_GRAPH
    np.testing.assert_array_equal(loaded.features(molecules),model.features(molecules))
    np.testing.assert_array_equal(loaded.predict_3d_mols(molecules)[0],model.predict_3d_mols(molecules)[0])
    assert not np.array_equal(loaded.features(molecules),FingerprintOracle().features(molecules))


def test_domain_save_reload_and_legacy_contract(tmp_path):
    path=tmp_path/'domain.json';equivalent=Chem.MolFromSmiles('C[S+](C)[O-]')
    domain=MorganDomain(['CS(C)=O'],.99);domain.save(path)
    loaded=MorganDomain.load(path)
    assert loaded.graph_normalization==NORMALIZED_GRAPH
    assert loaded.assess(equivalent)['in_domain']
    data=json.loads(path.read_text());data.pop('schema_version');data.pop('graph_normalization')
    path.write_text(json.dumps(data))
    legacy=MorganDomain.load(path)
    assert legacy.graph_normalization==RAW_GRAPH
    assert not legacy.assess(equivalent)['in_domain']


def test_graph_features_remove_nonphysical_stereo_annotations():
    source=Chem.MolFromSmiles('CCO')
    annotated=Chem.Mol(source)
    annotated.GetAtomWithIdx(1).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    model=FingerprintOracle()
    np.testing.assert_array_equal(model.features([source]),model.features([annotated]))
    domain=MorganDomain(['CCO'],.99)
    assert domain.score(annotated)==1.
    assert annotated.GetAtomWithIdx(1).GetChiralTag()==Chem.ChiralType.CHI_TETRAHEDRAL_CW


def test_feature_admission_preserves_charge_and_other_candidates(tmp_path,monkeypatch):
    # RDKit's P(V) rule changes this bridged oxonium input's total charge.
    unsupported=Chem.MolFromSmiles('C[O+]=P(C)=[O+]C')
    assert Chem.GetFormalCharge(unsupported)==2
    with pytest.raises(ValueError,match='changed total molecular charge'):
        fingerprint_graph(unsupported)
    valid=[Chem.MolFromSmiles(s) for s in ['CCO','CCN','CSC','CCC']]
    model=FingerprintOracle();model.fit(valid,np.arange(8).reshape(4,2),tmp_path,n_models=1,n_trees=4,n_jobs=1)
    expected=model.predict_3d_mols(valid[:2])[0]
    calls=[];predict=model.predict_features
    def counted(features):
        calls.append(len(features));return predict(features)
    monkeypatch.setattr(model,'predict_features',counted)
    result=model.evaluate_graph_mols([valid[0],unsupported,valid[1]])
    assert calls==[2]
    assert result['predicted']==2 and result['errors'][1]
    assert np.isnan(result['mean'][1]).all()
    np.testing.assert_array_equal(result['mean'][[0,2]],expected)
