import numpy as np
from orbit3d.generator.property_sampling import property_sampling_probabilities


def test_equal_width_mass_and_sparse_exposure_cap():
    values=np.array([0.]*90+[1.]*10)
    p,meta=property_sampling_probabilities(values,bins=2,max_exposure=10)
    np.testing.assert_allclose([p[:90].sum(),p[90:].sum()],[.5,.5])
    p,meta=property_sampling_probabilities(values,bins=2,max_exposure=2)
    assert p.max()<=.02000000001
    np.testing.assert_allclose([p[:90].sum(),p[90:].sum()],[.8,.2])


def test_replay_cannot_capture_half_batch_or_exceed_per_graph_cap():
    values=np.array([0.]*900+[1.]*100)
    p,meta=property_sampling_probabilities(values,bins=2,max_exposure=5,source_count=900,max_generated_fraction=.25)
    np.testing.assert_allclose(p[900:].sum(),.25)
    assert p.max()<=5/len(values)+1e-12
    assert meta['generated_fraction']<=.25000000001


def test_constant_labels_stay_uniform():
    p,_=property_sampling_probabilities(np.ones(100))
    np.testing.assert_allclose(p,np.full(100,.01))
