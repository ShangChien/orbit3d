import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('source_recovery',Path(__file__).parents[1]/'scripts/audit_source_graph_reconstruction.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_fine_grid_roundtrip_preserves_geometry_within_resolution():
    points=np.array([[-.126,.118,.005],[1.234,-.251,.009],[.371,1.218,-.641]])
    actual=module.quantize(points)
    np.testing.assert_allclose(actual.mean(0),0,atol=1e-7)
    original_dist=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    new_dist=np.linalg.norm(actual[:,None]-actual[None,:],axis=-1)
    np.testing.assert_allclose(original_dist,new_dist,atol=.018)
