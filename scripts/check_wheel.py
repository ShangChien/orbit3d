"""Check resources required by the installed octree and HyFF backends."""

from pathlib import Path
import zipfile

wheels = list(Path("dist").glob("*.whl"))
assert len(wheels) == 1, "Expected one wheel in dist/"
with zipfile.ZipFile(wheels[0]) as wheel:
    assert "orbit3d/compat/hyff/checkpoints/index.json" in wheel.namelist()
    assert "orbit3d/compat/hyff/LICENSE.txt" in wheel.namelist()
    assert "orbit3d/compat/hyff/NOTICE.txt" in wheel.namelist()
    assert "orbit3d/compat/unimol2/LICENSE.txt" in wheel.namelist()
    assert "orbit3d/compat/unimol2/upstream.json" in wheel.namelist()
    assert "orbit3d/oracles/unimol2_oracle.py" in wheel.namelist()
    assert "orbit3d/vendor/UNI3DAR_LICENSE.txt" in wheel.namelist()
    assert "orbit3d/vendor/uni3dar/models/uni3dar.py" in wheel.namelist()
print("Octree, HyFF and Uni-Mol2 source/license resources packaged")
