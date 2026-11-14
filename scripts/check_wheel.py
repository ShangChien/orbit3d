"""Check resources required by the installed octree and HyFF backends."""

from pathlib import Path
import argparse
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("wheel", nargs="?", type=Path, help="Wheel to inspect; defaults to the only wheel in dist/")
args = parser.parse_args()
wheels = [args.wheel] if args.wheel else list(Path("dist").glob("*.whl"))
assert len(wheels) == 1, "Specify a wheel or leave exactly one wheel in dist/"
with zipfile.ZipFile(wheels[0]) as wheel:
    assert "loom/compat/hyff/checkpoints/index.json" in wheel.namelist()
    assert "loom/compat/hyff/LICENSE.txt" in wheel.namelist()
    assert "loom/compat/hyff/NOTICE.txt" in wheel.namelist()
    assert "loom/compat/unimol2/LICENSE.txt" in wheel.namelist()
    assert "loom/compat/unimol2/upstream.json" in wheel.namelist()
    assert "loom/oracles/unimol2_oracle.py" in wheel.namelist()
    assert "loom/vendor/UNI3DAR_LICENSE.txt" in wheel.namelist()
    assert "loom/vendor/uni3dar/models/uni3dar.py" in wheel.namelist()
print(f"{wheels[0]}: Octree, HyFF and Uni-Mol2 source/license resources packaged")
