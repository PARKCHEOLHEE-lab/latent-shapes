import sys
import types

# Local-test shim only.
#
# runs/.../src/model.py does `import point_cloud_utils` at module load. pcu's
# only use is watertight post-processing (SDFDecoder.reconstruct, check_watertight
# branch), which the browser port drops, and pcu is a heavy native build with no
# arm64/py3.12 wheel here. Inject a stub so the tests can import model.py without
# installing pcu. The real export scripts run in the devcontainer, where the
# nvcr pytorch image already provides the genuine point_cloud_utils.
if "point_cloud_utils" not in sys.modules:
    _pcu = types.ModuleType("point_cloud_utils")
    _pcu.make_mesh_watertight = lambda *args, **kwargs: (None, None)
    sys.modules["point_cloud_utils"] = _pcu
