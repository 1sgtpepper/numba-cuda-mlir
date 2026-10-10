#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
py=/opt/python/cp312-cp312/bin/python
auditwheel show "/wheel/$WHEEL_NAME" > /results/auditwheel.txt
"$py" -m pip install "/wheel/$WHEEL_NAME" 'pytest>=8,<9' filecheck cffi ml-dtypes \
  nvidia-cuda-nvcc-cu12==12.9.86 nvidia-cuda-nvrtc-cu12==12.9.86 'cuda-toolkit[cudart]==13.4.2'
pkg=$("$py" -c 'import sysconfig; print(sysconfig.get_path("platlib") + "/numba_cuda_mlir")')
nvvm=$("$py" -c 'import site; from pathlib import Path; print(next(p for r in site.getsitepackages() if (p := Path(r)/"nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so").is_file()))')
export LD_LIBRARY_PATH="$pkg/_mlir/_mlir_libs:$pkg/lib:$(dirname "$nvvm")"
"$py" - <<'PY' > /results/installed.json
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
import numba_cuda_mlir
from numba_cuda_mlir.tools import get_llvm70_capi_path
from numba_cuda_mlir.mlir_optimization import _get_llvm70_capi, _get_libnvvm_path
from numba_cuda_mlir.numba_cuda.cudadrv.libs import get_libdevice
package = Path(numba_cuda_mlir.__file__).resolve().parent
bridge = Path(get_llvm70_capi_path()).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert bridge == package / "_mlir/_mlir_libs/libMLIRToLLVM70.so", bridge
assert Path(_get_libnvvm_path().decode()).is_file()
assert Path(get_libdevice()).is_file()
_get_llvm70_capi()
payload = json.loads(Path("/inputs/payload.json").read_text())
for name, digest in payload["sha256"].items():
    path = package.parent / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
assert all("/host/" not in str(p) for p in (package, bridge))
assert "/host/" not in os.environ["LD_LIBRARY_PATH"]
print(json.dumps({"package": str(package), "bridge": str(bridge),
    "python": sys.version, "search_path": os.environ["LD_LIBRARY_PATH"],
    "distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    "wheel_payload_sha256": payload["sha256"]}, indent=2))
PY
mkdir -p /tmp/wheel-tests
cp /inputs/test_*.py /inputs/gpu_utils.py /tmp/wheel-tests/
cd /tmp/wheel-tests
"$py" -m pytest -q --junitxml=/results/pytest.xml . 2>&1 | tee /results/pytest.txt
"$py" - <<'PY'
import xml.etree.ElementTree as ET
cases = ET.parse("/results/pytest.xml").getroot().findall(".//testcase")
assert len(cases) == 42, len(cases)
assert all(not list(c) for c in cases), "installed-wheel tests failed or skipped"
print("INSTALLED_EXACT_WHEEL_MLIR_CASES=42; SKIPPED=0")
PY
