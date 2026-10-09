import argparse
import ctypes as c
import os
from pathlib import Path
import re

import numba_cuda_mlir
from numba_cuda_mlir._mlir import ir

parser = argparse.ArgumentParser()
parser.add_argument("library", type=Path)
parser.add_argument("fixture", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
source = args.fixture.read_text()
target = '[#nvvm_llvm70.target<chip = "sm_75">]'
assert source.count(target) == 1
source = source.replace(target, "")

lib = c.CDLL(str(args.library))
translate = lib.llvm70_translate_gpu_module_from_op
translate.restype = c.c_int
translate.argtypes = (
    [c.c_void_p] + [c.c_char_p] * 5 + [c.c_int] * 8
    + [c.POINTER(c.c_void_p), c.POINTER(c.c_size_t)] * 2
    + [c.POINTER(c.c_char_p)]
)
lib.llvm70_free.argtypes = [c.c_void_p]
get_ptr = c.pythonapi.PyCapsule_GetPointer
get_ptr.restype = c.c_void_p
get_ptr.argtypes = [c.py_object, c.c_char_p]

nvvm_path = os.environ["LLVM70_LIBNVVM"]
nvvm = c.CDLL(nvvm_path)
version = [c.c_int() for _ in range(4)]
nvvm.nvvmIRVersion.argtypes = [c.POINTER(c.c_int)] * 4
nvvm.nvvmIRVersion.restype = c.c_int
assert nvvm.nvvmIRVersion(*(c.byref(v) for v in version)) == 0
versions = [v.value for v in version]
print("Python package:", numba_cuda_mlir.__file__)
print("C API:", args.library)
print("NVVM IR/debug versions:", versions)

with ir.Context():
    module = ir.Module.parse(source)
    assert module.operation.verify()
    gpu = module.body.operations[0].operation
    raw = get_ptr(gpu._CAPIPtr, b"numba_cuda_mlir._mlir.ir.Operation._CAPIPtr")
    results = {}
    for suffix, gen_ir in [("ll", 1), ("ptx", 0)]:
        out, length, error = c.c_void_p(), c.c_size_t(), c.c_char_p()
        rc = translate(
            raw, b"sm_75", None, os.environ["LIBLLVM7"].encode(),
            nvvm_path.encode(), None, 0, gen_ir, 3, 0, *versions,
            c.byref(out), c.byref(length), None, None, c.byref(error),
        )
        try:
            if rc:
                raise RuntimeError(error.value.decode() if error.value else "missing C API diagnostic")
            result = c.string_at(out, length.value).decode()
            args.output.with_suffix("." + suffix).write_text(result)
            results[suffix] = result
            print("C API succeeded:", suffix, "bytes=", length.value)
        finally:
            if out.value:
                lib.llvm70_free(out)
            if error.value:
                lib.llvm70_free(error)

llvm_ir = results["ll"]
assert "store volatile i32" in llvm_ir
assert "store i32" in llvm_ir
unused = re.search(r"\.entry\s+unused_volatile\s*\(", results["ptx"])
assert unused, results["ptx"]
unused_ptx = results["ptx"][unused.end():]
volatile_reads = re.findall(r"ld\.volatile\.global\.[bus]32\b", unused_ptx)
if args.baseline:
    assert "load volatile" not in llvm_ir
    assert llvm_ir.count("load i32, i32 addrspace(1)*") == 4
    assert not volatile_reads
    assert not re.search(r"\bld\.", unused_ptx)
    print("BASELINE: volatile flags absent; unused reads eliminated in optimized PTX")
else:
    assert llvm_ir.count("load volatile i32, i32 addrspace(1)*") == 3
    assert llvm_ir.count("load i32, i32 addrspace(1)*") == 1
    assert len(volatile_reads) == 2, unused_ptx
    print("PATCH: flags preserved; both unused volatile reads survive optimized PTX")
