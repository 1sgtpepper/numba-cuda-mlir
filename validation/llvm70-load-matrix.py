# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private differential smoke check for the LLVM 7.1.0 load bridge.

Run from separate baseline and candidate package environments, each with its
matching native CAPI library installed. This is not a project test: the raw
IR matrix includes address spaces that are not being claimed as NVVM kernel
ABI inputs. It deliberately uses the project's CAPI helper rather than a
second ctypes declaration so its output and error buffers retain the existing
ownership/free behavior.
"""

import argparse
from pathlib import Path
import re
from unittest.mock import patch

from numba_cuda_mlir import tools
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir_optimization import _call_llvm70_capi


SCALAR_TYPES = (
    ("i8", "i8"),
    ("i16", "i16"),
    ("i32", "i32"),
    ("i64", "i64"),
    ("f32", "float"),
    ("f64", "double"),
)
ADDRESS_SPACES = (0, 1, 3, 5)
PTX_TYPES = ("i32", "i64", "f32", "f64")
PTX_ADDRESS_SPACES = (0, 1)

LLVM_LOAD_RE = re.compile(
    r"^\s*%[-a-zA-Z$._0-9]+\s+= load (volatile )?(.+?), "
    r"(.+?) %[-a-zA-Z$._0-9]+(?:,|$)"
)
PTX_LD_RE = re.compile(r"^\s*(?:@!?%[-a-zA-Z$._0-9]+\s+)?(ld(?:\.[A-Za-z0-9_]+)+)\b")
PTX_ENTRY_RE = re.compile(r"(?m)^\s*(?:\.visible\s+)?\.entry\s+([^\s(]+)")


def _mlir_ptr_type(address_space):
    return "!llvm.ptr" if address_space == 0 else f"!llvm.ptr<{address_space}>"


def _llvm_ptr_type(llvm_type, address_space):
    space = "" if address_space == 0 else f" addrspace({address_space})"
    return f"{llvm_type}{space}*"


def _invoke(source, *, gen_llvmir, opt_level=2):
    with ir.Context():
        module = ir.Module.parse(source)
        assert module.operation.verify(), "generated MLIR module did not verify"
        # _call_llvm70_capi imports the helper inside the function, and its
        # target_options.get default eagerly calls it even when `chip` exists.
        with patch.object(tools, "get_gpu_compute_capability", return_value="sm_90"):
            return _call_llvm70_capi(
                module,
                {"chip": "sm_90", "opt_level": opt_level},
                gen_llvmir=gen_llvmir,
            )


def _raw_ir_source():
    args = ", ".join(f"%p{space}: {_mlir_ptr_type(space)}" for space in ADDRESS_SPACES)
    args += ", %output: !llvm.ptr<1>"
    body = []
    for mlir_type, _ in SCALAR_TYPES:
        for space in ADDRESS_SPACES:
            pointer_type = _mlir_ptr_type(space)
            body.extend(
                f"%v_{mlir_type}_as{space}_{index} = llvm.load volatile "
                f"%p{space} : {pointer_type} -> {mlir_type}"
                for index in range(3)
            )
            body.append(
                f"%v_{mlir_type}_as{space}_plain = llvm.load %p{space} : "
                f"{pointer_type} -> {mlir_type}"
            )
    body.extend(
        (
            "%store_value = llvm.mlir.constant(0 : i32) : i32",
            "llvm.store volatile %store_value, %output : i32, !llvm.ptr<1>",
            "llvm.store %store_value, %output : i32, !llvm.ptr<1>",
            "llvm.return",
        )
    )
    body_text = "\n".join(f"      {line}" for line in body)
    return f"""module {{
  gpu.module @kernels {{
    llvm.func @raw_matrix({args}) attributes {{gpu.kernel}} {{
{body_text}
    }}
  }}
}}"""


def _llvm_load_records(llvm_ir):
    records = []
    for line in llvm_ir.splitlines():
        match = LLVM_LOAD_RE.match(line)
        if match:
            records.append(
                (match.group(1) is not None, match.group(2).strip(), match.group(3).strip())
            )
    return records


def check_raw_ir_matrix(output, baseline):
    llvm_ir = _invoke(_raw_ir_source(), gen_llvmir=True).decode()
    (output / "raw-ir.ll").write_text(llvm_ir)
    expected = []
    for _, llvm_type in SCALAR_TYPES:
        for space in ADDRESS_SPACES:
            pointer_type = _llvm_ptr_type(llvm_type, space)
            expected.extend(
                (volatile and not baseline, llvm_type, pointer_type)
                for volatile in (True, True, True, False)
            )
    actual = _llvm_load_records(llvm_ir)
    if actual != expected:
        mismatches = [
            (
                index,
                expected[index] if index < len(expected) else None,
                actual[index] if index < len(actual) else None,
            )
            for index in range(max(len(expected), len(actual)))
            if index >= len(expected) or index >= len(actual) or expected[index] != actual[index]
        ][:8]
        raise AssertionError(
            f"raw IR expected {len(expected)} ordered load records, got "
            f"{len(actual)}; first differences (index, expected, actual): {mismatches}"
        )

    stores = [line for line in llvm_ir.splitlines() if re.match(r"^\s*store ", line)]
    store_flags = [line.lstrip().startswith("store volatile ") for line in stores]
    assert store_flags == [True, False], f"unexpected store controls: {stores}"


def _ptx_source(mlir_type, address_space):
    pointer_type = _mlir_ptr_type(address_space)
    return f"""module {{
  gpu.module @kernels {{
    llvm.func @ptx_probe(%p: {pointer_type}) attributes {{gpu.kernel}} {{
      %unused0 = llvm.load volatile %p : {pointer_type} -> {mlir_type}
      %unused1 = llvm.load volatile %p : {pointer_type} -> {mlir_type}
      %ordinary = llvm.load %p : {pointer_type} -> {mlir_type}
      llvm.return
    }}
  }}
}}"""


def check_optimized_ptx_case(mlir_type, address_space, output, baseline):
    ptx = _invoke(_ptx_source(mlir_type, address_space), gen_llvmir=False, opt_level=3).decode()
    (output / f"as{address_space}-{mlir_type}.ptx").write_text(ptx)
    entries = list(PTX_ENTRY_RE.finditer(ptx))
    assert len(entries) == 1, f"expected one PTX entry, found {len(entries)}"
    entry = entries[0]
    assert entry.group(1) == "ptx_probe", f"unexpected PTX entry: {entry.group(1)}"
    open_brace = ptx.find("{", entry.end())
    assert open_brace >= 0, "PTX entry has no body"
    depth = 0
    close_brace = None
    for index in range(open_brace, len(ptx)):
        if ptx[index] == "{":
            depth += 1
        elif ptx[index] == "}":
            depth -= 1
            if depth == 0:
                close_brace = index
                break
    assert close_brace is not None, "PTX entry body is not closed"

    data_loads = []
    entry_body = ptx[open_brace + 1 : close_brace]
    for line in entry_body.splitlines():
        match = PTX_LD_RE.match(line)
        if not match:
            continue
        opcode_parts = match.group(1).split(".")[1:]
        # Kernel argument loads are expected and are not the tested memory read.
        if "param" not in opcode_parts:
            data_loads.append((opcode_parts, line.strip()))
    volatile_loads = [entry for entry in data_loads if "volatile" in entry[0]]
    count = 0 if baseline else 2
    if len(data_loads) != count or len(volatile_loads) != count:
        raise AssertionError(
            f"AS{address_space} {mlir_type}: expected exactly {count} volatile "
            f"data loads and no ordinary one at opt=3; got {data_loads}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    try:
        check_raw_ir_matrix(args.output, args.baseline)
    except AssertionError as error:
        failures.append(f"raw IR matrix: {error}")

    for address_space in PTX_ADDRESS_SPACES:
        for mlir_type in PTX_TYPES:
            try:
                check_optimized_ptx_case(mlir_type, address_space, args.output, args.baseline)
            except AssertionError as error:
                failures.append(f"optimized PTX AS{address_space} {mlir_type}: {error}")

    if failures:
        raise SystemExit("\n".join(failures))
    state = "baseline bug confirmed" if args.baseline else "candidate contract passed"
    print(f"{state}: raw IR24 combinations/96 loads; optimized PTX8 cases")


if __name__ == "__main__":
    main()
