# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate CLI-default metadata acceptance from translation and GPU execution."""

import ctypes as c
import hashlib
import json
import sys
from pathlib import Path

library, fixture, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
nvvm = c.CDLL(str(library.resolve()))


def bind(name, argtypes):
    function = getattr(nvvm, name)
    function.restype = c.c_int
    function.argtypes = argtypes
    return function


version = bind("nvvmIRVersion", [c.POINTER(c.c_int)] * 4)
create = bind("nvvmCreateProgram", [c.POINTER(c.c_void_p)])
destroy = bind("nvvmDestroyProgram", [c.POINTER(c.c_void_p)])
add = bind("nvvmAddModuleToProgram", [c.c_void_p, c.c_char_p, c.c_size_t, c.c_char_p])
compile_program = bind("nvvmCompileProgram", [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)])
log_size = bind("nvvmGetProgramLogSize", [c.c_void_p, c.POINTER(c.c_size_t)])
get_log = bind("nvvmGetProgramLog", [c.c_void_p, c.c_void_p])
ptx_size = bind("nvvmGetCompiledResultSize", [c.c_void_p, c.POINTER(c.c_size_t)])
get_ptx = bind("nvvmGetCompiledResult", [c.c_void_p, c.c_void_p])
fields = [c.c_int() for _ in range(4)]
assert version(*(c.byref(field) for field in fields)) == 0
reported = [field.value for field in fields]
source = fixture.read_text()
node = "!2 = !{i32 2, i32 0, i32 3, i32 2}"
assert source.count(node) == 1
options = (c.c_char_p * 2)(b"-arch=compute_80", b"-opt=3")
results = {}
for role, values in (("default", [2, 0, 0, 0]), ("reported", reported)):
    metadata = "!2 = !{" + ", ".join(f"i32 {value}" for value in values) + "}"
    module = source.replace(node, metadata, 1).encode()
    (output / f"{role}.ll").write_bytes(module)
    program = c.c_void_p()
    assert create(c.byref(program)) == 0
    try:
        assert add(program, module, len(module), b"metadata_control") == 0
        status = compile_program(program, len(options), options)
        size = c.c_size_t()
        assert log_size(program, c.byref(size)) == 0
        log = c.create_string_buffer(max(1, size.value))
        assert get_log(program, log) == 0
        message = log.value.decode()
        (output / f"{role}.log").write_text(message)
        results[role] = {"tuple": values, "status": status, "log": message}
        if status == 0:
            assert ptx_size(program, c.byref(size)) == 0
            ptx = c.create_string_buffer(size.value)
            assert get_ptx(program, ptx) == 0
            text = ptx.value.decode()
            assert ".entry flags_i32" in text and ".entry unused_volatile" in text
            (output / f"{role}.ptx").write_text(text)
        elif role == "reported":
            raise RuntimeError(f"runtime tuple failed: status={status}, log={message}")
        else:
            assert status == 3 and "DBG version 0.0 incompatible" in message
    finally:
        assert destroy(c.byref(program)) == 0
report = {
    "library": str(library.resolve()),
    "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "reported_tuple": reported,
    "options": [option.decode() for option in options],
    "results": results,
    "scope": "metadata-only NVVM acceptance; no unmodified CLI, lit or GPU execution",
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
