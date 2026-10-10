# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("fixtures", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = []
for fixture in sorted(args.fixtures.rglob("*.mlir")):
    relative = fixture.relative_to(args.fixtures).as_posix()
    result = subprocess.run(
        [str(args.binary), str(fixture), "--dump-llvm"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    leaf = args.output / relative.removesuffix(".mlir")
    leaf.parent.mkdir(parents=True, exist_ok=True)
    leaf.with_suffix(".stdout").write_text(result.stdout)
    leaf.with_suffix(".stderr").write_text(result.stderr)
    rows.append(
        {
            "fixture": relative,
            "source_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "exit": result.returncode,
            "nvvm_debug_version_mismatch": "DBG version 0.0 incompatible with current version 3.2"
            in result.stderr,
            "stderr_file": leaf.with_suffix(".stderr").relative_to(args.output).as_posix(),
            "stdout_file": leaf.with_suffix(".stdout").relative_to(args.output).as_posix(),
        }
    )
assert len(rows) == 49, len(rows)
(args.output / "inventory.json").write_text(json.dumps(rows, indent=2) + "\n")
print("Unpiped native diagnostic calls:", len(rows))
print("Nonzero exits:", sum(row["exit"] != 0 for row in rows))
print(
    "Exact NVVM debug mismatch diagnostics:",
    sum(row["nvvm_debug_version_mismatch"] for row in rows),
)
print("These diagnostic calls do not replace or green the unmodified native lit suite.")
