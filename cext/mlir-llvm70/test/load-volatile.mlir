// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {
    llvm.func @flags_i32(%input: !llvm.ptr<1>, %output: !llvm.ptr<1>) attributes {gpu.kernel} {
      %v = llvm.load volatile %input : !llvm.ptr<1> -> i32
      %n = llvm.load %input : !llvm.ptr<1> -> i32
      llvm.store volatile %v, %output : i32, !llvm.ptr<1>
      llvm.store %n, %output : i32, !llvm.ptr<1>
      llvm.return
    }

    llvm.func @unused_volatile(%input: !llvm.ptr<1>) attributes {gpu.kernel} {
      %a = llvm.load volatile %input : !llvm.ptr<1> -> i32
      %b = llvm.load volatile %input : !llvm.ptr<1> -> i32
      llvm.return
    }
  }
}

// CHECK-LABEL: define ptx_kernel void @flags_i32(
// CHECK: load volatile i32, i32 addrspace(1)*
// CHECK: load i32, i32 addrspace(1)*
// CHECK: store volatile i32
// CHECK: store i32
// CHECK: ret void

// CHECK-LABEL: define ptx_kernel void @unused_volatile(
// CHECK-COUNT-2: load volatile i32, i32 addrspace(1)*
// CHECK: ret void
