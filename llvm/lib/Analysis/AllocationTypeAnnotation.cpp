//===- AllocationTypeAnnotation.cpp - Allocation Type Annotation
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/AllocationTypeAnnotation.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "type-annotation"


AllocationTypeAnnotation::AllocationTypeAnnotation(){
    return;
}

PreservedAnalyses AllocationTypeAnnotation::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Hello from Allocation type annotation!\n");
  return PreservedAnalyses::all();
}