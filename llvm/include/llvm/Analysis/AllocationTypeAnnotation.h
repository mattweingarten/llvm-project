//===- AllocationTypeAnnotation.h - -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_ANALYSIS_ALLOCTIONTYPEANNOTATION_H
#define LLVM_ANALYSIS_ALLOCTIONTYPEANNOTATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {

class AllocationTypeAnnotation
    : public PassInfoMixin<AllocationTypeAnnotation> {
public:
  explicit AllocationTypeAnnotation();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif