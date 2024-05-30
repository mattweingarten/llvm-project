//===- AllocationTypeAnnotation.cpp - Allocation Type Annotation
//-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/AllocationTypeAnnotation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "type-annotation"

AllocationTypeAnnotation::AllocationTypeAnnotation() { return; }

PreservedAnalyses AllocationTypeAnnotation::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  // LLVM_DEBUG(dbgs() << "Hello from Allocation type annotation!\n");
  // auto &FAM =
  // AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto &Ctx = M.getContext();
  // Stage 1: We find each allocationsite for now
  for (auto &F : M) {
    // const TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(F);

    for (auto &BB : F) {
      for (auto &I : BB) {

        if (I.isDebugOrPseudoInst())
          continue;
        auto *CI = dyn_cast<CallBase>(&I);
        if (!CI)
          continue;

        auto *CalledFunction = CI->getCalledFunction();
        if (!CalledFunction)
          continue;

        if (!(CalledFunction->hasFnAttribute(Attribute::AttrKind::AllocSize)))
          continue;

        // If the alloc already has heapallocsite, we don't need to attach
        // anything
        if (I.hasMetadata("heapallocsite"))
          continue;
        // We have found a specific allocation

        LLVM_DEBUG(dbgs() << "Found alloc: " << *CI << "\n");

        auto AllocatorFuncName = CI->getFunction()->getName();

        LLVM_DEBUG(dbgs() << "Resolved allocation function Name:"
                          << AllocatorFuncName << "\n");
        MDString *MDString = MDString::get(Ctx, AllocatorFuncName);
        LLVM_DEBUG(dbgs() << "Built MDString: " << *MDString << "\n");
        auto *MDNode = MDNode::get(Ctx, MDString);
        CI->setMetadata(LLVMContext::MD_memprof_alloc_type, MDNode);
        assert(CI->hasMetadata(LLVMContext::MD_memprof_alloc_type));
      }
    }
  }

  return PreservedAnalyses::all();
}