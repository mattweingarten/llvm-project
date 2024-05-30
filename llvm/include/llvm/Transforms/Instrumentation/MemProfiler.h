//===--------- Definition of the MemProfiler class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MemProfiler class.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_MEMPROFILER_H
#include "llvm/IR/DebugInfo.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/Analysis/MemprofTypeAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Mutex.h"
#include <mutex>

#define DEBUG_TYPE "memprof"
#define DUMP_YAML(X) *(this->OF.get()) << X

namespace llvm {
class Function;
class Module;

struct MemprofUsePassOptions {
  std::string ProfileFileName;
  std::string AccessCountFileName;
  bool dumpYAML;
};

namespace vfs {
class FileSystem;
} // namespace vfs

/// Public interface to the memory profiler pass for instrumenting code to
/// profile memory accesses.
///
/// The profiler itself is a function pass that works by inserting various
/// calls to the MemProfiler runtime library functions. The runtime library
/// essentially replaces malloc() and free() with custom implementations
/// that record data about the allocations.
class MemProfilerPass : public PassInfoMixin<MemProfilerPass> {
public:
  explicit MemProfilerPass();
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

/// Public interface to the memory profiler module pass for instrumenting code
/// to profile memory allocations and accesses.
class ModuleMemProfilerPass : public PassInfoMixin<ModuleMemProfilerPass> {
public:
  explicit ModuleMemProfilerPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};

class MemProfUsePass : public PassInfoMixin<MemProfUsePass> {
public:
  // explicit MemProfUsePass(const MemProfUsePass &MemprofUsePass);
  explicit MemProfUsePass(MemprofUsePassOptions MemProfOpt,
                          IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr);
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  explicit MemProfUsePass(std::string MemoryProfileFileName,
                          IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr);

private:
  std::string MemoryProfileFileName;
  std::string AccessCountFileName;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  std::unique_ptr<llvm::raw_fd_ostream> OF;
  std::map<uint64_t, Function *> IdToFunction;
  std::map<uint64_t, uint64_t> CallStackIdToCalleeGuid;
  DebugInfoFinder Finder;

  bool dumpYAML;
  void readMemprof(Module &M, Function &F,
                   IndexedInstrProfReader *MemProfReader,
                   const TargetLibraryInfo &TLI, LLVMContext &Ctx,
                   const DataLayout &DL);

  std::optional<llvm::memprof::AllocTypeTree>
  resolveStructTypeName(LLVMContext &Ctx, CallBase &CB,
                        const memprof::AllocationInfo *AI);

  std::optional<StructType *>
  resolveStructType(LLVMContext &Ctx, const DataLayout &DL, CallBase &CB,
                    const memprof::AllocationInfo *AI);

  void printYAML(StructType *STy, memprof::FieldAccessesT &FieldAccesses,
                 const memprof::AllocationInfo *AllocInfo);
  bool shouldDumpAccessCounts();
};

} // namespace llvm

#endif
