//===-- memprof_type_analyzer_tool.h ---------------------------------------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// MemProf-private header for memprof_type_analyzer_tool.cpp.
//===----------------------------------------------------------------------===//

#ifndef MEMPROF_ANALYZER_TOOL_H
#define MEMPROF_ANALYZER_TOOL_H

namespace __memprof {

// typedef int fd_t;
// typedef unsigned long uptr;

// class TypeAnalyzerTool {

// public:
//   // IsAlloc()
//   // GetAllocTypeName()
//   // GetAllocTypeOffsets()

// private:
//   LLVMDwarfDumpProcess *dwarf_process_;
// };

// class LLVMDwarfDumpProcess {
//   bool StartDwarfDumpSubporcess();
//   bool ReadFromDwarfDump();

//   const char *path_;

//   LLVMDwarfDumpProcess(const char *path_);

//   bool ReadFromDwarfDump();
//   bool WriteToDwarfDump(const char *buffer, uptr length);
// };
} // namespace __memprof
#endif