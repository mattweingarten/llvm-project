//===-llvm/Analysis/MemprofTypeAnalysis.h  type analysis for memory profiles- *-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_MEMPROFTYPEANALYSIS_H
#define LLVM_ANALYSIS_MEMPROFTYPEANALYSIS_H

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/Caching.h"

#include <algorithm>
#include <optional>
#include <string>

#define DEBUG_TYPE "memprof"

// Histogram Granularity in bytes
#define HiSTOGRAM_GRANULARITY 8

#define STRUCT_PREFIX "struct."
#define CLASS_PREFIX "class."

// TODO SPlit out into dwarf reader

namespace llvm {
namespace memprof {

static const StringRef PointerStringRef = StringRef("Ptr");
static const StringRef UnionStringRef = StringRef("Union");
static const StringRef ArrayStringRef = StringRef("Array");
static const StringRef EnumStringRef = StringRef("Enum");
static const StringRef SubroutineStringRef = StringRef("Subroutine");
static const StringRef StringStringRef = StringRef("String");
static const StringRef UnknownStringRef = StringRef("Unknown");

#define N_ALLOCATORS 2
static const std::string AllocatorNames[N_ALLOCATORS] = {"__new_allocator",
                                                         "_MakeUniq"};

#define N_MEMBUF 1
static const std::string MembufNames[N_MEMBUF] = {"__aligned_membuf"};

class AllocTypeTree {

  struct ParserContext {
    std::string::iterator Start, End;
    uint32_t OpenedAngleBrackets;
    ParserContext() = default;
    ParserContext(std::string::iterator Start, std::string::iterator End)
        : Start(Start), End(End), OpenedAngleBrackets(0) {}
    ParserContext(std::string::iterator Start, std::string::iterator End,
                  uint32_t OpenedAngleBrackets)
        : Start(Start), End(End), OpenedAngleBrackets(OpenedAngleBrackets) {}

    bool operator==(ParserContext other) {
      return Start == other.Start && End == other.End;
    }
    bool operator!=(ParserContext other) { return !(*this == other); }
    std::string str() { return std::string(Start, End); }
  };

  struct AllocTypeTreeNode {
    std::string TypeName;

    // We need a slightly different string when looking into DWARF file for type
    // declarations
    std::string DwarfTypeName;

    llvm::SmallVector<AllocTypeTreeNode *, 16> Children;

    uint32_t Size;
    uint32_t Offset;
    uint64_t FieldAccessCount;

    AllocTypeTreeNode() : Offset(0), FieldAccessCount(0) {};
    AllocTypeTreeNode(std::string DwarfTypeName, uint32_t Size, uint32_t Offset)
        : TypeName(DwarfTypeName), DwarfTypeName(DwarfTypeName), Size(Size),
          Offset(Offset), FieldAccessCount(0) {};

    std::string buildDwarfTypeName(bool Base);
    void resolveSize(LLVMContext &Ctx);
    bool hasChildren() const;
    size_t height() const;
    void printDwarf(raw_ostream &OS, size_t Level, size_t Height) const;
    void printParsed(raw_ostream &OS) const;
    void printYAML(raw_ostream &OS, size_t Level, std::string Prefix) const;
    void setDwarfName(LLVMContext &Ctx);
    friend raw_ostream &operator<<(raw_ostream &OS,
                                   const AllocTypeTreeNode &Node) {
      Node.printParsed(OS);
      OS << "\n";
      size_t Height = Node.height();
      for (size_t I = Height; I > 0; I--) {
        OS << "{" << I << "}" << " ";
        Node.printDwarf(OS, I, Height);
        OS << "\n";
      }
      return OS;
    }

    AllocTypeTreeNode *getChild(size_t I);

    void preOrderApply(std::function<void(AllocTypeTreeNode *)> F);
    void attachFieldAccessCount(std::vector<uint64_t> &Histogram,
                                uint32_t MaxOffset);
  };

  enum LookAhead { Subtype, Leaf };

  static LookAhead lookahead(ParserContext &Ctxt);

  static std::optional<AllocTypeTree::ParserContext>
  getAllocator(std::string &AllocatorString);

  static std::optional<AllocTypeTree::ParserContext>
  getMembufInternal(std::string &MembufString);

  static std::string parseTypeName(ParserContext &Ctxt);

  static ParserContext consumeBrackets(ParserContext Ctxt);

  static ParserContext getNextTypeInList(ParserContext &Ctxt);

  static std::optional<ParserContext> unwrapLibName(std::string &Str,
                                                    const std::string *Wrappers,
                                                    size_t WrappersSize,
                                                    bool StartsWith);

  AllocTypeTreeNode *parseType(ParserContext &Ctxt);

  AllocTypeTreeNode *Root;

  friend raw_ostream &operator<<(raw_ostream &OS, const AllocTypeTree &Tree) {
    if (!Tree.empty())
      Tree.Root->printYAML(OS, 2, "");
    // OS << "Parsed Tree: " << *Tree.Root << "\n";
    return OS;
  }

  void preOrderApply(std::function<void(AllocTypeTreeNode *)> F);

  size_t height();

  void deleteNode(AllocTypeTreeNode *Node);
  static std::optional<StringRef> resolveDITypeName(DIType *Ty);

  void attachFieldAccessCount(std::vector<uint64_t> &Histogram);

  static uint64_t sumHistogram(std::vector<uint64_t> &Histogram, uint32_t From,
                               uint32_t To);

  static StringRef getNameFromDIE(DIType *DITy);

  static bool shouldContinueResolveType(std::string DwarfTypeName);

public:
  AllocTypeTree() = default;
  AllocTypeTree(ParserContext &Ctxt);
  AllocTypeTree(std::string Ty);
  AllocTypeTree(DIType *Node);
  ~AllocTypeTree() {
    // TODO: Fix here
    // deleteNode(Root);
  }

  void clearNodesExceptRoot();

  bool empty() const { return Root == nullptr; }

  void buildDwarfNames(LLVMContext &Ctx);

  void resolveLayoutInformation(LLVMContext &Ctx, DebugInfoFinder &Finder);

  void buildResolvedTypeTree(LLVMContext &Ctx, const DataLayout &DL);

  void mergeWithHistogram(AccessCountHistogram Hist);

  static DIType *lookupDebugInfo(DebugInfoFinder &Finder,
                                 const std::string &TypeName);

  static std::optional<AllocTypeTree> parseFuntionName(std::string &FName);
};

} // namespace memprof
} // namespace llvm

#endif