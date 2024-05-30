//===-llvm/Analysis/MemprofTypeAnalysis.cpp  type analysis for memory profiles-
//*- C++ -*-===//
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

#include "llvm/Analysis/MemprofTypeAnalysis.h"
#include "llvm/BinaryFormat/Dwarf.h"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <regex>

#define DEBUG_TYPE "memprof"

namespace llvm {
namespace memprof {

void AllocTypeTree::AllocTypeTreeNode::preOrderApply(
    std::function<void(AllocTypeTreeNode *)> F) {
  F(this);
  for (auto C : Children) {
    C->preOrderApply(F);
  }
}

// Right now this is somewhat ineffecient, we should refactor out std::string
// for something like StringBuilder (std::stringstream) or Twine
std::string AllocTypeTree::AllocTypeTreeNode::buildDwarfTypeName(bool Base) {
  std::string Result = TypeName;

  if (Base) {
    // Step 1: Remove only initial namespaces
    // std::pair --> pair
    // x::y::z   --> z
    // std::pair<std::pair<x,y>> --> pair<std::pair<x,y>>
    size_t NamespaceIdx = Result.find("::");
    while (NamespaceIdx != std::string::npos) {
      Result = Result.substr(NamespaceIdx + 2, Result.size());
      NamespaceIdx = Result.find("::");
    }

    // Step 2: Remove "const" keyword if we are in base in a way that no extra
    // whitespace sneaks into the string
    Result = std::regex_replace(Result, std::regex("const "), "");
    Result = std::regex_replace(Result, std::regex(" const"), "");
  }

  Result = std::regex_replace(Result, std::regex("\\*"), " *");

  // Reorder keywords
  // unsigned long const --> const unsigned long
  std::vector<std::string> Words;
  size_t SpaceIdx = Result.find(" ");
  while (SpaceIdx != std::string::npos) {
    Words.push_back(Result.substr(0, SpaceIdx));
    Result = Result.substr(SpaceIdx + 1, Result.size());
    SpaceIdx = Result.find(" ");
  }

  Words.push_back(Result);

  auto KeyWordPriority = [](std::string x) {
    // TODO: Are there more quirks here?
    if (x.compare("const")) {
      return 1;
    } else {
      return 0;
    }
  };

  std::sort(Words.begin(), Words.end(),
            [KeyWordPriority](std::string x, std::string y) {
              return KeyWordPriority(x) < KeyWordPriority(y);
            });

  Result = Words[0];
  for (size_t I = 1; I < Words.size(); ++I) {
    Result.append(" ");
    Result.append(Words[I]);
  }

  if (hasChildren())
    Result.append("<");
  size_t I = 0;
  for (auto C : Children) {
    Result.append(C->buildDwarfTypeName(false));
    if (I != Children.size() - 1)
      Result.append(", ");
    I++;
  }

  // Add spacing between successive closing angle brackets
  // x<y<z>> --> x<y<z> >
  if (Result[Result.size() - 1] == '>') {
    Result.append(" ");
  }
  if (hasChildren())
    Result.append(">");
  return Result;
};

void AllocTypeTree::preOrderApply(std::function<void(AllocTypeTreeNode *)> F) {
  Root->preOrderApply(F);
}

bool AllocTypeTree::AllocTypeTreeNode::hasChildren() const {
  return !Children.empty();
};

AllocTypeTree::AllocTypeTreeNode *
AllocTypeTree::AllocTypeTreeNode::getChild(size_t I) {
  assert(I < Children.size() && "Access to child out of range.");
  return Children[I];
}

void AllocTypeTree::deleteNode(AllocTypeTreeNode *Node) {
  if (!Node)
    return;
  for (auto C : Node->Children) {
    deleteNode(C);
  }
  delete Node;
}

void AllocTypeTree::AllocTypeTreeNode::printParsed(raw_ostream &OS) const {
  OS << TypeName;
  if (hasChildren()) {
    OS << '<';
    size_t I = 0;
    for (auto C : Children) {
      C->printParsed(OS);
      if (I != Children.size() - 1) {
        OS << ", ";
      }
      I++;
    }
    OS << '>';
  }
}

void AllocTypeTree::AllocTypeTreeNode::printYAML(raw_ostream &OS, size_t Level,
                                                 std::string Prefix) const {

  std::string Spaces = "";
  for (size_t I = 0; I < Level; I++) {
    Spaces.append("  ");
  }

  std::string SpacesFirst = "";
  for (size_t I = 0; I < Level - 1; I++) {
    SpacesFirst.append("  ");
  }
  if (Level != 0) {

    OS << SpacesFirst << "- ";
  }
  OS << "TypeName: " << Prefix << DwarfTypeName << "\n";
  OS << Spaces << "FieldAccessCount: " << FieldAccessCount << "\n";
  OS << Spaces << "Offset: " << Offset << "\n";
  OS << Spaces << "Size: " << Size << "\n";
  if (hasChildren())
    OS << Spaces << "Children: \n";

  for (auto C : Children) {
    // C->printYAML(OS, Level + 2, Prefix.append(DwarfTypeName).append("."));
    C->printYAML(OS, Level + 2, "");
  }
}

AllocTypeTree::AllocTypeTree(std::string Ty) {
  ParserContext Ctxt(Ty.begin(), Ty.end());
  LLVM_DEBUG(dbgs() << "Parsing type after removing alloca brackets: "
                    << std::string(Ty.begin(), Ty.end()) << "\n");
  Root = parseType(Ctxt);
};

AllocTypeTree::AllocTypeTree(ParserContext &Ctxt) {
  LLVM_DEBUG(dbgs() << "Parsing type after removing alloca brackets: "
                    << std::string(Ctxt.Start, Ctxt.End) << "\n");
  Root = parseType(Ctxt);
}

AllocTypeTree::AllocTypeTree(DIType *DITyNode) {
  Root = new AllocTypeTreeNode(getNameFromDIE(DITyNode).str(),
                               DITyNode->getSizeInBits() / 8, 0);
};

std::optional<AllocTypeTree::ParserContext>
AllocTypeTree::getAllocator(std::string &AllocatorString) {
  return unwrapLibName(AllocatorString, AllocatorNames, N_ALLOCATORS, false);
}

std::optional<AllocTypeTree::ParserContext>
AllocTypeTree::getMembufInternal(std::string &MembufString) {
  // LLVM_DEBUG(dbgs() << "Looking for Membuf internal\n");
  return unwrapLibName(MembufString, MembufNames, N_MEMBUF, true);
}

std::optional<AllocTypeTree::ParserContext>
AllocTypeTree::unwrapLibName(std::string &Str, const std::string *Wrappers,
                             size_t WrappersSize, bool StartsWith) {

  for (size_t I = 0; I < WrappersSize; I++) {
    std::string Wrapper = Wrappers[I];

    if (StartsWith && Str.compare(0, Wrapper.size(), Wrapper))
      continue;

    size_t AllIdx = Str.find(Wrapper);
    if (AllIdx == std::string::npos)
      continue;
    ParserContext Ctxt(Str.begin(), Str.end());
    return std::make_optional<ParserContext>(consumeBrackets(Ctxt));
  }
  return std::nullopt;
}

AllocTypeTree::LookAhead AllocTypeTree::lookahead(ParserContext &Ctxt) {
  for (auto Curr = Ctxt.Start; Curr != Ctxt.End; ++Curr) {
    if (*Curr == '<')
      return Subtype;
  }
  return Leaf;
}

std::string AllocTypeTree::parseTypeName(ParserContext &Ctxt) {
  auto Curr = Ctxt.Start;
  for (; Curr != Ctxt.End && *Curr != '<' && *Curr != ',' && *Curr != '>';
       ++Curr) {
  }

  std::string str(Ctxt.Start, Curr);
  if (*Curr == ',')
    Curr++; // Consume extra space after comma

  Ctxt.Start = Curr;
  return str;
}

AllocTypeTree::ParserContext
AllocTypeTree::consumeBrackets(ParserContext Ctxt) {
  std::string::iterator NewStart, NewEnd;
  size_t Opened = 0;
  size_t Closed = 0;
  auto Curr = Ctxt.Start;
  for (; Curr != Ctxt.End; ++Curr) {
    if (*Curr == '>') {
      Closed++; 
      if (Closed == Opened) {
        NewEnd = Curr;
        return ParserContext(NewStart, NewEnd);
      }
    }
    if (*Curr == '<') {
      if (Opened == 0)
        NewStart = Curr + 1;
      Opened++;
    }
  }
  // Should never reach here
  assert(false && "Trying to consume bracket when no bracket present!");
  return ParserContext(NewStart, NewEnd);
}

AllocTypeTree::ParserContext
AllocTypeTree::getNextTypeInList(AllocTypeTree::ParserContext &Ctxt) {
  size_t Opened = Ctxt.OpenedAngleBrackets;
  size_t Closed = 0;
  std::string::iterator NewStart, NewEnd;
  for (auto Curr = Ctxt.Start; Curr != Ctxt.End; ++Curr) {

    if (*Curr == ',' && Closed == Opened) {
      NewStart = Ctxt.Start;
      NewEnd = Curr;
      Ctxt.Start = Curr + 2; // Consume space and comma
      return ParserContext(NewStart, NewEnd);
    }

    if (*Curr == '>')
      Closed++;
    if (*Curr == '<') {
      Opened++;
    }
  }
  assert(Opened >= Closed && "Created invalid ParserContext with Opened Angle "
                             "Brackets greater than closed Angle brackets");
  Ctxt.OpenedAngleBrackets = Opened - Closed;
  return ParserContext(Ctxt.Start, Ctxt.End);
}

AllocTypeTree::AllocTypeTreeNode *
AllocTypeTree::parseType(ParserContext &Ctxt) {

  AllocTypeTreeNode *Result = new AllocTypeTreeNode();
  auto LookAhead = lookahead(Ctxt);
  if (LookAhead == Leaf) {
    Result->TypeName = parseTypeName(Ctxt);

  } else if (LookAhead == Subtype) {
    Result->TypeName = parseTypeName(Ctxt);
    ParserContext NewCtxt = consumeBrackets(Ctxt);

    // Now handle weird edge case were part of type name is at the wrong place.
    // For example, common scenario is 'x<y> const' instead of 'const x<y>
    if (NewCtxt.End + 2 < Ctxt.End)
      Result->TypeName = std::string(NewCtxt.End + 2, Ctxt.End)
                             .append(" ")
                             .append(Result->TypeName);
    ParserContext ElemCtxt;
    while (true) {

      ElemCtxt = getNextTypeInList(NewCtxt);
      Result->Children.push_back(parseType(ElemCtxt));
      if (ElemCtxt.End == NewCtxt.End)
        break;
    }

    Ctxt.Start = Ctxt.End;
  }
  return Result;
}
std::optional<AllocTypeTree>
AllocTypeTree::parseFuntionName(std::string &FName) {
  auto CtxtOpt = getAllocator(FName);
  if (!CtxtOpt) {
    return std::nullopt;
  }
  ParserContext Ctxt = *CtxtOpt;
  return AllocTypeTree(Ctxt);
}

void AllocTypeTree::AllocTypeTreeNode::printDwarf(raw_ostream &OS, size_t level,
                                                  size_t height) const {
  if (height == level) {
    OS << DwarfTypeName << "[S: " << Size << "|O: " << Offset
       << "|A:" << FieldAccessCount << "]" << "   ";
  }

  for (auto C : Children) {
    C->printDwarf(OS, level, height - 1);
  }
}

size_t AllocTypeTree::AllocTypeTreeNode::height() const {
  if (!hasChildren())
    return 1;
  size_t MaxHeight = 1;
  for (auto C : Children) {
    size_t ChildHeight = C->height();
    MaxHeight = MaxHeight < ChildHeight + 1 ? ChildHeight + 1 : MaxHeight;
  }
  return MaxHeight;
}

void AllocTypeTree::buildDwarfNames(LLVMContext &Ctx) {

  preOrderApply([&](AllocTypeTreeNode *Node) {
    Node->DwarfTypeName = Node->buildDwarfTypeName(true);
  });
}

void AllocTypeTree::clearNodesExceptRoot() { Root->Children.clear(); }

DIType *AllocTypeTree::lookupDebugInfo(DebugInfoFinder &Finder,
                                       const std::string &TypeName) {
  // LLVM_DEBUG(dbgs() << "looking up dwarf type name " << TypeName << "\n");
  for (auto *Ty : Finder.types()) {
    auto NameOpt = resolveDITypeName(Ty);
    if (NameOpt) {
      // LLVM_DEBUG(dbgs() << "Comparing " << TypeName << " to " << *NameOpt
      //                   << " for DIType: " << *Ty << "\n");
      if (TypeName.compare(NameOpt->str()) == 0) {

        // LLVM_DEBUG(dbgs() << "Found Match!\n");
        return Ty;
      }
    }
  }
  return nullptr;
}

std::optional<StringRef> AllocTypeTree::resolveDITypeName(DIType *Ty) {

  if (Ty->getNumOperands() >= 2 && Ty->getOperand(2) &&
      isa<MDString>(Ty->getOperand(2))) {

    // Otherwise, type should be in Second Operand if present
    return std::make_optional<StringRef>(
        cast<MDString>(Ty->getOperand(2))->getString());
  }
  return std::nullopt;
}

StringRef AllocTypeTree::getNameFromDIE(DIType *DITy) {

  // The question here is, how far do we want to resolve? What is fundamentally
  // the "leaf" of this DWARF type names I think what we wanna do here is
  // resolve until we hit:

  // Base Type
  // Structure Type
  // Class Type
  // Union Type
  // Pointer Type
  // Enum Type
  // String Type
  // Array Type
  // ?

  // if (DIDerivedType *DIDer = dyn_cast<DIDerivedType>(DITy)) {
  //   DITy = DIDer->getBaseType();
  //   DIDer = dyn_cast<DIDerivedType>(DITy);
  //   if (DITy->getTag() == llvm::dwarf::DW_TAG_typedef) {
  //     DITy = DIDer->getBaseType();
  //   }
  // }

  // Handle special cases were we cannot further resolve the Type Tree and the
  // DebugInfo nodes do not have names.
  switch (DITy->getTag()) {
  case llvm::dwarf::DW_TAG_pointer_type:
    return PointerStringRef;
  case llvm::dwarf::DW_TAG_union_type:
    return UnionStringRef;
  case llvm::dwarf::DW_TAG_array_type:
    return ArrayStringRef;
  case llvm::dwarf::DW_TAG_string_type:
    return StringStringRef;
  case llvm::dwarf::DW_TAG_enumeration_type:
    return EnumStringRef;
  case llvm::dwarf::DW_TAG_reference_type:
    LLVM_DEBUG(dbgs() << "Reference type (shoudln't be possible)\n");
    return UnknownStringRef;
  case llvm::dwarf::DW_TAG_const_type:
    LLVM_DEBUG(dbgs() << "Const type (shoudln't be possible)\n");
    return UnknownStringRef;
  case llvm::dwarf::DW_TAG_subroutine_type:
    LLVM_DEBUG(dbgs() << "Subroutine type (shoudln't be possible)\n");
    return SubroutineStringRef;

  // Let all these through to rest of lookup
  case llvm::dwarf::DW_TAG_structure_type:
    return DITy->getName();
    break;
  case llvm::dwarf::DW_TAG_class_type:
    return DITy->getName();
    break;
  case llvm::dwarf::DW_TAG_base_type:
    return DITy->getName();
    break;
  case llvm::dwarf::DW_TAG_typedef:
    if (DIDerivedType *DIDer = dyn_cast<DIDerivedType>(DITy)) {
      return getNameFromDIE(DIDer->getBaseType());
    } else {
      LLVM_DEBUG(dbgs() << "Panic! Hit typedef, but is not derived type!\n");
    }
    break;
  case llvm::dwarf::DW_TAG_inheritance:
    if (DIDerivedType *DIDer = dyn_cast<DIDerivedType>(DITy)) {
      return getNameFromDIE(DIDer->getBaseType());
    } else {
      LLVM_DEBUG(
          dbgs() << "Panic! Hit inheritance, but is not derived type!\n");
    }
    break;

  case llvm::dwarf::DW_TAG_member:
    if (DIDerivedType *DIDer = dyn_cast<DIDerivedType>(DITy)) {
      return getNameFromDIE(DIDer->getBaseType());
    } else {
      LLVM_DEBUG(dbgs() << "Panic! Hit typedef, but is not derived type!\n");
    }
    break;

  // Otherwise unknown, unexpected debuginfo
  default:
    LLVM_DEBUG(dbgs() << "Unkown Tag:" << DITy->getTag() << " for " << *DITy
                      << "\n");
    return UnknownStringRef;
    break;
  }
  LLVM_DEBUG(dbgs() << "Unkown Tag: " << DITy->getTag() << " for " << *DITy
                    << "\n");
  return UnknownStringRef;
}

bool AllocTypeTree::shouldContinueResolveType(std::string DwarfTypeName) {
  if (DwarfTypeName.compare("Ptr") == 0 ||
      DwarfTypeName.compare("Union") == 0 ||
      DwarfTypeName.compare("Array") == 0 ||
      DwarfTypeName.compare("Enum") == 0 ||
      DwarfTypeName.compare("Subroutine") == 0 ||
      DwarfTypeName.compare("Unknown") == 0 ||
      DwarfTypeName.compare("String") == 0)
    return false;
  else
    return true;
}

// TODO: we want some better form of error handling here in case our
// assumptions are not fulfilled
void AllocTypeTree::resolveLayoutInformation(LLVMContext &Ctx,
                                             DebugInfoFinder &Finder) {
  preOrderApply({[&Ctx, &Finder, this](AllocTypeTreeNode *Node) {
    // In this case, we have reached a leaf node to a raw pointer, and we
    // cannot resolve any further type information from DebugInfo
    if (!shouldContinueResolveType(Node->DwarfTypeName))
      return;

    // Special Case for STL internals:
    // Somtimes the type is held within an opaque template type
    // "__aligned_membuf". We to unwrap this to get further type information
    if (auto InternalOpt = getMembufInternal(Node->DwarfTypeName)) {
      LLVM_DEBUG(dbgs() << "Found Membuf: " << Node->DwarfTypeName << "\n");
      AllocTypeTree ATT(*InternalOpt);
      ATT.buildDwarfNames(Ctx);
      Node->DwarfTypeName = ATT.Root->DwarfTypeName;
      // deleteNode(ATT.Root);
      LLVM_DEBUG(dbgs() << "After dwarfing and unwrapping membuf: "
                        << Node->DwarfTypeName << "\n");
    }
    // LLVM_DEBUG(dbgs() << "Looking up DIE for: " << Node->DwarfTypeName <<
    // "\n");
    DIType *DITy = AllocTypeTree::lookupDebugInfo(Finder, Node->DwarfTypeName);

    if (DITy == nullptr) {
      LLVM_DEBUG(dbgs() << "Panic! We coudln't find DebugInfo for type: "
                        << Node->DwarfTypeName << "\n");
      return;
    }

    // LLVM_DEBUG(dbgs() << "Found DIType: " << *DITy << " for "
    //                   << Node->DwarfTypeName << "\n");
    Node->Size = DITy->getSizeInBits() / 8;

    DICompositeType *DICompTy = dyn_cast<DICompositeType>(DITy);
    if (DICompTy) {
      // Before we start looking at each member, we need to filter out a few
      // special cases, sometimes, the STL will have two members of a struct
      // with the same offset (essentially overlapping), which would confuse
      // our tree structure. An example of this is std::pair<x,y>, which has
      // three members: __pair_base<x,y>, x, y. In this case, __pair_base
      // shares the offset with x, but is actually a "fake" template member
      // implemented for ABI compatibility.

      // Solution: Make sure that each offset only has a single member. We
      // look at all members. If there are two members with the same offset,
      // remove the member that is from inheritance, with DW_TAG_inheritance.

      // Add all the elements with the same offset to map
      std::map<uint64_t, llvm::SmallVector<DIType *, 8>> OffsetToElements;
      for (DINode *Elem : DICompTy->getElements()) {
        if (!(Elem->getTag() == llvm::dwarf::DW_TAG_member ||
              Elem->getTag() == llvm::dwarf::DW_TAG_inheritance)) {
          continue;
        } else {
          if (DIType *DITyNode = dyn_cast<DIType>(Elem)) {
            uint64_t Offset = DITyNode->getOffsetInBits();
            auto It = OffsetToElements.find(Offset);
            if (It == OffsetToElements.end()) {
              auto V = llvm::SmallVector<DIType *, 8>();
              V.push_back(DITyNode);
              OffsetToElements.insert({Offset, V});
            } else {
              // Very inefficient here, lots of copying
              //  TODO: make this better
              auto V = It->second;
              V.push_back(DITyNode);
              OffsetToElements.insert({Offset, V});
            }
          }
        }
      }

      // Clean out all elements with Inheritance type out of the map
      for (auto &[Offset, Elements] : OffsetToElements) {
        if (Elements.size() == 1)
          continue;
        for (llvm::SmallVector<DIType *>::iterator It = Elements.begin();
             It != Elements.end();) {
          if ((*It)->getTag() == llvm::dwarf::DW_TAG_inheritance)
            Elements.erase(It);
          else
            It++;
        }
        assert(Elements.size() == 1 && "We must have removed duplicate members "
                                       "that share the same offset");
      }

      size_t ChildIdx = 0;
      for (auto &[Offset, Elements] : OffsetToElements) {

        for (DIType *DITyNode : Elements) {

          // DIType *DITyNode = dyn_cast<DIType>(Elem);
          StringRef TypeName = getNameFromDIE(DITyNode);
          // if (!DITyNode) {

          //   LLVM_DEBUG(dbgs() << "Panic!" << *Elem
          //                     << " is member, but is not a Type? \n");

          //   return;
          // }

          // // If element is Derived Type, check BaseType name
          // DIDerivedType *Der = dyn_cast<DIDerivedType>(Elem);
          // if (Der) {
          //   if (Der->getBaseType()->getTag() ==
          //       llvm::dwarf::DW_TAG_pointer_type) {
          //     // Special Case: if base type is pointer, it will have no
          //     name.
          //     // Just set Name to Ptr*

          //     TypeName =
          //         StringRef("Ptr*"); // As of Right now, we don't actually
          //         know
          //                            // what type we are pointing to
          //   } else {
          //     TypeName = Der->getBaseType()->getName();
          //     LLVM_DEBUG(dbgs() << "basetype: " << *(Der->getBaseType()));
          //   }
          // } else {
          //   TypeName = DITyNode->getName();
          // }

          // LLVM_DEBUG(dbgs() << "Resolved Member Name: " << TypeName <<
          // "\n");

          // Check if current type we are pointing to is already contained
          // in our tree.If not, add this type to tree, and set offset. If
          // it is, just set offset of existing tree node.

          if (ChildIdx < Node->Children.size() &&
              TypeName.contains(Node->getChild(ChildIdx)->DwarfTypeName)) {
            // We have a match! All we need to do is set Offset
            Node->getChild(ChildIdx)->Offset =
                Node->Offset + DITyNode->getOffsetInBits() / 8;
            // LLVM_DEBUG(dbgs()
            //            << "Matched " << TypeName << " with "
            //            << Node->getChild(ChildIdx)->DwarfTypeName << "\n");
            ChildIdx++;
          } else {

            if (TypeName.str() == "") {
              LLVM_DEBUG(
                  dbgs()
                  << "Panic! We are adding a node without name for member: "
                  << *DITyNode << "\n");
            }

            AllocTypeTreeNode *ChildNode = new AllocTypeTreeNode(
                TypeName.str(), DITyNode->getSizeInBits() / 8,
                Node->Offset + DITyNode->getOffsetInBits() / 8);
            Node->Children.insert(Node->Children.begin() + ChildIdx, ChildNode);
            ChildIdx++;

            // LLVM_DEBUG(dbgs() << "We add a new Node to the tree:"
            //                   << ChildNode->DwarfTypeName << "\n");
          }
        }
      }
      // LLVM_DEBUG(dbgs() << "\n");
    }
  }});
}
// TODO: Handle some edge cases where the alignment is not perfect between type
// and histogram
void AllocTypeTree::mergeWithHistogram(AccessCountHistogram Hist) {
  size_t CollapsedSize = Root->Size / HiSTOGRAM_GRANULARITY;
  if (Root->Size < HiSTOGRAM_GRANULARITY) {
    CollapsedSize = 1;
  }

  if (Root->Size == 0) {

    LLVM_DEBUG(dbgs() << "Panic! Resolved Type is 0 \n");
    return;
  }

  std::vector<uint64_t> CollapsedHistogram =
      std::vector<uint64_t>(CollapsedSize, 0);

  if (Hist.Size == CollapsedSize) {
    // LLVM_DEBUG(dbgs() << "Histogram Size Match!\n");
    for (size_t I = 0; I < CollapsedSize; I++) {
      CollapsedHistogram[I] = Hist.Ptr[I];
    }
    attachFieldAccessCount(CollapsedHistogram);
  } else if (Hist.Size % CollapsedSize == 0) {

    size_t NumElementsInContainer = Hist.Size / CollapsedSize;
    std::vector<uint64_t> CollapsedHistogram =
        std::vector<uint64_t>(CollapsedSize, 0);
    // LLVM_DEBUG(dbgs() << "Histogram Size match in container, for size : "
    //                   << NumElementsInContainer << "\n");
    for (size_t I = 0; I < NumElementsInContainer; I++) {
      for (size_t J = 0; J < CollapsedSize; J++) {
        CollapsedHistogram[J] += Hist.Ptr[I * CollapsedSize + J];
      }
    }
    attachFieldAccessCount(CollapsedHistogram);
  } else {
    LLVM_DEBUG(dbgs() << "Panic! Size mismatch for Hist Size " << Hist.Size
                      << " and resolved type size: " << CollapsedSize << "\n");
  }
}

void AllocTypeTree::attachFieldAccessCount(std::vector<uint64_t> &Histogram) {
  Root->FieldAccessCount =
      sumHistogram(Histogram, 0, Histogram.size() * HiSTOGRAM_GRANULARITY);
  Root->attachFieldAccessCount(Histogram,
                               Histogram.size() * HiSTOGRAM_GRANULARITY);
}

void AllocTypeTree::AllocTypeTreeNode::attachFieldAccessCount(
    std::vector<uint64_t> &Histogram, uint32_t MaxOffset) {
  for (size_t I = 0; I < Children.size(); I++) {
    AllocTypeTreeNode *Child = getChild(I);
    if (I == Children.size() - 1) {
      // Last Node
      // sum Histogram fom this child offset to end
      Child->FieldAccessCount =
          sumHistogram(Histogram, Child->Offset, MaxOffset);
      Child->attachFieldAccessCount(Histogram, MaxOffset);
    } else {

      // sum Histogram from this child offset to next offset
      Child->FieldAccessCount =
          sumHistogram(Histogram, Child->Offset, getChild(I + 1)->Offset);

      Child->attachFieldAccessCount(Histogram, getChild(I + 1)->Offset);
    }
  }
}

uint64_t AllocTypeTree::sumHistogram(std::vector<uint64_t> &Histogram,
                                     uint32_t From, uint32_t To) {

  size_t FromIdx = From / HiSTOGRAM_GRANULARITY;
  size_t ToIdx;
  if (To % HiSTOGRAM_GRANULARITY > 0)
    ToIdx = To / HiSTOGRAM_GRANULARITY + 1;
  else
    ToIdx = To / HiSTOGRAM_GRANULARITY;
  uint64_t Acc = 0;
  for (size_t I = FromIdx; I < ToIdx; I++) {
    Acc += Histogram[I];
  }

  return Acc;
}

} // namespace memprof
} // namespace llvm