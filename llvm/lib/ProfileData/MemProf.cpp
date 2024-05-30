#include "llvm/ProfileData/MemProf.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/HashBuilder.h"

namespace llvm {
namespace memprof {
static size_t serializedSizeV0(const IndexedAllocationInfo &IAI) {
  size_t Size = 0;
  // The number of frames to serialize.
  Size += sizeof(uint64_t);
  // The callstack frame ids.
  Size += sizeof(FrameId) * IAI.CallStack.size();
  // The size of the payload.
  Size += PortableMemInfoBlock::serializedSize();
  return Size;
}

static size_t serializedSizeV2(const IndexedAllocationInfo &IAI) {
  size_t Size = 0;
  // The CallStackId
  Size += sizeof(CallStackId);
  // The size of the payload.
  Size += PortableMemInfoBlock::serializedSize();
  return Size;
}

static size_t serializedSizeV0(const AccessCountHistogram &H) {
  // Num Entries in Histogram + 1 entry for size
  return (H.Size + 1) * sizeof(uint64_t);
}

size_t AccessCountHistogram::serializedSize(IndexedVersion Version) const {
  switch (Version) {
  case Version0:
  case Version1:
  case Version2:
    return serializedSizeV0(*this);
  }
  llvm_unreachable("unsupported MemProf version");
}

size_t IndexedAllocationInfo::serializedSize(IndexedVersion Version) const {
  switch (Version) {
  case Version0:
  case Version1:
    return serializedSizeV0(*this);
  case Version2:
    return serializedSizeV2(*this);
  }
  llvm_unreachable("unsupported MemProf version");
}

static size_t serializedSizeV0(const IndexedMemProfRecord &Record) {
  size_t Result = sizeof(GlobalValue::GUID);
  for (const IndexedAllocationInfo &N : Record.AllocSites)
    Result += N.serializedSize(Version0);

  // The number of callsites we have information for.
  Result += sizeof(uint64_t);
  for (const auto &Frames : Record.CallSites) {
    // The number of frame ids to serialize.
    Result += sizeof(uint64_t);
    Result += Frames.size() * sizeof(FrameId);
  }

  Result += sizeof(uint64_t);
  for (const auto &Histogram : Record.AccessCountHistograms)
    Result += Histogram.serializedSize(Version0);

  return Result;
}

static size_t serializedSizeV2(const IndexedMemProfRecord &Record) {
  size_t Result = sizeof(GlobalValue::GUID);
  for (const IndexedAllocationInfo &N : Record.AllocSites)
    Result += N.serializedSize(Version2);

  // The number of callsites we have information for.
  Result += sizeof(uint64_t);
  // The CallStackId
  Result += Record.CallSiteIds.size() * sizeof(CallStackId);
  return Result;
}

size_t IndexedMemProfRecord::serializedSize(IndexedVersion Version) const {
  switch (Version) {
  case Version0:
  case Version1:
    return serializedSizeV0(*this);
  case Version2:
    return serializedSizeV2(*this);
  }
  llvm_unreachable("unsupported MemProf version");
}

static void serializeV0(const IndexedMemProfRecord &Record,
                        const MemProfSchema &Schema, raw_ostream &OS) {
  using namespace support;

  endian::Writer LE(OS, llvm::endianness::little);

  LE.write<uint64_t>(Record.AllocSites.size());
  for (const IndexedAllocationInfo &N : Record.AllocSites) {
    LE.write<uint64_t>(N.CallStack.size());
    for (const FrameId &Id : N.CallStack)
      LE.write<FrameId>(Id);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(Record.CallSites.size());
  for (const auto &Frames : Record.CallSites) {
    LE.write<uint64_t>(Frames.size());
    for (const FrameId &Id : Frames)
      LE.write<FrameId>(Id);
  }

  // Histogram
  LE.write<uint64_t>(Record.AccessCountHistograms.size());
  for (const auto &Histogram : Record.AccessCountHistograms) {
    LE.write<uint64_t>(Histogram.Size);
    for (uint64_t I = 0; I < Histogram.Size; I++) {
      LE.write<uint64_t>(Histogram.Ptr[I]);
    }
    // free(Histogram.Ptr); //TODO: FIx this, we have double free here
  }
}

static void serializeV2(const IndexedMemProfRecord &Record,
                        const MemProfSchema &Schema, raw_ostream &OS) {
  using namespace support;

  endian::Writer LE(OS, llvm::endianness::little);

  LE.write<uint64_t>(Record.AllocSites.size());
  for (const IndexedAllocationInfo &N : Record.AllocSites) {
    LE.write<CallStackId>(N.CSId);
    N.Info.serialize(Schema, OS);
  }

  // Related contexts.
  LE.write<uint64_t>(Record.CallSiteIds.size());
  for (const auto &CSId : Record.CallSiteIds)
    LE.write<CallStackId>(CSId);
}

void IndexedMemProfRecord::serialize(const MemProfSchema &Schema,
                                     raw_ostream &OS, IndexedVersion Version) {
  switch (Version) {
  case Version0:
  case Version1:
    serializeV0(*this, Schema, OS);
    return;
  case Version2:
    serializeV2(*this, Schema, OS);
    return;
  }
  llvm_unreachable("unsupported MemProf version");
}

static IndexedMemProfRecord deserializeV0(const MemProfSchema &Schema,
                                          const unsigned char *Ptr) {
  using namespace support;

  IndexedMemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  for (uint64_t I = 0; I < NumNodes; I++) {
    IndexedAllocationInfo Node;
    const uint64_t NumFrames =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    for (uint64_t J = 0; J < NumFrames; J++) {
      const FrameId Id =
          endian::readNext<FrameId, llvm::endianness::little, unaligned>(Ptr);
      Node.CallStack.push_back(Id);
    }
    Node.CSId = hashCallStack(Node.CallStack);
    Node.Info.deserialize(Schema, Ptr);
    Ptr += PortableMemInfoBlock::serializedSize();
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    const uint64_t NumFrames =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    llvm::SmallVector<FrameId> Frames;
    Frames.reserve(NumFrames);
    for (uint64_t K = 0; K < NumFrames; K++) {
      const FrameId Id =
          endian::readNext<FrameId, llvm::endianness::little, unaligned>(Ptr);
      Frames.push_back(Id);
    }
    Record.CallSites.push_back(Frames);
    Record.CallSiteIds.push_back(hashCallStack(Frames));
  }

  // Read AccessCountHistograms
  const uint64_t NumHistograms =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  for (uint64_t L = 0; L < NumHistograms; L++) {
    AccessCountHistogram Histogram;
    Histogram.Size =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    Histogram.Ptr = (uint64_t *)malloc(Histogram.Size * sizeof(uint64_t));
    for (uint64_t M = 0; M < Histogram.Size; M++) {
      Histogram.Ptr[M] =
          endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    }
    Record.AccessCountHistograms.push_back(Histogram);
  }
  return Record;
}

static IndexedMemProfRecord deserializeV2(const MemProfSchema &Schema,
                                          const unsigned char *Ptr) {
  using namespace support;

  IndexedMemProfRecord Record;

  // Read the meminfo nodes.
  const uint64_t NumNodes =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  for (uint64_t I = 0; I < NumNodes; I++) {
    IndexedAllocationInfo Node;
    Node.CSId =
        endian::readNext<CallStackId, llvm::endianness::little, unaligned>(Ptr);
    Node.Info.deserialize(Schema, Ptr);
    Ptr += PortableMemInfoBlock::serializedSize();
    Record.AllocSites.push_back(Node);
  }

  // Read the callsite information.
  const uint64_t NumCtxs =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  for (uint64_t J = 0; J < NumCtxs; J++) {
    CallStackId CSId =
        endian::readNext<CallStackId, llvm::endianness::little, unaligned>(Ptr);
    Record.CallSiteIds.push_back(CSId);
  }

  return Record;
}

IndexedMemProfRecord
IndexedMemProfRecord::deserialize(const MemProfSchema &Schema,
                                  const unsigned char *Ptr,
                                  IndexedVersion Version) {
  switch (Version) {
  case Version0:
  case Version1:
    return deserializeV0(Schema, Ptr);
  case Version2:
    return deserializeV2(Schema, Ptr);
  }
  llvm_unreachable("unsupported MemProf version");
}

GlobalValue::GUID IndexedMemProfRecord::getGUID(const StringRef FunctionName) {
  // Canonicalize the function name to drop suffixes such as ".llvm.". Note
  // we do not drop any ".__uniq." suffixes, as getCanonicalFnName does not drop
  // those by default. This is by design to differentiate internal linkage
  // functions during matching. By dropping the other suffixes we can then match
  // functions in the profile use phase prior to their addition. Note that this
  // applies to both instrumented and sampled function names.
  StringRef CanonicalName =
      sampleprof::FunctionSamples::getCanonicalFnName(FunctionName);

  // We use the function guid which we expect to be a uint64_t. At
  // this time, it is the lower 64 bits of the md5 of the canonical
  // function name.
  return Function::getGUID(CanonicalName);
}

Expected<MemProfSchema> readMemProfSchema(const unsigned char *&Buffer) {
  using namespace support;

  const unsigned char *Ptr = Buffer;
  const uint64_t NumSchemaIds =
      endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
  if (NumSchemaIds > static_cast<uint64_t>(Meta::Size)) {
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "memprof schema invalid");
  }

  MemProfSchema Result;
  for (size_t I = 0; I < NumSchemaIds; I++) {
    const uint64_t Tag =
        endian::readNext<uint64_t, llvm::endianness::little, unaligned>(Ptr);
    if (Tag >= static_cast<uint64_t>(Meta::Size)) {
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "memprof schema invalid");
    }
    Result.push_back(static_cast<Meta>(Tag));
  }
  // Advace the buffer to one past the schema if we succeeded.
  Buffer = Ptr;
  return Result;
}

CallStackId hashCallStack(ArrayRef<FrameId> CS) {
  llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
      HashBuilder;
  for (FrameId F : CS)
    HashBuilder.add(F);
  llvm::BLAKE3Result<8> Hash = HashBuilder.final();
  CallStackId CSId;
  std::memcpy(&CSId, Hash.data(), sizeof(Hash));
  return CSId;
}

void verifyIndexedMemProfRecord(const IndexedMemProfRecord &Record) {
  for (const auto &AS : Record.AllocSites) {
    assert(AS.CSId == hashCallStack(AS.CallStack));
    (void)AS;
  }
}

void verifyFunctionProfileData(
    const llvm::MapVector<GlobalValue::GUID, IndexedMemProfRecord>
        &FunctionProfileData) {
  for (const auto &[GUID, Record] : FunctionProfileData) {
    (void)GUID;
    verifyIndexedMemProfRecord(Record);
  }
}

} // namespace memprof
} // namespace llvm
