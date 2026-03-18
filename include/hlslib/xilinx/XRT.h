/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#define HLSLIB_XILINX_XRT_H

#ifndef HLSLIB_XILINX
#define HLSLIB_XILINX
#endif

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HLSLIB_SIMULATE_OPENCL
#if __has_include(<xrt/xrt_device.h>)
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_uuid.h>
#else
#include <experimental/xrt_device.h>
#include <experimental/xrt_kernel.h>
#include <experimental/xrt_bo.h>
#include <experimental/xrt_uuid.h>
#endif
#endif

namespace hlslib {

namespace ocl {

//#############################################################################
// Enumerations and types
//#############################################################################

enum class Access { read, write, readWrite };

enum class MemoryBank { unspecified, bank0, bank1, bank2, bank3 };

enum class StorageType { DDR, HBM };

template <typename T>
struct _SimulationOnly {
  _SimulationOnly(T _simulation) : simulation(_simulation) {}
  T simulation;
};
template <typename T>
auto SimulationOnly(T &&simulation) {
  return _SimulationOnly<typename std::conditional<
      std::is_rvalue_reference<T &&>::value,
      typename std::remove_reference<T>::type, T &&>::type>(
      std::forward<T>(simulation));
}

#ifndef HLSLIB_SIMULATE_OPENCL
class Event {
 public:
  Event() = default;

  Event(xrt::run run) : run_(std::move(run)) {}

  Event(Event &&) = default;
  Event(Event const &) = default;
  Event &operator=(Event &&) = default;
  Event &operator=(Event const &) = default;

  void wait() const {
    if (run_) {
      run_.wait();
    }
  }

 private:
  mutable xrt::run run_;
};
#else
class Event {
 public:
  Event() = default;

  Event(std::function<void(void)> const &f) {
    future_ = std::async(std::launch::async, f).share();
  }

  Event(Event &&) = default;
  Event(Event const &) = default;
  Event &operator=(Event &&) = default;
  Event &operator=(Event const &) = default;

  void wait() const {
    if (future_.valid()) {
      future_.wait();
    }
  }

 private:
  std::shared_future<void> future_;
};
#endif

//#############################################################################
// Exceptions
//#############################################################################

class ConfigurationError : public std::logic_error {
 public:
  ConfigurationError(std::string const &message) : std::logic_error(message) {}
  ConfigurationError(char const *const message) : std::logic_error(message) {}
};

class RuntimeError : public std::runtime_error {
 public:
  RuntimeError(std::string const &message) : std::runtime_error(message) {}
  RuntimeError(char const *const message) : std::runtime_error(message) {}
};

//#############################################################################
// Internal helpers
//#############################################################################

namespace {

template <typename IteratorType>
constexpr bool IsRandomAccess() {
  return std::is_base_of<
      std::random_access_iterator_tag,
      typename std::iterator_traits<IteratorType>::iterator_category>::value;
}

template <typename IteratorType, typename T>
constexpr bool IsIteratorOfType() {
  return std::is_same<typename std::iterator_traits<IteratorType>::value_type,
                      T>::value;
}

template <typename IntCollection,
          typename ICIt = decltype(*begin(std::declval<IntCollection>())),
          typename ICIty = std::decay_t<ICIt>>
constexpr bool IsIntCollection() {
  return std::is_convertible<ICIty, int>();
}

void ThrowConfigurationError(std::string const &message) {
#ifndef HLSLIB_DISABLE_EXCEPTIONS
  throw ConfigurationError(message);
#else
  std::cerr << "XRT [Configuration Error]: " << message << std::endl;
#endif
}

void ThrowRuntimeError(std::string const &message) {
#ifndef HLSLIB_DISABLE_EXCEPTIONS
  throw RuntimeError(message);
#else
  std::cerr << "XRT [Runtime Error]: " << message << std::endl;
#endif
}

int MemoryBankToGroup(MemoryBank bank) {
  switch (bank) {
    case MemoryBank::bank0: return 0;
    case MemoryBank::bank1: return 1;
    case MemoryBank::bank2: return 2;
    case MemoryBank::bank3: return 3;
    case MemoryBank::unspecified: return 0;
  }
  return 0;
}

int StorageToGroup(StorageType storage, int bankIndex) {
  switch (storage) {
    case StorageType::DDR:
      return bankIndex >= 0 ? bankIndex : 0;
    case StorageType::HBM:
      return bankIndex;
  }
  return 0;
}

}  // End anonymous namespace

// Forward declarations
template <typename, Access>
class Buffer;
class Kernel;
class Program;

//#############################################################################
// Context
//#############################################################################

class Context {
 public:
  inline Context(int index) {
#ifndef HLSLIB_SIMULATE_OPENCL
    device_ = xrt::device(index);
#endif
  }

  inline Context() : Context(0) {}

  inline Context(std::string const &deviceName) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Iterate devices to find by name
    for (unsigned i = 0;; ++i) {
      try {
        xrt::device dev(i);
        auto name = dev.get_info<xrt::info::device::name>();
        if (name.find(deviceName) != std::string::npos) {
          device_ = std::move(dev);
          return;
        }
      } catch (...) {
        break;
      }
    }
    ThrowConfigurationError("Device \"" + deviceName + "\" not found.");
#endif
  }

  inline Context(std::string const &, std::string const &deviceName)
      : Context(deviceName) {}

  inline Context(std::string const &, int index) : Context(index) {}

  inline Context(Context const &) = delete;
  inline Context(Context &&) = default;
  inline Context &operator=(Context const &) = delete;
  inline Context &operator=(Context &&) = default;
  inline ~Context() = default;

  inline Program MakeProgram(std::string const &path);

  inline std::string DeviceName() const {
#ifndef HLSLIB_SIMULATE_OPENCL
    return device_.get_info<xrt::info::device::name>();
#else
    return "Simulation";
#endif
  }

  inline Program CurrentlyLoadedProgram() const;

  template <typename T, Access access, typename... Ts>
  Buffer<T, access> MakeBuffer(Ts &&... args);

#ifndef HLSLIB_SIMULATE_OPENCL
  xrt::device &device() { return device_; }
  xrt::device const &device() const { return device_; }
  xrt::uuid const &uuid() const { return uuid_; }
#endif

 protected:
  friend Program;
  friend Kernel;
  template <typename U, Access access>
  friend class Buffer;

  std::mutex &memcopyMutex() { return memcopyMutex_; }
  std::mutex &reprogramMutex() { return reprogramMutex_; }

 private:
#ifndef HLSLIB_SIMULATE_OPENCL
  xrt::device device_;
  xrt::uuid uuid_;
#endif
  std::string loadedPath_;
  std::mutex memcopyMutex_;
  std::mutex reprogramMutex_;
};

//#############################################################################
// Buffer
//#############################################################################

template <typename T, Access access>
class Buffer {
 public:
  Buffer() : context_(nullptr), nElements_(0) {}

  Buffer(Buffer<T, access> const &other) = delete;

  Buffer(Buffer<T, access> &&other) : Buffer() {
    swap(*this, other);
  }

  // Allocate and copy to device (DDR with bank)
  template <
      typename IteratorType,
      typename = typename std::enable_if<
          !std::is_convertible<IteratorType, int>()>::type,
      typename = typename std::enable_if<IsIteratorOfType<IteratorType, T>() &&
                                         IsRandomAccess<IteratorType>()>::type>
  Buffer(Context &context, MemoryBank memoryBank, IteratorType begin,
         IteratorType end)
      : context_(&context), nElements_(std::distance(begin, end)) {
#ifndef HLSLIB_SIMULATE_OPENCL
    hostStaging_.assign(reinterpret_cast<const char *>(&(*begin)),
                        reinterpret_cast<const char *>(&(*begin)) +
                            sizeof(T) * nElements_);
#else
    devicePtr_ = std::make_unique<T[]>(nElements_);
    std::copy(begin, end, devicePtr_.get());
#endif
  }

  template <typename IteratorType, typename = typename std::enable_if<
                                       IsIteratorOfType<IteratorType, T>() &&
                                       IsRandomAccess<IteratorType>()>::type>
  Buffer(Context &context, IteratorType begin, IteratorType end)
      : Buffer(context, MemoryBank::unspecified, begin, end) {}

  // Allocate without transfer (DDR with bank)
  Buffer(Context &context, MemoryBank memoryBank, size_t nElements)
      : context_(&context), nElements_(nElements) {
#ifndef HLSLIB_SIMULATE_OPENCL
    // Deferred: bo_ created when passed to kernel
#else
    devicePtr_ = std::make_unique<T[]>(nElements_);
#endif
  }

  Buffer(Context &context, size_t nElements)
      : Buffer(context, MemoryBank::unspecified, nElements) {}

  // Allocate DDR/HBM without transfer — deferred until passed to kernel
  Buffer(Context &context, StorageType storageType, int bankIndex,
         size_t nElements)
      : context_(&context), nElements_(nElements) {
#ifndef HLSLIB_SIMULATE_OPENCL
    hintGroup_ = StorageToGroup(storageType, bankIndex);
#else
    devicePtr_ = std::make_unique<T[]>(nElements_);
#endif
  }

  // Allocate DDR/HBM and copy to device — deferred until passed to kernel
  template <typename IteratorType, typename = typename std::enable_if<
                                       IsIteratorOfType<IteratorType, T>() &&
                                       IsRandomAccess<IteratorType>()>::type>
  Buffer(Context &context, StorageType storageType, int bankIndex,
         IteratorType begin, IteratorType end)
      : context_(&context), nElements_(std::distance(begin, end)) {
#ifndef HLSLIB_SIMULATE_OPENCL
    hintGroup_ = StorageToGroup(storageType, bankIndex);
    hostStaging_.assign(reinterpret_cast<const char *>(&(*begin)),
                        reinterpret_cast<const char *>(&(*begin)) +
                            sizeof(T) * nElements_);
#else
    devicePtr_ = std::make_unique<T[]>(nElements_);
    std::copy(begin, end, devicePtr_.get());
#endif
  }

  friend void swap(Buffer<T, access> &first, Buffer<T, access> &second) {
    std::swap(first.context_, second.context_);
#ifndef HLSLIB_SIMULATE_OPENCL
    std::swap(first.bo_, second.bo_);
    std::swap(first.allocated_, second.allocated_);
    std::swap(first.hintGroup_, second.hintGroup_);
    std::swap(first.hostStaging_, second.hostStaging_);
#else
    std::swap(first.devicePtr_, second.devicePtr_);
#endif
    std::swap(first.nElements_, second.nElements_);
  }

  Buffer<T, access> &operator=(Buffer<T, access> const &other) = delete;

  Buffer<T, access> &operator=(Buffer<T, access> &&other) {
    swap(*this, other);
    return *this;
  }

  ~Buffer() = default;

  template <typename DataIterator, typename = typename std::enable_if<
                                       IsIteratorOfType<DataIterator, T>() &&
                                       IsRandomAccess<DataIterator>()>::type>
  void CopyFromHost(int deviceOffset, int numElements, DataIterator source) {
#ifndef HLSLIB_SIMULATE_OPENCL
    if (!allocated_) {
      // Stage to host buffer; will be flushed on Allocate()
      size_t required = sizeof(T) * (deviceOffset + numElements);
      if (hostStaging_.size() < required) {
        hostStaging_.resize(required, 0);
      }
      std::memcpy(hostStaging_.data() + sizeof(T) * deviceOffset,
                  &(*source), sizeof(T) * numElements);
    } else {
      bo_.write(&(*source), sizeof(T) * numElements, sizeof(T) * deviceOffset);
      bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE, sizeof(T) * numElements,
               sizeof(T) * deviceOffset);
    }
#else
    std::copy(source, source + numElements, devicePtr_.get() + deviceOffset);
#endif
  }

  template <typename DataIterator, typename = typename std::enable_if<
                                       IsIteratorOfType<DataIterator, T>() &&
                                       IsRandomAccess<DataIterator>()>::type>
  void CopyFromHost(DataIterator source) {
    CopyFromHost(0, nElements_, source);
  }

  // Overloads with event dependencies (events are waited on, then ignored)
  template <
      typename DataIterator, typename EventIterator = Event *,
      typename = typename std::enable_if<IsIteratorOfType<DataIterator, T>() &&
                                         IsRandomAccess<DataIterator>()>::type,
      typename =
          typename std::enable_if<IsIteratorOfType<EventIterator, Event>() &&
                                  IsRandomAccess<EventIterator>()>::type>
  void CopyFromHost(int deviceOffset, int numElements, DataIterator source,
                    EventIterator eventsBegin, EventIterator eventsEnd) {
    for (auto it = eventsBegin; it != eventsEnd; ++it) {
      it->wait();
    }
    CopyFromHost(deviceOffset, numElements, source);
  }

  template <
      typename DataIterator, typename EventIterator = Event *,
      typename = typename std::enable_if<IsIteratorOfType<DataIterator, T>() &&
                                         IsRandomAccess<DataIterator>()>::type,
      typename =
          typename std::enable_if<IsIteratorOfType<EventIterator, Event>() &&
                                  IsRandomAccess<EventIterator>()>::type>
  void CopyFromHost(DataIterator source, EventIterator eventBegin,
                    EventIterator eventEnd) {
    return CopyFromHost(0, nElements_, source, eventBegin, eventEnd);
  }

  template <typename DataIterator, typename = typename std::enable_if<
                                       IsIteratorOfType<DataIterator, T>() &&
                                       IsRandomAccess<DataIterator>()>::type>
  void CopyToHost(size_t deviceOffset, size_t numElements,
                  DataIterator target) {
#ifndef HLSLIB_SIMULATE_OPENCL
    EnsureAllocated();
    bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE, sizeof(T) * numElements,
             sizeof(T) * deviceOffset);
    bo_.read(&(*target), sizeof(T) * numElements, sizeof(T) * deviceOffset);
#else
    std::copy(devicePtr_.get() + deviceOffset,
              devicePtr_.get() + deviceOffset + numElements, target);
#endif
  }

  template <typename DataIterator, typename = typename std::enable_if<
                                       IsIteratorOfType<DataIterator, T>() &&
                                       IsRandomAccess<DataIterator>()>::type>
  void CopyToHost(DataIterator target) {
    CopyToHost(0, nElements_, target);
  }

  template <
      typename DataIterator, typename EventIterator = Event *,
      typename = typename std::enable_if<IsIteratorOfType<DataIterator, T>() &&
                                         IsRandomAccess<DataIterator>()>::type,
      typename =
          typename std::enable_if<IsIteratorOfType<EventIterator, Event>() &&
                                  IsRandomAccess<EventIterator>()>::type>
  void CopyToHost(size_t deviceOffset, size_t numElements, DataIterator target,
                  EventIterator eventsBegin, EventIterator eventsEnd) {
    for (auto it = eventsBegin; it != eventsEnd; ++it) {
      it->wait();
    }
    CopyToHost(deviceOffset, numElements, target);
  }

  template <
      typename DataIterator, typename EventIterator = Event *,
      typename = typename std::enable_if<IsIteratorOfType<DataIterator, T>() &&
                                         IsRandomAccess<DataIterator>()>::type,
      typename =
          typename std::enable_if<IsIteratorOfType<EventIterator, Event>() &&
                                  IsRandomAccess<EventIterator>()>::type>
  void CopyToHost(DataIterator target, EventIterator eventsBegin,
                  EventIterator eventsEnd) {
    return CopyToHost(0, nElements_, target, eventsBegin, eventsEnd);
  }

  template <Access accessType>
  void CopyToDevice(size_t offsetSource, size_t numElements,
                    Buffer<T, accessType> &other, size_t offsetDestination) {
#ifndef HLSLIB_SIMULATE_OPENCL
    EnsureAllocated();
    other.EnsureAllocated();
    other.bo_.copy(bo_, sizeof(T) * numElements, sizeof(T) * offsetSource,
                   sizeof(T) * offsetDestination);
    other.bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
#else
    std::copy(devicePtr_.get() + offsetSource,
              devicePtr_.get() + offsetSource + numElements,
              other.devicePtr_.get() + offsetDestination);
#endif
  }

  template <Access accessType>
  void CopyToDevice(size_t offsetSource, size_t numElements,
                    Buffer<T, accessType> &other) {
    CopyToDevice(offsetSource, numElements, other, 0);
  }

  template <Access accessType>
  void CopyToDevice(Buffer<T, accessType> &other) {
    if (other.nElements() != nElements_) {
      ThrowRuntimeError(
          "Device to device copy issued for buffers of different size.");
    }
    CopyToDevice(0, nElements_, other, 0);
  }

  template <Access accessType, typename EventIterator = Event *,
            typename = typename std::enable_if<
                IsIteratorOfType<EventIterator, Event>() &&
                IsRandomAccess<EventIterator>()>::type>
  void CopyToDevice(size_t offsetSource, size_t numElements,
                    Buffer<T, accessType> &other, size_t offsetDestination,
                    EventIterator eventsBegin, EventIterator eventsEnd) {
    for (auto it = eventsBegin; it != eventsEnd; ++it) {
      it->wait();
    }
    CopyToDevice(offsetSource, numElements, other, offsetDestination);
  }

  // 3D block copies using mapped pointer + strided loops
  template <
      typename IntCollection, typename IteratorType,
      typename = typename std::enable_if<IsIteratorOfType<IteratorType, T>() &&
                                         IsRandomAccess<IteratorType>()>::type,
      typename =
          typename std::enable_if<IsIntCollection<IntCollection>()>::type>
  void CopyBlockFromHost(const IntCollection &hostBlockOffset,
                         const IntCollection &deviceBlockOffset,
                         const IntCollection &copyBlockSize,
                         const IntCollection &hostBlockSize,
                         const IntCollection &deviceBlockSize,
                         IteratorType source) {
#ifndef HLSLIB_SIMULATE_OPENCL
    EnsureAllocated();
    T *mapped = bo_.template map<T *>();
    CopyMemoryBlockSimulate(hostBlockOffset, deviceBlockOffset, copyBlockSize,
                            hostBlockSize, deviceBlockSize, source, mapped);
    bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
#else
    CopyMemoryBlockSimulate(hostBlockOffset, deviceBlockOffset, copyBlockSize,
                            hostBlockSize, deviceBlockSize, source,
                            devicePtr_.get());
#endif
  }

  template <
      typename IteratorType, typename IntCollection,
      typename = typename std::enable_if<IsIteratorOfType<IteratorType, T>() &&
                                         IsRandomAccess<IteratorType>()>::type,
      typename =
          typename std::enable_if<IsIntCollection<IntCollection>()>::type>
  void CopyBlockToHost(const IntCollection &hostBlockOffset,
                       const IntCollection &deviceBlockOffset,
                       const IntCollection &copyBlockSize,
                       const IntCollection &hostBlockSize,
                       const IntCollection &deviceBlockSize,
                       IteratorType target) {
#ifndef HLSLIB_SIMULATE_OPENCL
    EnsureAllocated();
    bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    T const *mapped = bo_.template map<T *>();
    CopyMemoryBlockSimulate(deviceBlockOffset, hostBlockOffset, copyBlockSize,
                            deviceBlockSize, hostBlockSize, mapped, target);
#else
    CopyMemoryBlockSimulate(deviceBlockOffset, hostBlockOffset, copyBlockSize,
                            deviceBlockSize, hostBlockSize, devicePtr_.get(),
                            target);
#endif
  }

  template <Access accessType, typename IntCollection,
            typename =
                typename std::enable_if<IsIntCollection<IntCollection>()>::type>
  void CopyBlockToDevice(const IntCollection &sourceBlockOffset,
                         const IntCollection &destBlockOffset,
                         const IntCollection &copyBlockSize,
                         const IntCollection &sourceBlockSize,
                         const IntCollection &destBlockSize,
                         Buffer<T, accessType> &other) {
#ifndef HLSLIB_SIMULATE_OPENCL
    EnsureAllocated();
    other.EnsureAllocated();
    bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    T const *srcMapped = bo_.template map<T *>();
    T *dstMapped = other.bo_.template map<T *>();
    CopyMemoryBlockSimulate(sourceBlockOffset, destBlockOffset, copyBlockSize,
                            sourceBlockSize, destBlockSize, srcMapped,
                            dstMapped);
    other.bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
#else
    CopyMemoryBlockSimulate(sourceBlockOffset, destBlockOffset, copyBlockSize,
                            sourceBlockSize, destBlockSize, devicePtr_.get(),
                            other.devicePtr_.get());
#endif
  }

#ifndef HLSLIB_SIMULATE_OPENCL
  xrt::bo const &devicePtr() const { return bo_; }
  xrt::bo &devicePtr() { return bo_; }
#else
  T const *devicePtr() const { return devicePtr_.get(); }
  T *devicePtr() { return devicePtr_.get(); }
#endif

  size_t nElements() const { return nElements_; }

 private:
  template <
      typename IteratorType1, typename IteratorType2>
  static void CopyMemoryBlockSimulate(
      const std::array<size_t, 3> blockOffsetSource,
      const std::array<size_t, 3> blockOffsetDest,
      const std::array<size_t, 3> copyBlockSize,
      const std::array<size_t, 3> blockSizeSource,
      const std::array<size_t, 3> blockSizeDest,
      const IteratorType1 source, const IteratorType2 dst) {
    size_t sourceSliceJmp =
        blockSizeSource[0] - copyBlockSize[0] +
        (blockSizeSource[1] - copyBlockSize[1]) * blockSizeSource[0];
    size_t destSliceJmp =
        blockSizeDest[0] - copyBlockSize[0] +
        (blockSizeDest[1] - copyBlockSize[1]) * blockSizeDest[0];
    size_t srcindex =
        blockOffsetSource[0] + blockOffsetSource[1] * blockSizeSource[0] +
        blockOffsetSource[2] * blockSizeSource[1] * blockSizeSource[0];
    size_t dstindex = blockOffsetDest[0] +
                      blockOffsetDest[1] * blockSizeDest[0] +
                      blockOffsetDest[2] * blockSizeDest[1] * blockSizeDest[0];
    for (size_t sliceCounter = 0; sliceCounter < copyBlockSize[2];
         sliceCounter++) {
      size_t nextaddsource = 0;
      size_t nextadddest = 0;
      for (size_t rowCounter = 0; rowCounter < copyBlockSize[1]; rowCounter++) {
        srcindex += nextaddsource;
        dstindex += nextadddest;
        std::copy(source + srcindex, source + srcindex + copyBlockSize[0],
                  dst + dstindex);
        nextaddsource = blockSizeSource[0];
        nextadddest = blockSizeDest[0];
      }
      srcindex += sourceSliceJmp + copyBlockSize[0];
      dstindex += destSliceJmp + copyBlockSize[0];
    }
  }

  Context *context_;
#ifndef HLSLIB_SIMULATE_OPENCL
  xrt::bo bo_;
  bool allocated_ = false;
  int hintGroup_ = -1;
  std::vector<char> hostStaging_;

  void EnsureAllocated() {
    if (!allocated_) {
      if (hintGroup_ >= 0) {
        Allocate(hintGroup_);
      } else {
        ThrowRuntimeError(
            "Buffer not yet allocated. Pass it to a kernel first, or call "
            "Allocate() with the correct memory group.");
      }
    }
  }

 public:
  void Allocate(int memoryGroup) {
    if (allocated_) return;
    bo_ = xrt::bo(context_->device(), sizeof(T) * nElements_,
                   xrt::bo::flags::normal, memoryGroup);
    allocated_ = true;
    if (!hostStaging_.empty()) {
      bo_.write(hostStaging_.data(), hostStaging_.size(), 0);
      bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      hostStaging_.clear();
    }
  }

  bool IsAllocated() const { return allocated_; }

 private:
#else
  std::unique_ptr<T[]> devicePtr_{};
#endif
  size_t nElements_;
};

//#############################################################################
// Program
//#############################################################################

class Program {
 public:
  Program() = delete;
  Program(Program const &) = default;
  Program(Program &&) = default;
  ~Program() = default;

  inline Context &context() { return context_; }
  inline Context const &context() const { return context_; }
  inline std::string const &path() const { return path_; }

  template <typename... Ts>
  Kernel MakeKernel(std::string const &kernelName, Ts &&... args);

  template <class F, typename... Ts>
  Kernel MakeKernel(F &&hostFunction, std::string const &kernelName,
                    Ts &&... args);

 protected:
  friend Context;

  inline Program(Context &context, std::string const &path)
      : context_(context), path_(path) {}

 private:
  Context &context_;
  std::string path_;
};

//#############################################################################
// Kernel
//#############################################################################

class Kernel {
 private:
  template <typename T, Access access>
  void SetKernelArguments(size_t index, Buffer<T, access> &arg) {
#ifndef HLSLIB_SIMULATE_OPENCL
    if (!arg.IsAllocated()) {
      arg.Allocate(kernel_.group_id(index));
    }
    run_.set_arg(index, arg.devicePtr());
#endif
  }

  template <typename T>
  void SetKernelArguments(size_t index, _SimulationOnly<T> const &) {}

  template <typename T>
  void SetKernelArguments(size_t index, _SimulationOnly<T> &&) {}

  template <typename T>
  void SetKernelArguments(size_t index, T &&arg) {
#ifndef HLSLIB_SIMULATE_OPENCL
    run_.set_arg(index, arg);
#endif
  }

  void SetKernelArguments(size_t) {}
  void SetKernelArguments() {}

  template <typename T, typename... Ts>
  void SetKernelArguments(size_t index, T &&arg, Ts &&... args) {
    SetKernelArguments(index, std::forward<T>(arg));
    SetKernelArguments(index + 1, std::forward<Ts>(args)...);
  }

  template <typename T, Access access>
  static auto UnpackPointers(Buffer<T, access> &buffer) {
#ifndef HLSLIB_SIMULATE_OPENCL
    T *ptr = nullptr;
    return ptr;
#else
    return buffer.devicePtr();
#endif
  }

  template <typename T>
  static typename std::conditional<
      std::is_reference<T>::value,
      std::reference_wrapper<typename std::decay<T>::type>, T>::type
  UnpackPointers(_SimulationOnly<T> const &arg) {
    return arg.simulation;
  }

  template <typename T>
  static typename std::conditional<
      std::is_reference<T>::value,
      std::reference_wrapper<typename std::decay<T>::type>, T>::type
  UnpackPointers(_SimulationOnly<T> &&arg) {
    return arg.simulation;
  }

  template <typename T>
  static auto UnpackPointers(T &&arg) {
    return std::forward<T>(arg);
  }

  template <class F, typename... Ts>
  static std::function<void(void)> Bind(F &&f, Ts &&... args) {
    return std::bind(f, UnpackPointers(std::forward<Ts>(args))...);
  }

 public:
  template <typename F, typename... Ts>
  Kernel(Program &program, F &&hostFunction, std::string const &kernelName,
         Ts &&... kernelArgs)
      : Kernel(program, kernelName, std::forward<Ts>(kernelArgs)...) {
    hostFunction_ =
        Bind(std::forward<F>(hostFunction), std::forward<Ts>(kernelArgs)...);
  }

  template <typename... Ts>
  Kernel(Program &program, std::string const &kernelName, Ts &&... kernelArgs)
      : program_(program) {
#ifndef HLSLIB_SIMULATE_OPENCL
    kernel_ = xrt::kernel(program.context().device(), program.context().uuid(),
                           kernelName);
    run_ = xrt::run(kernel_);
    SetKernelArguments(0, std::forward<Ts>(kernelArgs)...);
#endif
  }

  inline ~Kernel() = default;

  inline Program const &program() const { return program_; }

#ifndef HLSLIB_SIMULATE_OPENCL
  inline xrt::kernel const &kernel() const { return kernel_; }

  int group_id(int argno) const { return kernel_.group_id(argno); }
#endif

  std::pair<double, double> ExecuteTask() {
    const auto start = std::chrono::high_resolution_clock::now();
    auto event = ExecuteTaskAsync();
    event.wait();
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed =
        1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count();
    return {elapsed, elapsed};
  }

  template <typename EventIterator = Event *,
            typename = typename std::enable_if<
                IsIteratorOfType<EventIterator, Event>() &&
                IsRandomAccess<EventIterator>()>::type>
  std::pair<double, double> ExecuteTask(EventIterator eventsBegin,
                                        EventIterator eventsEnd) {
    for (auto it = eventsBegin; it != eventsEnd; ++it) {
      it->wait();
    }
    return ExecuteTask();
  }

  Event ExecuteTaskAsync() {
#ifndef HLSLIB_SIMULATE_OPENCL
    run_.start();
    return Event(run_);
#else
    return Event([this]() { hostFunction_(); });
#endif
  }

  template <typename EventIterator = Event *,
            typename = typename std::enable_if<
                IsIteratorOfType<EventIterator, Event>() &&
                IsRandomAccess<EventIterator>()>::type>
  Event ExecuteTaskAsync(EventIterator eventsBegin, EventIterator eventsEnd) {
#ifndef HLSLIB_SIMULATE_OPENCL
    for (auto it = eventsBegin; it != eventsEnd; ++it) {
      it->wait();
    }
    run_.start();
    return Event(run_);
#else
    return Event([this, eventsBegin, eventsEnd]() {
      for (auto i = eventsBegin; i != eventsEnd; ++i) {
        i->wait();
      }
      hostFunction_();
    });
#endif
  }

 private:
  Program &program_;
#ifndef HLSLIB_SIMULATE_OPENCL
  xrt::kernel kernel_;
  xrt::run run_;
#endif
  std::function<void(void)> hostFunction_{};
};

//#############################################################################
// Implementations
//#############################################################################

template <typename T, Access access, typename... Ts>
Buffer<T, access> Context::MakeBuffer(Ts &&... args) {
  return Buffer<T, access>(*this, std::forward<Ts>(args)...);
}

Program Context::MakeProgram(std::string const &path) {
#ifndef HLSLIB_SIMULATE_OPENCL
  std::lock_guard<std::mutex> lock(reprogramMutex_);
  uuid_ = device_.load_xclbin(path);
#endif
  loadedPath_ = path;
  return Program(*this, path);
}

Program Context::CurrentlyLoadedProgram() const {
  if (loadedPath_.empty()) {
    ThrowRuntimeError("No program is currently loaded.");
  }
  return Program(*const_cast<Context *>(this), loadedPath_);
}

template <typename... Ts>
Kernel Program::MakeKernel(std::string const &kernelName, Ts &&... args) {
  return Kernel(*this, kernelName, std::forward<Ts>(args)...);
}

template <typename F, typename... Ts>
Kernel Program::MakeKernel(F &&hostFunction, std::string const &kernelName,
                           Ts &&... args) {
  return Kernel(*this, std::forward<F>(hostFunction), kernelName,
                std::forward<Ts>(args)...);
}

inline void WaitForEvents(std::vector<Event> const &events) {
  for (auto &e : events) {
    e.wait();
  }
}

//#############################################################################
// Aligned allocator
//#############################################################################

namespace detail {
template <size_t alignment>
inline void *allocate_aligned_memory(size_t size) {
  if (size == 0) return nullptr;
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) return nullptr;
  return ptr;
}
inline void deallocate_aligned_memory(void *ptr) noexcept {
  free(ptr);
}
}  // namespace detail

template <typename T, size_t alignment>
class AlignedAllocator;

template <size_t alignment>
class AlignedAllocator<void, alignment> {
 public:
  typedef void *pointer;
  typedef const void *const_pointer;
  typedef void value_type;
  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, alignment> other;
  };
};

template <typename T, size_t alignment>
class AlignedAllocator {
 public:
  typedef T value_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef std::true_type propagate_on_container_move_assignment;

  template <class U>
  struct rebind {
    typedef AlignedAllocator<U, alignment> other;
  };

  AlignedAllocator() noexcept {}

  template <class U>
  AlignedAllocator(const AlignedAllocator<U, alignment> &) noexcept {}

  size_type max_size() const noexcept {
    return (size_type(~0) - size_type(alignment)) / sizeof(T);
  }

  pointer address(reference x) const noexcept { return std::addressof(x); }

  const_pointer address(const_reference x) const noexcept {
    return std::addressof(x);
  }

  pointer allocate(
      size_type n,
      typename AlignedAllocator<void, alignment>::const_pointer = 0) {
    void *ptr = detail::allocate_aligned_memory<alignment>(n * sizeof(T));
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return reinterpret_cast<pointer>(ptr);
  }

  void deallocate(pointer p, size_type) noexcept {
    return detail::deallocate_aligned_memory(p);
  }

  template <class U, class... Args>
  void construct(U *p, Args &&... args) {
    ::new (reinterpret_cast<void *>(p)) U(std::forward<Args>(args)...);
  }

  void destroy(pointer p) { p->~T(); }
};

template <typename T, size_t TAlignment, typename U, size_t UAlignment>
inline bool operator==(AlignedAllocator<T, TAlignment> const &,
                       AlignedAllocator<U, UAlignment> const &) noexcept {
  return TAlignment == UAlignment;
}

template <typename T, size_t TAlignment, typename U, size_t UAlignment>
inline bool operator!=(AlignedAllocator<T, TAlignment> const &,
                       AlignedAllocator<U, UAlignment> const &) noexcept {
  return TAlignment != UAlignment;
}

}  // End namespace ocl

}  // End namespace hlslib
