#pragma once

#ifndef HLSLIB_SYNTHESIS

#include <cstddef>
#include <queue>

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/link/LinkProperties.h"
#include "hlslib/xilinx/link/PipeChannel.h"

namespace hlslib {
namespace link {

template <typename T>
inline void SenderBridge(hlslib::Stream<T> &stream, PipeChannel &pipe,
                         size_t count, const LinkProperties &props) {
  std::queue<T> delay_queue;
  size_t sent = 0;
  size_t popped = 0;

  while (sent < count) {
    if (popped < count) {
      delay_queue.push(stream.Pop());
      ++popped;
    }

    if (delay_queue.size() > props.latency_elements || popped >= count) {
      T out = delay_queue.front();
      delay_queue.pop();
      pipe.FullWrite(&out, sizeof(T));
      ++sent;
    }
  }
}

template <typename T>
inline void ReceiverBridge(PipeChannel &pipe, hlslib::Stream<T> &stream,
                           size_t count) {
  for (size_t i = 0; i < count; ++i) {
    T val;
    if (!pipe.FullRead(&val, sizeof(T))) {
      break;
    }
    stream.Push(val);
  }
}

}  // namespace link
}  // namespace hlslib

#endif  // HLSLIB_SYNTHESIS
