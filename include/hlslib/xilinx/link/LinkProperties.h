#pragma once

#ifndef HLSLIB_SYNTHESIS

#include <cstddef>

namespace hlslib {
namespace link {

struct LinkProperties {
  size_t latency_elements = 0;
  size_t pipe_buffer_bytes = 0;
  size_t stream_depth = 16;
};

}  // namespace link
}  // namespace hlslib

#endif  // HLSLIB_SYNTHESIS
