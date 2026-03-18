#pragma once

#ifndef HLSLIB_SYNTHESIS

#include <string>
#include <thread>

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/link/LinkBridge.h"
#include "hlslib/xilinx/link/LinkProperties.h"
#include "hlslib/xilinx/link/PipeChannel.h"

namespace hlslib {
namespace link {

template <typename T>
class LinkEndpoint {
 public:
  enum class Direction { kSend, kReceive };

  inline LinkEndpoint(const std::string &pipe_path, Direction direction,
                      size_t transfer_count,
                      const LinkProperties &props = {})
      : pipe_path_(pipe_path),
        direction_(direction),
        transfer_count_(transfer_count),
        props_(props),
        stream_(pipe_path.c_str()) {}

  LinkEndpoint(const LinkEndpoint &) = delete;
  LinkEndpoint &operator=(const LinkEndpoint &) = delete;

  inline ~LinkEndpoint() {
    if (bridge_thread_.joinable()) {
      bridge_thread_.join();
    }
  }

  hlslib::Stream<T> &stream() { return stream_; }

  inline void Start() {
    auto mode = (direction_ == Direction::kSend) ? PipeChannel::Mode::kWrite
                                                 : PipeChannel::Mode::kRead;

    if (direction_ == Direction::kSend) {
      bridge_thread_ = std::thread(
          [this, mode]() {
            PipeChannel pipe(pipe_path_, mode);
            if (props_.pipe_buffer_bytes > 0) {
              pipe.SetBufferSize(props_.pipe_buffer_bytes);
            }
            SenderBridge<T>(stream_, pipe, transfer_count_, props_);
          });
    } else {
      bridge_thread_ = std::thread(
          [this, mode]() {
            PipeChannel pipe(pipe_path_, mode);
            ReceiverBridge<T>(pipe, stream_, transfer_count_);
          });
    }
  }

  inline void Join() {
    if (bridge_thread_.joinable()) {
      bridge_thread_.join();
    }
  }

 private:
  std::string pipe_path_;
  Direction direction_;
  size_t transfer_count_;
  LinkProperties props_;
  hlslib::Stream<T> stream_;
  std::thread bridge_thread_;
};

}  // namespace link
}  // namespace hlslib

#endif  // HLSLIB_SYNTHESIS
