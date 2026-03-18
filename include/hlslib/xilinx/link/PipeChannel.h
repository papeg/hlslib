#pragma once

#ifndef HLSLIB_SYNTHESIS

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace hlslib {
namespace link {

class PipeChannel {
 public:
  enum class Mode { kRead, kWrite };

  inline PipeChannel(const std::string &path, Mode mode) : path_(path) {
    int flags = (mode == Mode::kRead) ? O_RDONLY : O_WRONLY;
    fd_ = ::open(path.c_str(), flags);
    if (fd_ < 0) {
      throw std::runtime_error("PipeChannel: failed to open " + path + ": " +
                               std::strerror(errno));
    }
  }

  PipeChannel(const PipeChannel &) = delete;
  PipeChannel &operator=(const PipeChannel &) = delete;

  inline PipeChannel(PipeChannel &&other) noexcept
      : fd_(other.fd_), path_(std::move(other.path_)) {
    other.fd_ = -1;
  }

  inline PipeChannel &operator=(PipeChannel &&other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) ::close(fd_);
      fd_ = other.fd_;
      path_ = std::move(other.path_);
      other.fd_ = -1;
    }
    return *this;
  }

  inline ~PipeChannel() {
    if (fd_ >= 0) ::close(fd_);
  }

  inline bool FullRead(void *buf, size_t count) {
    auto *ptr = static_cast<char *>(buf);
    size_t remaining = count;
    while (remaining > 0) {
      ssize_t n = ::read(fd_, ptr, remaining);
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("PipeChannel::FullRead: " +
                                 std::string(std::strerror(errno)));
      }
      if (n == 0) return false;
      ptr += n;
      remaining -= static_cast<size_t>(n);
    }
    return true;
  }

  inline void FullWrite(const void *buf, size_t count) {
    auto *ptr = static_cast<const char *>(buf);
    size_t remaining = count;
    while (remaining > 0) {
      ssize_t n = ::write(fd_, ptr, remaining);
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("PipeChannel::FullWrite: " +
                                 std::string(std::strerror(errno)));
      }
      ptr += n;
      remaining -= static_cast<size_t>(n);
    }
  }

  inline void SetBufferSize(size_t bytes) {
    if (bytes == 0) return;
#ifdef __linux__
    ::fcntl(fd_, F_SETPIPE_SZ, static_cast<int>(bytes));
#endif
    (void)bytes;
  }

  int fd() const { return fd_; }
  const std::string &path() const { return path_; }

 private:
  int fd_ = -1;
  std::string path_;
};

}  // namespace link
}  // namespace hlslib

#endif  // HLSLIB_SYNTHESIS
