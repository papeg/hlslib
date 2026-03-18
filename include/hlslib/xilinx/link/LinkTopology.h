#pragma once

#ifndef HLSLIB_SYNTHESIS

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace hlslib {
namespace link {

struct LinkDescriptor {
  int src_fpga;
  int dst_fpga;
  int link_index;
};

inline void CreatePipe(const std::string &path) {
  if (::mkfifo(path.c_str(), 0600) != 0 && errno != EEXIST) {
    throw std::runtime_error("CreatePipe: mkfifo failed for " + path + ": " +
                             std::strerror(errno));
  }
}

inline void CreateSymlink(const std::string &target,
                          const std::string &link_path) {
  ::unlink(link_path.c_str());
  if (::symlink(target.c_str(), link_path.c_str()) != 0) {
    throw std::runtime_error("CreateSymlink: symlink(" + target + ", " +
                             link_path + "): " + std::strerror(errno));
  }
}

inline void CreateDirectory(const std::string &path) {
  if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    throw std::runtime_error("CreateDirectory: mkdir failed for " + path +
                             ": " + std::strerror(errno));
  }
}

inline void Remove(const std::string &path) {
  ::unlink(path.c_str());
}

inline std::string PipeName(const LinkDescriptor &link) {
  return "fpga" + std::to_string(link.src_fpga) + "_to_fpga" +
         std::to_string(link.dst_fpga) + "_link" +
         std::to_string(link.link_index);
}

inline std::string SetupTopology(const std::string &workspace, int num_fpgas,
                                 const std::vector<LinkDescriptor> &links) {
  CreateDirectory(workspace);
  CreateDirectory(workspace + "/pipes");

  for (int i = 0; i < num_fpgas; ++i) {
    CreateDirectory(workspace + "/fpga_" + std::to_string(i));
  }

  for (const auto &link : links) {
    std::string name = PipeName(link);
    std::string pipe_path = workspace + "/pipes/" + name;
    CreatePipe(pipe_path);

    std::string src_link = workspace + "/fpga_" +
                           std::to_string(link.src_fpga) + "/out_" +
                           std::to_string(link.link_index);
    CreateSymlink("../pipes/" + name, src_link);

    std::string dst_link = workspace + "/fpga_" +
                           std::to_string(link.dst_fpga) + "/in_" +
                           std::to_string(link.link_index);
    CreateSymlink("../pipes/" + name, dst_link);
  }

  return workspace;
}

inline void TeardownTopology(const std::string &workspace, int num_fpgas,
                             const std::vector<LinkDescriptor> &links) {
  for (const auto &link : links) {
    std::string name = PipeName(link);
    Remove(workspace + "/fpga_" + std::to_string(link.src_fpga) + "/out_" +
           std::to_string(link.link_index));
    Remove(workspace + "/fpga_" + std::to_string(link.dst_fpga) + "/in_" +
           std::to_string(link.link_index));
    Remove(workspace + "/pipes/" + name);
  }

  for (int i = 0; i < num_fpgas; ++i) {
    ::rmdir((workspace + "/fpga_" + std::to_string(i)).c_str());
  }
  ::rmdir((workspace + "/pipes").c_str());
  ::rmdir(workspace.c_str());
}

inline std::string SendPath(const std::string &workspace, int fpga_rank,
                            int link_index) {
  return workspace + "/fpga_" + std::to_string(fpga_rank) + "/out_" +
         std::to_string(link_index);
}

inline std::string RecvPath(const std::string &workspace, int fpga_rank,
                            int link_index) {
  return workspace + "/fpga_" + std::to_string(fpga_rank) + "/in_" +
         std::to_string(link_index);
}

}  // namespace link
}  // namespace hlslib

#endif  // HLSLIB_SYNTHESIS
