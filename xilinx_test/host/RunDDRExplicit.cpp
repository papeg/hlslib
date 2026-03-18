/// @author    Jannis Widmer (widmerja@ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#include "hlslib/xilinx/XRT.h"
#include <algorithm>
#include <assert.h>
#include <iostream>

constexpr int kDataSize = 1024;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: ./RunDDRExplicit [sw_emu|hw_emu|hw]\n";
    return 1;
  }
  std::string mode(argv[1]);
  if (mode != "sw_emu" && mode != "hw_emu" && mode != "hw") {
    std::cerr << "Unrecognized mode: " << mode << std::endl;
    return 2;
  }
  std::string kernel_path = "DDRMapping_" + mode + ".xclbin";
  std::cout << "Running " << mode << " (" << kernel_path << ")" << std::endl;

  hlslib::ocl::Context context;

  std::cout << std::endl << "Loading Kernel" << std::endl;
  auto program = context.MakeProgram(kernel_path);

  std::cout << "Done" << std::endl << "Initializing memory..." << std::endl;
  std::vector<int, hlslib::ocl::AlignedAllocator<int, 4096>> ddr0mem(kDataSize);
  std::vector<int, hlslib::ocl::AlignedAllocator<int, 4096>> ddr1mem(kDataSize);
  std::fill(ddr1mem.begin(), ddr1mem.end(), 15);

  auto memDevice1 = context.MakeBuffer<int, hlslib::ocl::Access::readWrite>(
      hlslib::ocl::StorageType::DDR, 0, kDataSize);
  auto memDevice2 = context.MakeBuffer<int, hlslib::ocl::Access::readWrite>(
      hlslib::ocl::StorageType::DDR, 1, ddr1mem.begin(), ddr1mem.end());

  // Those calls are the equivalent calls to the ones above with the old
  // interface
  // auto memDevice1 = context.MakeBuffer<int,
  // hlslib::ocl::Access::readWrite>(hlslib::ocl::MemoryBank::bank0, kDataSize);
  // auto memDevice2 = context.MakeBuffer<int,
  // hlslib::ocl::Access::readWrite>(hlslib::ocl::MemoryBank::bank1,ddr1mem.begin(),
  // ddr1mem.end());

  std::cout << "Done" << std::endl;
  std::cout << "Running Kernel" << std::endl;

  auto kernel = program.MakeKernel("DDRExplicit", memDevice1, memDevice2);
  kernel.ExecuteTask();
  memDevice1.CopyToHost(ddr0mem.begin());

  for (int i = 0; i < kDataSize; i++) {
    assert(ddr0mem[i] == ddr1mem[i]);
  }

  std::cout << "Done" << std::endl;
}