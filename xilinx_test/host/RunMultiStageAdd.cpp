/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/xilinx/XRT.h"
#include "MultiStageAdd.h"
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: ./RunMultiStageAdd [sw_emu|hw_emu|hw]\n";
    return 1;
  }
  std::string mode(argv[1]);
  if (mode != "sw_emu" && mode != "hw_emu" && mode != "hw") {
    std::cerr << "Unrecognized mode: " << mode << std::endl;
    return 2;
  }

  std::string xclbin = "MultiStageAdd_" + mode + ".xclbin";
  std::cout << "Running " << mode << " (" << xclbin << ")" << std::endl;

  hlslib::ocl::Context context;

  std::vector<Data_t> memHost(kNumElements, 0);
  auto memDevice = context.MakeBuffer<int, hlslib::ocl::Access::readWrite>(
      hlslib::ocl::MemoryBank::bank0, memHost.cbegin(), memHost.cend());

  auto program = context.MakeProgram(xclbin);
  auto kernel = program.MakeKernel("MultiStageAdd", memDevice, memDevice);
  kernel.ExecuteTask();

  memDevice.CopyToHost(memHost.begin());
  for (auto &m : memHost) {
    if (m != kStages) {
      std::cerr << "Verification failed." << std::endl;
      return 3;
    }
  }
  std::cout << "Kernel ran successfully." << std::endl;
  return 0;
}
