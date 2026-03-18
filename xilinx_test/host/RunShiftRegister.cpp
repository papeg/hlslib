#include <cmath>
#include <iostream>
#include "ShiftRegister.h"
#include "hlslib/xilinx/XRT.h"

void Reference(std::vector<Data_t> &domain) {
  std::vector<Data_t> buffer(domain);
  for (int t = 0; t < T; ++t) {
    for (int i = 1; i < H - 1; ++i) {
      for (int j = 1; j < W - 1; ++j) {
        buffer[i * W + j] = static_cast<Data_t>(0.25) *
                            (domain[(i - 1) * W + j] + domain[(i + 1) * W + j] +
                             domain[i * W + j - 1] + domain[i * W + j + 1]);
      }
    }
    domain.swap(buffer);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: ./RunShiftRegister [sw_emu|hw_emu|hw]\n";
    return 1;
  }
  std::string mode(argv[1]);
  if (mode != "sw_emu" && mode != "hw_emu" && mode != "hw") {
    std::cerr << "Unrecognized mode: " << mode << std::endl;
    return 2;
  }
  std::string kernel_path = "ShiftRegister_" + mode + ".xclbin";
  std::cout << "Running " << mode << " (" << kernel_path << ")" << std::endl;

  std::cout << "Initializing host memory...\n" << std::flush;
  std::vector<Data_t> host_buffer(W * H, 0);
  // Set boundaries to 1
  for (int i = 0; i < W; ++i) {
    host_buffer[i] = 1;
    host_buffer[W * (H - 1) + i] = 1;
  }
  for (int i = 0; i < H; ++i) {
    host_buffer[i * W] = 1;
    host_buffer[i * W + W - 1] = 1;
  }
  std::vector<Data_t> reference(host_buffer);

  std::cout << "Creating context...\n" << std::flush;
  hlslib::ocl::Context context;
  std::cout << "Allocating device memory...\n" << std::flush;
  auto device_buffer =
      context.MakeBuffer<Data_t, hlslib::ocl::Access::readWrite>(2 * W * H);
  std::cout << "Creating program from binary...\n" << std::flush;
  auto program = context.MakeProgram(kernel_path);
  std::cout << "Creating kernels...\n" << std::flush;
  auto kernel =
      program.MakeKernel("ShiftRegister", device_buffer, device_buffer);
  std::cout << "Copying data to device...\n" << std::flush;
  // Copy to both sections of device memory, so that the boundary conditions
  // are reflected in both
  device_buffer.CopyFromHost(0, W * H, host_buffer.cbegin());
  device_buffer.CopyFromHost(W * H, W * H, host_buffer.cbegin());

  // Execute kernel
  std::cout << "Launching kernels...\n" << std::flush;
  kernel.ExecuteTask();

  // Copy back result
  std::cout << "Copying back result...\n" << std::flush;
  int offset = (T % 2 == 0) ? 0 : H * W;
  device_buffer.CopyToHost(offset, H * W, host_buffer.begin());

  // Run reference implementation
  std::cout << "Running reference implementation...\n" << std::flush;
  Reference(reference);

  // Compare result
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      auto diff = std::abs(host_buffer[i * W + j] - reference[i * W + j]);
      if (diff > 1e-4 * reference[i * W + j]) {
        std::cerr << "Mismatch found at (" << i << ", " << j
                  << "): " << host_buffer[i * W + j] << " (should be "
                  << reference[i * W + j] << ").\n";
        return 3;
      }
    }
  }

  std::cout << "Successfully verified result.\n";

  return 0;
}
