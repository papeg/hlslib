#include "catch.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "hlslib/xilinx/DataPack.h"
#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/link/Link.h"

using namespace hlslib::link;

static std::string MakeWorkspace(const std::string &test_name) {
  return "/tmp/hlslib_link_test_" + std::to_string(getpid()) + "_" + test_name;
}

static pid_t ForkChild(std::function<int()> func) {
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("fork failed");
  }
  if (pid == 0) {
    try {
      int rc = func();
      _exit(rc);
    } catch (...) {
      _exit(1);
    }
  }
  return pid;
}

static bool WaitChild(pid_t pid) {
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

TEST_CASE("LinkBasicTransfer", "[link]") {
  using T = hlslib::DataPack<uint32_t, 8>;
  constexpr size_t N = 100;

  std::string ws = MakeWorkspace("basic");
  std::vector<LinkDescriptor> links = {{0, 1, 0}};
  SetupTopology(ws, 2, links);

  pid_t child = ForkChild([&]() -> int {
    LinkEndpoint<T> rx(RecvPath(ws, 1, 0), LinkEndpoint<T>::Direction::kReceive,
                       N);
    rx.Start();
    for (size_t i = 0; i < N; ++i) {
      T val = rx.stream().Pop();
      for (int j = 0; j < 8; ++j) {
        if (static_cast<uint32_t>(val[j]) != static_cast<uint32_t>(i + j)) {
          return 1;
        }
      }
    }
    rx.Join();
    return 0;
  });

  {
    LinkEndpoint<T> tx(SendPath(ws, 0, 0), LinkEndpoint<T>::Direction::kSend,
                       N);
    tx.Start();
    for (size_t i = 0; i < N; ++i) {
      T val;
      for (int j = 0; j < 8; ++j) {
        val.Set(j, static_cast<uint32_t>(i + j));
      }
      tx.stream().Push(val);
    }
    tx.Join();
  }

  bool child_ok = WaitChild(child);
  TeardownTopology(ws, 2, links);
  REQUIRE(child_ok);
}

TEST_CASE("LinkBidirectional", "[link]") {
  using T = hlslib::DataPack<uint32_t, 8>;
  constexpr size_t N = 50;

  std::string ws = MakeWorkspace("bidir");
  std::vector<LinkDescriptor> links = {{0, 1, 0}, {1, 0, 1}};
  SetupTopology(ws, 2, links);

  pid_t child = ForkChild([&]() -> int {
    LinkEndpoint<T> rx(RecvPath(ws, 1, 0), LinkEndpoint<T>::Direction::kReceive,
                       N);
    LinkEndpoint<T> tx(SendPath(ws, 1, 1), LinkEndpoint<T>::Direction::kSend,
                       N);
    rx.Start();
    tx.Start();

    for (size_t i = 0; i < N; ++i) {
      T val = rx.stream().Pop();
      if (static_cast<uint32_t>(val[0]) != static_cast<uint32_t>(i)) return 1;
      T reply;
      reply.Fill(static_cast<uint32_t>(i * 2));
      tx.stream().Push(reply);
    }

    rx.Join();
    tx.Join();
    return 0;
  });

  {
    LinkEndpoint<T> tx(SendPath(ws, 0, 0), LinkEndpoint<T>::Direction::kSend,
                       N);
    LinkEndpoint<T> rx(RecvPath(ws, 0, 1), LinkEndpoint<T>::Direction::kReceive,
                       N);
    tx.Start();
    rx.Start();

    for (size_t i = 0; i < N; ++i) {
      T val;
      val.Fill(static_cast<uint32_t>(i));
      tx.stream().Push(val);
    }

    for (size_t i = 0; i < N; ++i) {
      T val = rx.stream().Pop();
      REQUIRE(static_cast<uint32_t>(val[0]) == static_cast<uint32_t>(i * 2));
    }

    tx.Join();
    rx.Join();
  }

  bool child_ok = WaitChild(child);
  TeardownTopology(ws, 2, links);
  REQUIRE(child_ok);
}

TEST_CASE("LinkBackPressure", "[link]") {
  using T = uint64_t;
  constexpr size_t N = 200;

  std::string ws = MakeWorkspace("backpressure");
  std::vector<LinkDescriptor> links = {{0, 1, 0}};
  SetupTopology(ws, 2, links);

  pid_t child = ForkChild([&]() -> int {
    LinkEndpoint<T> rx(RecvPath(ws, 1, 0), LinkEndpoint<T>::Direction::kReceive,
                       N);
    rx.Start();
    for (size_t i = 0; i < N; ++i) {
      if (i % 10 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
      T val = rx.stream().Pop();
      if (val != static_cast<T>(i)) return 1;
    }
    rx.Join();
    return 0;
  });

  {
    LinkEndpoint<T> tx(SendPath(ws, 0, 0), LinkEndpoint<T>::Direction::kSend,
                       N);
    tx.Start();
    for (size_t i = 0; i < N; ++i) {
      tx.stream().Push(static_cast<T>(i));
    }
    tx.Join();
  }

  bool child_ok = WaitChild(child);
  TeardownTopology(ws, 2, links);
  REQUIRE(child_ok);
}

TEST_CASE("LinkLatency", "[link]") {
  using T = uint64_t;
  constexpr size_t N = 20;
  constexpr size_t kLatency = 5;

  std::string ws = MakeWorkspace("latency");
  std::vector<LinkDescriptor> links = {{0, 1, 0}};
  SetupTopology(ws, 2, links);

  LinkProperties props;
  props.latency_elements = kLatency;

  pid_t child = ForkChild([&]() -> int {
    LinkEndpoint<T> rx(RecvPath(ws, 1, 0), LinkEndpoint<T>::Direction::kReceive,
                       N);
    rx.Start();
    for (size_t i = 0; i < N; ++i) {
      T val = rx.stream().Pop();
      if (val != static_cast<T>(i)) return 1;
    }
    rx.Join();
    return 0;
  });

  {
    LinkEndpoint<T> tx(SendPath(ws, 0, 0), LinkEndpoint<T>::Direction::kSend, N,
                       props);
    tx.Start();
    for (size_t i = 0; i < N; ++i) {
      tx.stream().Push(static_cast<T>(i));
    }
    tx.Join();
  }

  bool child_ok = WaitChild(child);
  TeardownTopology(ws, 2, links);
  REQUIRE(child_ok);
}

TEST_CASE("Link512bit", "[link]") {
  using T = hlslib::DataPack<uint64_t, 8>;
  constexpr size_t N = 50;

  std::string ws = MakeWorkspace("512bit");
  std::vector<LinkDescriptor> links = {{0, 1, 0}};
  SetupTopology(ws, 2, links);

  pid_t child = ForkChild([&]() -> int {
    LinkEndpoint<T> rx(RecvPath(ws, 1, 0), LinkEndpoint<T>::Direction::kReceive,
                       N);
    rx.Start();
    for (size_t i = 0; i < N; ++i) {
      T val = rx.stream().Pop();
      for (int j = 0; j < 8; ++j) {
        if (static_cast<uint64_t>(val[j]) !=
            static_cast<uint64_t>(i * 100 + j)) {
          return 1;
        }
      }
    }
    rx.Join();
    return 0;
  });

  {
    LinkEndpoint<T> tx(SendPath(ws, 0, 0), LinkEndpoint<T>::Direction::kSend,
                       N);
    tx.Start();
    for (size_t i = 0; i < N; ++i) {
      T val;
      for (int j = 0; j < 8; ++j) {
        val.Set(j, static_cast<uint64_t>(i * 100 + j));
      }
      tx.stream().Push(val);
    }
    tx.Join();
  }

  bool child_ok = WaitChild(child);
  TeardownTopology(ws, 2, links);
  REQUIRE(child_ok);
}
