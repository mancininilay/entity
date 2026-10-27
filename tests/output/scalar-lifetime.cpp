#include "global.h"
#include "output/checkpoint.h"
#include "output/utils/writers.h"
#include <adios2.h>
#if defined(MPI_ENABLED)
#include <mpi.h>
#endif
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// The scalar helper must consume its input before returning: callers pass
// temporary mesh vectors and particle-count values. Mutate live storage here
// to exercise the same lifetime contract without relying on undefined behavior.
auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  int rank = 0, size = 1;
#if defined(MPI_ENABLED)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  int failed = 0;
  try {
    long long token = std::chrono::system_clock::now().time_since_epoch().count();
#if defined(MPI_ENABLED)
    MPI_Bcast(&token, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    adios2::ADIOS adios(MPI_COMM_WORLD);
#else
    adios2::ADIOS adios;
#endif
    const auto root = "scalar-lifetime-" + std::to_string(token);
    checkpoint::Writer checkpoint;
    checkpoint.init(&adios, root, 1, 0.0, -1);
    // Force pointer puts to defer even for a single element. Otherwise BP5 can
    // hide a lifetime bug by eagerly copying small values on some ranks.
    checkpoint.io().SetParameter("MinDeferredSize", "0");
    checkpoint.io().DefineVariable<ncells_t>("nx", {static_cast<size_t>(size)},
                                            {static_cast<size_t>(rank)}, {1});
    checkpoint.io().DefineVariable<real_t>("xmin", {static_cast<size_t>(size)},
                                          {static_cast<size_t>(rank)}, {1});
    checkpoint.io().DefineVariable<npart_t>("npart", {static_cast<size_t>(size)},
                                           {static_cast<size_t>(rank)}, {1});
    for (timestep_t step : {7, 19}) {
      const simtime_t time = static_cast<simtime_t>(step) * 0.25;
      checkpoint.beginSaving(step, time);
      ncells_t nx = 2080 + rank + step;
      real_t xmin = static_cast<real_t>(rank * 2080) + 0.5;
      npart_t npart = 1000 + rank + step;
      out::WriteVariable(checkpoint.io(), checkpoint.writer(), "nx", nx, size, rank);
      out::WriteVariable(checkpoint.io(), checkpoint.writer(), "xmin", xmin, size, rank);
      out::WriteVariable(checkpoint.io(), checkpoint.writer(), "npart", npart, size, rank);
      nx = 999999;
      xmin = -9876;
      npart = 888888;
      checkpoint.endSaving();
#if defined(MPI_ENABLED)
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      auto io = adios.DeclareIO("read-" + std::to_string(step));
      auto reader = io.Open(checkpoint.written().back().first, adios2::Mode::ReadRandomAccess);
      std::vector<ncells_t> sizes(size);
      std::vector<real_t> bounds(size);
      std::vector<npart_t> counts(size);
      reader.Get(io.InquireVariable<ncells_t>("nx"), sizes.data(), adios2::Mode::Sync);
      reader.Get(io.InquireVariable<real_t>("xmin"), bounds.data(), adios2::Mode::Sync);
      reader.Get(io.InquireVariable<npart_t>("npart"), counts.data(), adios2::Mode::Sync);
      timestep_t stored_step = 0;
      simtime_t stored_time = 0;
      reader.Get(io.InquireVariable<timestep_t>("Step"), &stored_step, adios2::Mode::Sync);
      reader.Get(io.InquireVariable<simtime_t>("Time"), &stored_time, adios2::Mode::Sync);
      reader.Close();
      if (stored_step != step || stored_time != time) {
        throw std::runtime_error("checkpoint step/time changed before flush");
      }
      for (int r = 0; r < size; ++r) {
        if (sizes[r] != 2080 + r + step ||
            bounds[r] != static_cast<real_t>(r * 2080) + 0.5 ||
            counts[r] != 1000 + r + step) {
          throw std::runtime_error("scalar lifetime failure at rank " + std::to_string(r) +
              ": nx=" + std::to_string(sizes[r]) + ", xmin=" + std::to_string(bounds[r]) +
              ", npart=" + std::to_string(counts[r]));
        }
      }
    }
    if (rank == 0) std::cout << "SCALAR_LIFETIME_PASS ranks=" << size << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    failed = 1;
  }
#if defined(MPI_ENABLED)
  int all_failed = 0;
  MPI_Allreduce(&failed, &all_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  failed = all_failed;
#endif
  ntt::GlobalFinalize();
  return failed;
}
