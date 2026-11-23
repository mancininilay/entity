/**
 * The capacity test of the particle exchange (`Particles::Communicate`).
 *
 * Received particles first fill the holes of a species (dead and outgoing
 * particles among its first npart slots) and only the rest is appended, so an
 * exchange fits exactly when
 *
 *   received <= maxnpart - (number of alive particles).
 *
 * Four cases on a ring of ranks in 1D (run with 1 or 2 ranks), selected by the
 * first argument:
 *
 *   full   a species that is exactly full and receives nothing;
 *   fill   a species that is exactly full after the exchange (2 ranks: the even
 *          rank holds maxnpart - k and receives k; 1 rank: k particles leave
 *          and come back);
 *   holes  a full species with k dead slots that receives k particles (they
 *          must land in the dead slots);
 *   over   one particle more than fits (2 ranks): the engine must stop with
 *          "Too many particles to receive".
 *
 * The first three print "OK <case>"; `over` must not reach its last line. An
 * expectation of this test that fails prints a message starting with
 * "comm-capacity:".
 */

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "arch/mpi_tags.h"
#include "utils/error.h"

#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <plog/Appenders/ConsoleAppender.h>
#include <plog/Formatters/MessageOnlyFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {
  using namespace ntt;
  using prtls_t = Particles<Dim::_1D, Coord::Cartesian>;

  const short TAG_UP = mpi::PrtlSendTag<Dim::_1D>::dir2tag({ 1 });

  // slot p of the species gets the tag `tags[p]` and a weight that names the
  // rank and the slot
  void fill(prtls_t& sp, const std::vector<short>& tags, int rank) {
    const auto n      = static_cast<npart_t>(tags.size());
    auto       tag_h  = Kokkos::create_mirror_view(sp.tag);
    auto       wei_h  = Kokkos::create_mirror_view(sp.weight);
    auto       i1_h   = Kokkos::create_mirror_view(sp.i1);
    auto       i1p_h  = Kokkos::create_mirror_view(sp.i1_prev);
    auto       dx1_h  = Kokkos::create_mirror_view(sp.dx1);
    auto       dx1p_h = Kokkos::create_mirror_view(sp.dx1_prev);
    for (npart_t p = 0; p < n; ++p) {
      tag_h(p)  = tags[p];
      wei_h(p)  = static_cast<real_t>(1000 * (rank + 1) + static_cast<int>(p));
      i1_h(p)   = 3;
      i1p_h(p)  = 3;
      dx1_h(p)  = static_cast<prtldx_t>(0.5);
      dx1p_h(p) = static_cast<prtldx_t>(0.5);
    }
    Kokkos::deep_copy(sp.tag, tag_h);
    Kokkos::deep_copy(sp.weight, wei_h);
    Kokkos::deep_copy(sp.i1, i1_h);
    Kokkos::deep_copy(sp.i1_prev, i1p_h);
    Kokkos::deep_copy(sp.dx1, dx1_h);
    Kokkos::deep_copy(sp.dx1_prev, dx1p_h);
    sp.set_npart(n);
  }

  void exchange(prtls_t& sp, int rank, int size) {
    array_t<int*> shifts { "shifts", static_cast<std::size_t>(sp.ntags() - 2) };
    dir::dirs_t<Dim::_1D>     dirs { { -1 }, { 1 } };
    dir::map_t<Dim::_1D, int> send_ranks, recv_ranks;
    send_ranks[{ -1 }] = (rank - 1 + size) % size;
    recv_ranks[{ -1 }] = (rank + 1) % size;
    send_ranks[{ 1 }]  = (rank + 1) % size;
    recv_ranks[{ 1 }]  = (rank - 1 + size) % size;
    sp.Communicate(dirs, shifts, shifts, shifts, send_ranks, recv_ranks);
  }

  // weights of the alive particles among the first npart slots, sorted; and
  // the weights slot by slot (0 for a slot that is not alive)
  auto alive_weights(const prtls_t& sp, std::vector<int>& by_slot)
    -> std::vector<int> {
    auto tag_h = Kokkos::create_mirror_view(sp.tag);
    auto wei_h = Kokkos::create_mirror_view(sp.weight);
    Kokkos::deep_copy(tag_h, sp.tag);
    Kokkos::deep_copy(wei_h, sp.weight);
    std::vector<int> w;
    by_slot.assign(sp.npart(), 0);
    for (npart_t p = 0; p < sp.npart(); ++p) {
      if (tag_h(p) == ParticleTag::alive) {
        by_slot[p] = static_cast<int>(wei_h(p));
        w.push_back(by_slot[p]);
      }
    }
    std::sort(w.begin(), w.end());
    return w;
  }

  void expect(bool ok, const std::string& what) {
    raise::ErrorIf(not ok, "comm-capacity: " + what, HERE);
  }
} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  // the message of an error raised by the engine goes to the console before
  // the engine aborts
  static plog::ConsoleAppender<plog::MessageOnlyFormatter> console;
  plog::init(plog::error, &console);
  try {
    const std::string which = (argc > 1) ? std::string(argv[1]) : std::string();
    int               rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    expect(size == 1 or size == 2, "run with 1 or 2 ranks");

    const npart_t    N = 64, k = 5;
    auto             sp     = prtls_t(1,
                      "test",
                      1.0f,
                      1.0f,
                      N,
                      0u,
                      0u,
                      ParticlePusher::BORIS,
                      false,
                      RadiativeDrag::NONE,
                      EmissionType::NONE,
                      0,
                      0);
    const bool       sender = (size == 1) or (rank == 1);
    const int        from = (size == 1) ? 0 : 1; // rank whose particles arrive
    std::vector<int> by_slot;

    if (which == "full") {
      fill(sp, std::vector<short>(N, ParticleTag::alive), rank);
      exchange(sp, rank, size);
      const auto w = alive_weights(sp, by_slot);
      expect(sp.npart() == N, "full: npart changed");
      expect(w.size() == N, "full: a particle was lost");
      for (npart_t p = 0; p < N; ++p) {
        expect(by_slot[p] == 1000 * (rank + 1) + static_cast<int>(p),
               "full: a slot changed");
      }
    } else if (which == "fill") {
      // 1 rank: N slots, the last k leave upwards and come back.
      // 2 ranks: rank 1 holds 2 k, of which the last k leave; rank 0 holds N - k.
      std::vector<short> tags;
      if (size == 1) {
        tags.assign(N, ParticleTag::alive);
        std::fill(tags.end() - k, tags.end(), TAG_UP);
      } else if (rank == 1) {
        tags.assign(2 * k, ParticleTag::alive);
        std::fill(tags.end() - k, tags.end(), TAG_UP);
      } else {
        tags.assign(N - k, ParticleTag::alive);
      }
      fill(sp, tags, rank);
      exchange(sp, rank, size);
      const auto w = alive_weights(sp, by_slot);
      if (size == 1 or rank == 0) {
        expect(sp.npart() == N, "fill: the species is not exactly full");
        expect(w.size() == N, "fill: a particle was lost");
        // the k that arrived carry the weights of the k that left on `from`
        const npart_t first = (size == 1) ? N - k : k;
        for (npart_t j = 0; j < k; ++j) {
          const int wanted = 1000 * (from + 1) + static_cast<int>(first + j);
          expect(std::binary_search(w.begin(), w.end(), wanted),
                 "fill: a received particle is missing");
        }
      } else {
        expect(sp.npart() == 2 * k and w.size() == k, "fill: the sender is wrong");
      }
    } else if (which == "holes") {
      // the receiver is full: its first k slots are dead, the rest alive (on 1
      // rank the last k leave and come back). The arrivals must sit in the
      // dead slots.
      std::vector<short> tags;
      if (size == 1) {
        tags.assign(N, ParticleTag::alive);
        std::fill(tags.begin(), tags.begin() + k, ParticleTag::dead);
        std::fill(tags.end() - k, tags.end(), TAG_UP);
      } else if (rank == 1) {
        tags.assign(2 * k, ParticleTag::alive);
        std::fill(tags.end() - k, tags.end(), TAG_UP);
      } else {
        tags.assign(N, ParticleTag::alive);
        std::fill(tags.begin(), tags.begin() + k, ParticleTag::dead);
      }
      fill(sp, tags, rank);
      exchange(sp, rank, size);
      const auto w = alive_weights(sp, by_slot);
      if (size == 1 or rank == 0) {
        expect(sp.npart() == N, "holes: npart changed");
        expect(w.size() == ((size == 1) ? N - k : N), "holes: wrong number alive");
        const npart_t    first = (size == 1) ? N - k : k;
        std::vector<int> in_holes(by_slot.begin(), by_slot.begin() + k);
        std::sort(in_holes.begin(), in_holes.end());
        for (npart_t j = 0; j < k; ++j) {
          expect(in_holes[j] == 1000 * (from + 1) + static_cast<int>(first + j),
                 "holes: the arrivals are not in the dead slots");
        }
      }
    } else if (which == "over") {
      expect(size == 2, "over: needs 2 ranks");
      // rank 0 is full and alive; rank 1 sends it one particle
      std::vector<short> tags;
      if (rank == 1) {
        tags.assign(2 * k, ParticleTag::alive);
        tags.back() = TAG_UP;
      } else {
        tags.assign(N, ParticleTag::alive);
      }
      fill(sp, tags, rank);
      exchange(sp, rank, size);
      if (rank == 0) {
        std::cout << "OVERFLOW NOT DETECTED" << std::endl;
      }
      GlobalFinalize();
      return 0;
    } else {
      expect(false, "unknown case, give one of full, fill, holes, over");
    }
    (void)sender;
    if (rank == 0) {
      std::cout << "OK " << which << " on " << size << " rank(s)" << std::endl;
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
}
