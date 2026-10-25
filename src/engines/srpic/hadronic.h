/**
 * @file engines/srpic/hadronic.h
 * @brief [fork:hadronic] Production placement of the hadronic interaction stage
 *        (spec II.1, T14b-lite): the engine launches the generator's stage after
 *        the current deposit and before the particle exchange, so that the ions
 *        absorbed (dead-tagged by the pusher) or send-tagged in this step are
 *        still in place when the kernel runs; newborn pairs inherit the parent's
 *        tag and are exchanged by the same `CommunicateParticles`. The switch is
 *        the generator's `hadronic_placement_production()` (read from its input;
 *        default off, in which case the generator runs the stage in its
 *        `CustomPostStep` as before and nothing here is called).
 * @implements
 *   - ntt::srpic::HadronicInteractions<> -> void
 * @namespaces:
 *   - ntt::srpic::
 */

#ifndef ENGINES_SRPIC_HADRONIC_H
#define ENGINES_SRPIC_HADRONIC_H

#include "enums.h"
#include "global.h"

#include "traits/pgen.h"
#include "utils/log.h"

#include "framework/domain/domain.h"

namespace ntt {
  namespace srpic {

    /**
     * @brief Launch the generator's hadronic stage on one local domain at the
     *        production placement
     * @param domain the local domain (particles after the push, dead and send
     *               tags set, before the exchange)
     * @param pgen the problem generator (must satisfy traits::pgen::HasHadronicStage)
     * @param step, time the current step and its time t^n (the stage stamps t^{n+1})
     */
    template <class M, class PG>
    void HadronicInteractions(Domain<SimEngine::SRPIC, M>& domain,
                              PG&                          pgen,
                              timestep_t                   step,
                              simtime_t                    time) {
      static_assert(::traits::pgen::HasHadronicStage<PG, Domain<SimEngine::SRPIC, M>>,
                    "HadronicInteractions needs a generator with HadronicStage and "
                    "hadronic_placement_production");
      logger::Checkpoint("Launching the hadronic interaction stage (production placement)", HERE);
      pgen.HadronicStage(step, time, domain);
    }

  } // namespace srpic
} // namespace ntt

#endif // ENGINES_SRPIC_HADRONIC_H
