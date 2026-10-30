# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Run-based tests of the Bethe-Heitler prototype (P2 steps 2 and 3)
#
# Registered only when the build's problem generator is `bh_proto`. The bash
# scripts run `entity.xc` of this build and, where the test needs them, of a
# double-precision build (`-D BH_DOUBLE_ENTITY=/path/to/entity.xc`) and of a
# Serial-backend build (`-D BH_SERIAL_ENTITY=/path/to/entity.xc`); a script
# whose other build is not given exits with 77 and ctest reports it skipped.
# The comparisons are done by the host-only tool `bh_check` (C++, double).
# ------------------------------

set(BH_DOUBLE_ENTITY
    ""
    CACHE FILEPATH "entity.xc of a double-precision bh_proto build (T5b cross-build)")
set(BH_SERIAL_ENTITY
    ""
    CACHE FILEPATH "entity.xc of a Serial-backend bh_proto build (T8 bit identity)")

# host-only tools and the parameter-reader test: no Kokkos, no ADIOS2
add_executable(bh_check bh_check.cpp)
target_include_directories(bh_check PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET bh_check PROPERTY LINK_LIBRARIES "")

add_executable(test-hadronic-bh_proto_params.xc bh_proto_params.cpp)
target_include_directories(test-hadronic-bh_proto_params.xc
                           PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET test-hadronic-bh_proto_params.xc PROPERTY LINK_LIBRARIES "")
add_test(NAME "HADRONIC::bh_proto_params" COMMAND test-hadronic-bh_proto_params.xc
                                              ${CMAKE_SOURCE_DIR}/pgens/bh_proto/inputs/mpi)

if("${pgen}" STREQUAL "bh_proto")
  set(BH_SINGLE_ENTITY $<TARGET_FILE:entity.xc>)
  # step 5 round 2 (F1, 2026-10-30): the test-only engine binary. Same translation unit as entity.xc
  # (src/entity.cpp, which includes pgen.hpp), compiled with BH_TESTING defined here and nowhere else:
  # it alone can skip the a_max refusal of the event mode (environment BH_TESTING_BYPASS_AMAX=1), which
  # the on_bound run-level test (E8 (i)) needs. The production target entity.xc compiles no such path.
  add_executable(entity_bh_testing.xc ${CMAKE_SOURCE_DIR}/src/entity.cpp)
  target_link_libraries(entity_bh_testing.xc PUBLIC ntt_global ntt_framework ntt_metrics ntt_engines ntt_pgen)
  target_compile_definitions(entity_bh_testing.xc PRIVATE BH_TESTING=1)
  set(BH_TESTING_ENTITY $<TARGET_FILE:entity_bh_testing.xc>)
  set(BH_INPUTS ${CMAKE_SOURCE_DIR}/pgens/bh_proto/inputs)
  set(BH_WORK ${CMAKE_CURRENT_BINARY_DIR}/work)
  set(BH_SCRIPTS ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(NAME "HADRONIC::reject"
           COMMAND bash ${BH_SCRIPTS}/reject.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/reject)
  add_test(NAME "HADRONIC::t7a_fields_B"
           COMMAND bash ${BH_SCRIPTS}/t7a.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t7a)
  add_test(NAME "HADRONIC::t7b_fields_E"
           COMMAND bash ${BH_SCRIPTS}/t7b.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t7b)
  add_test(NAME "HADRONIC::t5b_cross_build"
           COMMAND bash ${BH_SCRIPTS}/t5b_cross.sh ${BH_SINGLE_ENTITY} "${BH_DOUBLE_ENTITY}"
                   $<TARGET_FILE:bh_check> ${BH_INPUTS} ${BH_WORK}/t5b)
  add_test(NAME "HADRONIC::t8_bit_identity"
           COMMAND bash ${BH_SCRIPTS}/t8.sh "${BH_SERIAL_ENTITY}" $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t8)
  # step 3: pair plumbing
  add_test(NAME "HADRONIC::t12_capacity"
           COMMAND bash ${BH_SCRIPTS}/t12.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t12)
  add_test(NAME "HADRONIC::t12b_estimate"
           COMMAND bash ${BH_SCRIPTS}/t12b.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t12b)
  # step 4 round 2 (finding 5): the sampler counters reach the ledger file (non-MPI code path)
  add_test(NAME "HADRONIC::sampler_overflow"
           COMMAND bash ${BH_SCRIPTS}/sampler_overflow.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/sampler_overflow)
  add_test(NAME "HADRONIC::t13_gauss"
           COMMAND bash ${BH_SCRIPTS}/t13.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t13)
  add_test(NAME "HADRONIC::t10_poisson_determinism"
           COMMAND bash ${BH_SCRIPTS}/t10det.sh "${BH_SERIAL_ENTITY}" ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t10det)
  add_test(NAME "HADRONIC::texp_exported_debt"
           COMMAND bash ${BH_SCRIPTS}/texp.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/texp)
  add_test(NAME "HADRONIC::r2_tiny_pair_weight"
           COMMAND bash ${BH_SCRIPTS}/r2_tiny.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/r2_tiny)
  # round 2: production placement of the stage (the one core change) against the post-step placement
  add_test(NAME "HADRONIC::t14b_placement"
           COMMAND bash ${BH_SCRIPTS}/t14b_placement.sh ${BH_SINGLE_ENTITY} "${BH_SERIAL_ENTITY}" $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/t14b)
  # step 5 (event mode): E8 at the run level, the host-side policies on_bound, on_capacity, on_max_events
  add_test(NAME "HADRONIC::event_e8"
           COMMAND bash ${BH_SCRIPTS}/event_e8.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/event_e8 ${BH_TESTING_ENTITY})
  set_tests_properties("HADRONIC::event_e8" PROPERTIES TIMEOUT 1800)
  # step 5: the restart refusal (a continuous-mode checkpoint resumed in the event mode) and its controls
  add_test(NAME "HADRONIC::event_restart"
           COMMAND bash ${BH_SCRIPTS}/event_restart.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/event_restart)
  set_tests_properties("HADRONIC::event_restart" PROPERTIES TIMEOUT 1800)
  # step 7 (scaling harness, contract 2026-10-30): the Mac smoke of the perf driver at 2^16 ions, every mode and
  # variant plus the M5 reference runs, one repeat, 20 warm-up and 100 measured steps (numbers labelled as a smoke)
  add_test(NAME "HADRONIC::perf_smoke"
           COMMAND ${CMAKE_COMMAND} -E env PERF_N=65536 PERF_REPEATS=1 PERF_WARMUP=20 PERF_MEASURED=100 PERF_DIAG=10
                   PERF_M5=1 PERF_LABEL=ctest-smoke
                   bash ${BH_SCRIPTS}/perf/perf_driver.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_SCRIPTS}/perf/inputs ${BH_WORK}/perf_smoke)
  set_tests_properties("HADRONIC::perf_smoke" PROPERTIES TIMEOUT 3600)
  set_tests_properties("HADRONIC::t5b_cross_build" "HADRONIC::t8_bit_identity" "HADRONIC::t10_poisson_determinism"
                       PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties("HADRONIC::t12_capacity" "HADRONIC::t12b_estimate" "HADRONIC::t13_gauss"
                       "HADRONIC::t10_poisson_determinism" "HADRONIC::texp_exported_debt" "HADRONIC::t14b_placement" "HADRONIC::r2_tiny_pair_weight"
                       PROPERTIES TIMEOUT 1800)
  set_tests_properties("HADRONIC::t5b_cross_build" PROPERTIES TIMEOUT 7200)
  set_tests_properties("HADRONIC::t7a_fields_B" "HADRONIC::t7b_fields_E" "HADRONIC::t8_bit_identity"
                       PROPERTIES TIMEOUT 1800)
endif()
