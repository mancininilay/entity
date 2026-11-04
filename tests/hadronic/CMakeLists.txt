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
  # [fork:hadronic/P2 closing round, Codex 1's F1] the pair dump under bounds-checked host views: the method text is copied out of
  # pgen.hpp at build time, the frozen method of 22449c32 is the negative control; host only (no device views are involved)
  if(NOT DEVICE_ENABLED)
    set(BH_DUMP_METHOD ${CMAKE_CURRENT_BINARY_DIR}/dump_bounds_method.inc)
    add_custom_command(OUTPUT ${BH_DUMP_METHOD}
                       COMMAND python3 ${BH_SCRIPTS}/dump_bounds_extract.py ${CMAKE_SOURCE_DIR}/pgens/bh_proto/pgen.hpp ${BH_DUMP_METHOD}
                       DEPENDS ${CMAKE_SOURCE_DIR}/pgens/bh_proto/pgen.hpp ${BH_SCRIPTS}/dump_bounds_extract.py
                       VERBATIM)
    add_executable(test-hadronic-dump_bounds.xc dump_bounds.cpp ${BH_DUMP_METHOD})
    target_include_directories(test-hadronic-dump_bounds.xc PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    add_executable(test-hadronic-dump_bounds_old.xc dump_bounds.cpp)
    target_compile_definitions(test-hadronic-dump_bounds_old.xc PRIVATE BH_DUMP_OLD_METHOD)
    foreach(exe test-hadronic-dump_bounds.xc test-hadronic-dump_bounds_old.xc)
      target_include_directories(${exe} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${exe} PRIVATE ntt_global)
    endforeach()
    add_test(NAME "HADRONIC::dump_bounds"
             COMMAND bash ${BH_SCRIPTS}/dump_bounds.sh $<TARGET_FILE:test-hadronic-dump_bounds.xc>
                     $<TARGET_FILE:test-hadronic-dump_bounds_old.xc> ${BH_WORK}/dump_bounds)
  endif()
  # [fork:hadronic/P2 closing round, Codex 1's D2] the array closure and the pair dump species by species after MPI migration:
  # restframe periodic box on 1, 2 and 4 ranks in an MPI build (1 rank otherwise), the old index-paired sum as a negative
  # control, the delta control, the absorbing box with lone survivors; python3 reads the complete dumps
  if(${mpi})
    set(BH_ARRAYS_RANKS "1 2 4")
  else()
    set(BH_ARRAYS_RANKS "1")
  endif()
  add_test(NAME "HADRONIC::arrays_species"
           COMMAND bash ${BH_SCRIPTS}/arrays_species.sh ${BH_SINGLE_ENTITY} ${BH_INPUTS}/mpi ${BH_WORK}/arrays_species
                   ${precision} "${BH_ARRAYS_RANKS}" ${MPIEXEC_EXECUTABLE})
  set_tests_properties("HADRONIC::arrays_species" PROPERTIES TIMEOUT 1800 ENVIRONMENT "OMP_NUM_THREADS=1")
  # step 5: the restart refusal (a continuous-mode checkpoint resumed in the event mode) and its controls
  add_test(NAME "HADRONIC::event_restart"
           COMMAND bash ${BH_SCRIPTS}/event_restart.sh ${BH_SINGLE_ENTITY} $<TARGET_FILE:bh_check>
                   ${BH_INPUTS} ${BH_WORK}/event_restart)
  set_tests_properties("HADRONIC::event_restart" PROPERTIES TIMEOUT 1800)
  set_tests_properties("HADRONIC::t5b_cross_build" "HADRONIC::t8_bit_identity" "HADRONIC::t10_poisson_determinism"
                       PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties("HADRONIC::t12_capacity" "HADRONIC::t12b_estimate" "HADRONIC::t13_gauss"
                       "HADRONIC::t10_poisson_determinism" "HADRONIC::texp_exported_debt" "HADRONIC::t14b_placement" "HADRONIC::r2_tiny_pair_weight"
                       PROPERTIES TIMEOUT 1800)
  set_tests_properties("HADRONIC::t5b_cross_build" PROPERTIES TIMEOUT 7200)
  set_tests_properties("HADRONIC::t7a_fields_B" "HADRONIC::t7b_fields_E" "HADRONIC::t8_bit_identity"
                       PROPERTIES TIMEOUT 1800)
endif()
