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
  set_tests_properties("HADRONIC::t5b_cross_build" "HADRONIC::t8_bit_identity" "HADRONIC::t10_poisson_determinism"
                       PROPERTIES SKIP_RETURN_CODE 77)
  set_tests_properties("HADRONIC::t12_capacity" "HADRONIC::t12b_estimate" "HADRONIC::t13_gauss"
                       "HADRONIC::t10_poisson_determinism" "HADRONIC::texp_exported_debt" "HADRONIC::t14b_placement" "HADRONIC::r2_tiny_pair_weight"
                       PROPERTIES TIMEOUT 1800)
  set_tests_properties("HADRONIC::t5b_cross_build" PROPERTIES TIMEOUT 7200)
  set_tests_properties("HADRONIC::t7a_fields_B" "HADRONIC::t7b_fields_E" "HADRONIC::t8_bit_identity"
                       PROPERTIES TIMEOUT 1800)
endif()
