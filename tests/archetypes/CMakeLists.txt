# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Generates tests for the `ntt_archetypes` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title)
  set(exec test-archetypes-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_archetypes ntt_global ntt_metrics)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  add_test(NAME "ARCHETYPES::${title}" COMMAND "${exec}")
endfunction()

gen_test(energy_dist)
gen_test(spatial_dist)
gen_test(field_setter)
gen_test(powerlaw)
gen_test(pgen)
gen_test(bh_physics)
target_compile_definitions(
  test-archetypes-bh_physics.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
gen_test(bh_fixtures)
target_compile_definitions(
  test-archetypes-bh_fixtures.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
# [fork:hadronic/P2 step 4] the rest-frame laws of the pair sampler (A1 to A5)
gen_test(bh_restframe)
target_compile_definitions(
  test-archetypes-bh_restframe.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
find_package(Threads REQUIRED)
target_link_libraries(test-archetypes-bh_restframe.xc PRIVATE Threads::Threads)
# [fork:hadronic/P2 step 4] the CDF tables and the two sampling chains (accuracy
# budget, A5b, A7, A9); Monte Carlo at every rate-table node: its own timeout
gen_test(bh_chain)
target_compile_definitions(
  test-archetypes-bh_chain.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
target_link_libraries(test-archetypes-bh_chain.xc PRIVATE Threads::Threads)
set_tests_properties("ARCHETYPES::bh_chain" PROPERTIES TIMEOUT 7200)
