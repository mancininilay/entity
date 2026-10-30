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
find_package(Threads REQUIRED)
# [fork:hadronic/P2 step 6] q_k from the Born machinery for the loss tables (A10b, A10c, A3, the
# fit-path regression of the cumulatives), against the bhref 0.3.0 candidate fixtures
gen_test(bh_born_qk)
target_compile_definitions(
  test-archetypes-bh_born_qk.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
target_link_libraries(test-archetypes-bh_born_qk.xc PRIVATE Threads::Threads)
set_tests_properties("ARCHETYPES::bh_born_qk" PROPERTIES TIMEOUT 1800)
# [fork:hadronic/P2 step 4] the rest-frame laws of the pair sampler (A1 to A5)
gen_test(bh_restframe)
target_compile_definitions(
  test-archetypes-bh_restframe.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
target_link_libraries(test-archetypes-bh_restframe.xc PRIVATE Threads::Threads)
# [fork:hadronic/P2 step 4] the CDF tables and the two sampling chains (accuracy
# budget, A5b, A7, A9); Monte Carlo at every rate-table node: its own timeout
add_executable(test-archetypes-bh_chain.xc bh_chain.cpp)
add_dependencies(test-archetypes-bh_chain.xc ntt_archetypes ntt_global ntt_metrics)
target_link_libraries(test-archetypes-bh_chain.xc PRIVATE ntt_archetypes ntt_global ntt_metrics Threads::Threads)
target_compile_definitions(
  test-archetypes-bh_chain.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
foreach(grp epsbias budget kinematics efficiency a9 a7w a7wall a7)
  add_test(NAME "ARCHETYPES::bh_chain_${grp}" COMMAND test-archetypes-bh_chain.xc ${grp})
endforeach()
set_tests_properties("ARCHETYPES::bh_chain_a7" PROPERTIES TIMEOUT 7200)
set_tests_properties("ARCHETYPES::bh_chain_a7w" PROPERTIES TIMEOUT 3600)
set_tests_properties("ARCHETYPES::bh_chain_a9" PROPERTIES TIMEOUT 3600)
# [fork:hadronic/P2 step 6 round 2, Codex 1 F2] the weighted chain at every node with exact N = 1e6: ~2e10
# proposals, the long group (run separately, log kept)
set_tests_properties("ARCHETYPES::bh_chain_a7wall" PROPERTIES TIMEOUT 86400)
# [fork:hadronic/P2 step 6] the same against the Born rate tables (contract 1.5: A5-numerical in the budget
# group, A7 at 4 s.e. with no floor); the step-1 suite on the Born tables likewise
foreach(grp budget a7w a7wall a7)
  add_test(NAME "ARCHETYPES::bh_chain_${grp}_born" COMMAND test-archetypes-bh_chain.xc ${grp})
  set_tests_properties("ARCHETYPES::bh_chain_${grp}_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born")
endforeach()
set_tests_properties("ARCHETYPES::bh_chain_a7_born" PROPERTIES TIMEOUT 7200)
set_tests_properties("ARCHETYPES::bh_chain_a7w_born" PROPERTIES TIMEOUT 3600)
set_tests_properties("ARCHETYPES::bh_chain_a7wall_born" PROPERTIES TIMEOUT 86400)
set_tests_properties("ARCHETYPES::bh_chain_budget_born" PROPERTIES TIMEOUT 3600)
add_test(NAME "ARCHETYPES::bh_physics_born" COMMAND test-archetypes-bh_physics.xc)
set_tests_properties("ARCHETYPES::bh_physics_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born" TIMEOUT 1800)
# [fork:hadronic/P2 step 4] the UR lab electron spectrum anchor (A8)
gen_test(bh_ka2008)
target_compile_definitions(
  test-archetypes-bh_ka2008.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
target_link_libraries(test-archetypes-bh_ka2008.xc PRIVATE Threads::Threads)
set_tests_properties("ARCHETYPES::bh_ka2008" PROPERTIES TIMEOUT 1800)
# [fork:hadronic/P2 step 6 round 2, Codex 1 F9] the same consumer on the Born loss tables
add_test(NAME "ARCHETYPES::bh_ka2008_born" COMMAND test-archetypes-bh_ka2008.xc)
set_tests_properties("ARCHETYPES::bh_ka2008_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born" TIMEOUT 1800)
# [fork:hadronic/P2 step 4, round 2] contract A10 inside ctest: the C++ sampler
# fixtures against the reference's (bhref 0.2.0 (frozen)), file against file
gen_test(bh_sampler_fixtures)
target_compile_definitions(
  test-archetypes-bh_sampler_fixtures.xc
  PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
target_link_libraries(test-archetypes-bh_sampler_fixtures.xc PRIVATE Threads::Threads)
set_tests_properties("ARCHETYPES::bh_sampler_fixtures" PROPERTIES TIMEOUT 1800)
# [fork:hadronic/P2 step 6 round 2, Codex 1 F9] the same consumer on the Born loss tables
add_test(NAME "ARCHETYPES::bh_sampler_fixtures_born" COMMAND test-archetypes-bh_sampler_fixtures.xc)
set_tests_properties("ARCHETYPES::bh_sampler_fixtures_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born" TIMEOUT 1800)
