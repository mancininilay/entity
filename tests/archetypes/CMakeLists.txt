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

  set(libs ntt_archetypes ntt_framework ntt_global ntt_metrics)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  add_test(NAME "ARCHETYPES::${title}" COMMAND "${exec}")
endfunction()

gen_test(energy_dist)
gen_test(spatial_dist)
gen_test(field_setter)
gen_test(powerlaw)
gen_test(pgen)
gen_test(qed_compton)
gen_test(bethe_heitler)

# reproduction of an upstream issue (expected to fail): built, not registered as a test
add_executable(repro-emission-overflow.xc repro_emission_overflow.cpp)
target_link_libraries(repro-emission-overflow.xc PRIVATE ntt_archetypes ntt_framework ntt_global ntt_metrics)

# replay of the random streams of a pool through the policy (analysis tool): built, not registered
add_executable(bhv-replay-pool.xc bhv_replay_pool.cpp)
target_link_libraries(bhv-replay-pool.xc PRIVATE ntt_archetypes ntt_framework ntt_global ntt_metrics)
