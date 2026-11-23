# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Generates tests for the `ntt_framework` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * mpi [optional]
# * adios2 [optional]
#
# !TODO:
#
# * add tests for mesh separately
# * add test for 3D metadomain
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title is_parallel)
  set(exec test-framework-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_framework ntt_global ntt_metrics)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  if(${is_parallel})
    add_test(NAME "FRAMEWORK::${title}"
             COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
                     "${exec}")
  else()
    add_test(NAME "FRAMEWORK::${title}" COMMAND "${exec}")
  endif()
endfunction()

gen_test(parameters false)
if(${mpi})
  gen_test(particles true)
  gen_test(particles_sort true)
else()
  gen_test(particles false)
  gen_test(particles_sort false)
endif()

# tiled_deposit X-3: per-backend sort_by_key permutation test (only built when
# the compile-time tiled_deposit toggle is on).
if(${tiled_deposit})
  gen_test(sort_by_key false)
endif()

gen_test(fields false)
gen_test(grid_mesh false)
if(${DEBUG})
  gen_test(metadomain false)
endif()

if(${mpi})
  gen_test(comm-mpi true)

  # the capacity test of the particle exchange: three cases that must pass on 1
  # and on 2 ranks, and one overflow on 2 ranks that must stop the engine
  set(cap_exec test-framework-comm-capacity.xc)
  add_executable(${cap_exec} comm-capacity.cpp)
  set(cap_libs ntt_framework ntt_global ntt_metrics)
  add_dependencies(${cap_exec} ${cap_libs})
  target_link_libraries(${cap_exec} PRIVATE ${cap_libs})
  foreach(cap_case full fill holes)
    foreach(cap_np 1 2)
      add_test(NAME "FRAMEWORK::comm-capacity-${cap_case}-${cap_np}"
               COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}"
                       "${cap_np}" "$<TARGET_FILE:${cap_exec}>" "${cap_case}")
    endforeach()
  endforeach()
  add_test(NAME "FRAMEWORK::comm-capacity-over-2"
           COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" "2"
                   "$<TARGET_FILE:${cap_exec}>" "over")
  set_tests_properties(
    "FRAMEWORK::comm-capacity-over-2"
    PROPERTIES PASS_REGULAR_EXPRESSION "Too many particles to receive"
               FAIL_REGULAR_EXPRESSION "OVERFLOW NOT DETECTED")
else()
  gen_test(comm-nompi false)
endif()

if(${output})
  if(${mpi})
    gen_test(checkpoint-extents true)
    gen_test(checkpoint-data true)
    gen_test(spectra true)
  else()
    gen_test(checkpoint-extents false)
    gen_test(checkpoint-data false)
    gen_test(spectra false)
  endif()
endif()

# this test is only run manually to ensure ...
# ... command line args are working properly ...
# ... and that the logging is done correctly
#
# gen_test(simulation)
