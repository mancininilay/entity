# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Generates tests for the `ntt_kernels` module
#
# @uses:
#
# * kokkos [required]
# * plog [required]
# * mpi [optional]
# ------------------------------

set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)

function(gen_test title)
  set(exec test-kernels-${title}.xc)
  set(src ${title}.cpp)
  add_executable(${exec} ${src})

  set(libs ntt_kernels ntt_metrics ntt_framework)
  add_dependencies(${exec} ${libs})
  target_link_libraries(${exec} PRIVATE ${libs})

  add_test(NAME "KERNELS::${title}" COMMAND "${exec}")
endfunction()

gen_test(faraday_mink)
gen_test(ampere_mink)
gen_test(deposit)
if(${tiled_deposit})
  gen_test(deposit_tiled)
endif()
gen_test(digital_filter)
gen_test(particle_moments)
gen_test(fields_to_phys)
gen_test(prtls_to_phys)
gen_test(gca_pusher)
gen_test(prtl_bc)
gen_test(flds_bc)
gen_test(pusher)
gen_test(ext_force)
gen_test(reduced_stats)
gen_test(custom_emission)
gen_test(custom_prtl_update)
# Bethe-Heitler loss-only kernel (P2 step 2): one ctest entry per pre-registered
# group so that each verdict is visible; the executable runs every group when
# called without arguments
add_executable(test-kernels-bh_loss.xc bh_loss.cpp)
add_dependencies(test-kernels-bh_loss.xc ntt_kernels ntt_metrics ntt_framework)
target_link_libraries(test-kernels-bh_loss.xc PRIVATE ntt_kernels ntt_metrics ntt_framework)
foreach(grp tables t9 tags above subcycling t6 t5 t5b)
  add_test(NAME "KERNELS::bh_loss_${grp}" COMMAND test-kernels-bh_loss.xc ${grp})
endforeach()

if("${pgen}" STREQUAL "rec_mix")
  gen_test(rec_mix_segments)
endif()
