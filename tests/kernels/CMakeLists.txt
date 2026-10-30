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
# [fork:hadronic/P9] every real payload slot of an injected particle is zero (T10b, kernel level)
gen_test(injectors_payloads)
# [fork:hadronic/P5] weighted reduced particle statistics against a direct double sum (T-P5)
gen_test(reduced_stats_weights)
# Bethe-Heitler loss-only kernel (P2 step 2): one ctest entry per pre-registered
# group so that each verdict is visible; the executable runs every group when
# called without arguments
add_executable(test-kernels-bh_loss.xc bh_loss.cpp)
add_dependencies(test-kernels-bh_loss.xc ntt_kernels ntt_metrics ntt_framework)
target_link_libraries(test-kernels-bh_loss.xc PRIVATE ntt_kernels ntt_metrics ntt_framework)
foreach(grp tables overflow huge t9 tags above subcycling t6 t5 t5fine t5b)
  add_test(NAME "KERNELS::bh_loss_${grp}" COMMAND test-kernels-bh_loss.xc ${grp})
endforeach()
# the two long groups (T5 dt-convergence at three resolutions, T5b ulp scan) get
# their own timeouts (seconds); every other group keeps the ctest default
set_tests_properties("KERNELS::bh_loss_t5fine" PROPERTIES TIMEOUT 7200)
set_tests_properties("KERNELS::bh_loss_t5b" PROPERTIES TIMEOUT 7200)
# [fork:hadronic/P2 step 6] the same groups on the Born rate tables (BH_QK_SOURCE=born, contract 1.5 stage 3)
foreach(grp tables overflow huge t9 tags above subcycling t6 t5 t5fine t5b)
  add_test(NAME "KERNELS::bh_loss_${grp}_born" COMMAND test-kernels-bh_loss.xc ${grp})
  set_tests_properties("KERNELS::bh_loss_${grp}_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born")
endforeach()
set_tests_properties("KERNELS::bh_loss_t5fine_born" "KERNELS::bh_loss_t5b_born" PROPERTIES TIMEOUT 7200)
# Bethe-Heitler pair plumbing (P2 step 3): T10 (budget, capped, closing), poisson,
# none, pairs per unit energy, T12, T10b, T-exp; one ctest entry per group
add_executable(test-kernels-bh_pairs.xc bh_pairs.cpp)
add_dependencies(test-kernels-bh_pairs.xc ntt_kernels ntt_metrics ntt_framework)
target_link_libraries(test-kernels-bh_pairs.xc PRIVATE ntt_kernels ntt_metrics ntt_framework)
foreach(grp t10 poisson none energy momentum t12 t10b texp r1 r2 r3 cmpi1)
  add_test(NAME "KERNELS::bh_pairs_${grp}" COMMAND test-kernels-bh_pairs.xc ${grp})
endforeach()
# [fork:hadronic/P2 step 6] on the Born rate tables
foreach(grp t10 poisson none energy momentum t12 t10b texp r1 r2 r3 cmpi1)
  add_test(NAME "KERNELS::bh_pairs_${grp}_born" COMMAND test-kernels-bh_pairs.xc ${grp})
  set_tests_properties("KERNELS::bh_pairs_${grp}_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born")
endforeach()

if("${pgen}" STREQUAL "rec_mix")
  gen_test(rec_mix_segments)
endif()
# Bethe-Heitler rest-frame sampler in the kernel (P2 step 4, stage 4): device
# tables, A11 (mu_d independence), A12 (cost, number and energy per paid energy,
# the ledger identity, the pair spectrum vs the host chain); one entry per group
add_executable(test-kernels-bh_sampler.xc bh_sampler.cpp)
add_dependencies(test-kernels-bh_sampler.xc ntt_kernels ntt_metrics ntt_framework)
target_link_libraries(test-kernels-bh_sampler.xc PRIVATE ntt_kernels ntt_metrics ntt_framework)
target_compile_definitions(test-kernels-bh_sampler.xc PRIVATE BH_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data/hadronic")
foreach(grp tables a11 identity a12 spectrum)
  add_test(NAME "KERNELS::bh_sampler_${grp}_born" COMMAND test-kernels-bh_sampler.xc ${grp})
  set_tests_properties("KERNELS::bh_sampler_${grp}_born" PROPERTIES ENVIRONMENT "BH_QK_SOURCE=born")
endforeach()
set_tests_properties("KERNELS::bh_sampler_a12_born" PROPERTIES TIMEOUT 14400)
foreach(grp tables a11 identity a12 spectrum)
  add_test(NAME "KERNELS::bh_sampler_${grp}" COMMAND test-kernels-bh_sampler.xc ${grp})
endforeach()
set_tests_properties("KERNELS::bh_sampler_a12" PROPERTIES TIMEOUT 14400)
# Bethe-Heitler ledger reduction timing harness (P2 step 4, stage 5; G3 open item):
# the double-double reducer vs Kokkos::atomic_add on double, a diagnostic
gen_test(bh_reduction_timing)
