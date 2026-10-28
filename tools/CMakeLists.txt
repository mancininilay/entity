# cmake-lint: disable=C0103,C0111
# ------------------------------
# @brief: Host-side tools built alongside the code
#
# * bh_tables_dump: Bethe-Heitler startup tables in the exchange-file format
# ------------------------------

add_executable(bh_tables_dump bh_tables_dump.cpp)
target_include_directories(bh_tables_dump PRIVATE ${CMAKE_SOURCE_DIR}/src)

# the tool uses only the standard library: drop the directory-wide link
# dependencies (Kokkos, ADIOS2) inherited from the top-level link_libraries()
set_property(TARGET bh_tables_dump PROPERTY LINK_LIBRARIES "")

# bh_sampler_dump: the step-4 fixture files (qk_table and, in later stages,
# the sampler moments and histograms); standard library and threads only
add_executable(bh_sampler_dump bh_sampler_dump.cpp)
target_include_directories(bh_sampler_dump PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_property(TARGET bh_sampler_dump PROPERTY LINK_LIBRARIES "")
find_package(Threads REQUIRED)
target_link_libraries(bh_sampler_dump PRIVATE Threads::Threads)
