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
