add_subdirectory( data )
#
add_executable( vem-poisson-pmesh EXCLUDE_FROM_ALL vem-poisson-pmesh.cc )
target_link_libraries( vem-poisson-pmesh dunevem )
target_link_dune_default_libraries( vem-poisson-pmesh )
