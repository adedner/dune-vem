add_subdirectory( data )

add_executable( dune-poisson-agglo-DG EXCLUDE_FROM_ALL dune-poisson-agglo-DG.cc )
target_link_libraries( dune-poisson-agglo-DG dunevem )
target_link_dune_default_libraries( dune-poisson-agglo-DG )
