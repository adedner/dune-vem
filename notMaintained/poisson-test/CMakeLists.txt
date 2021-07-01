add_subdirectory( data )
add_subdirectory( agglomeration-DG )
add_subdirectory( vem-test )

add_executable( dune-poisson-agglo-DG EXCLUDE_FROM_ALL dune-poisson-agglo-DG.cc )
target_link_libraries( dune-poisson-agglo-DG dunevem )
target_link_dune_default_libraries( dune-poisson-agglo-DG )

add_executable( trivial-agglomeration EXCLUDE_FROM_ALL trivial-agglomeration.cc )
target_link_libraries( trivial-agglomeration dunevem )
target_link_dune_default_libraries( trivial-agglomeration )
