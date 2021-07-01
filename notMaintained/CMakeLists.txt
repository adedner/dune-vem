add_subdirectory( gmsh )
add_subdirectory( poisson-test )
add_subdirectory( heat-test )
add_subdirectory( nonlinear-test )
add_subdirectory( quasilinear-test )
add_subdirectory( quasilinear-fixed-point )

add_executable( test-gmsh EXCLUDE_FROM_ALL test-gmsh.cc )
target_link_libraries( test-gmsh dunevem )
target_link_dune_default_libraries( test-gmsh )

add_executable( test-dgspace EXCLUDE_FROM_ALL test-dgspace.cc )
target_link_libraries( test-dgspace dunevem )
target_link_dune_default_libraries( test-dgspace )

add_executable( test-indexset EXCLUDE_FROM_ALL test-indexset.cc )
target_link_libraries( test-indexset dunevem )
target_link_dune_default_libraries( test-indexset )

add_executable( test-reactiondiffusion EXCLUDE_FROM_ALL test-reactiondiffusion.cc )
target_link_libraries( test-reactiondiffusion dunevem )
target_link_dune_default_libraries( test-reactiondiffusion )

add_executable( test-reactiondiffusion2 EXCLUDE_FROM_ALL test-reactiondiffusion2.cc )
target_link_libraries( test-reactiondiffusion2 dunevem )
target_link_dune_default_libraries( test-reactiondiffusion2 )

add_executable( test-leastsquares EXCLUDE_FROM_ALL test-leastsquares.cc )
target_link_libraries( test-leastsquares dunevem )
target_link_dune_default_libraries( test-leastsquares )
