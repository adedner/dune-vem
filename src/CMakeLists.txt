add_executable( test-gmsh EXCLUDE_FROM_ALL test-gmsh.cc )
target_link_libraries( test-gmsh dunevem )
target_link_dune_default_libraries( test-gmsh )
