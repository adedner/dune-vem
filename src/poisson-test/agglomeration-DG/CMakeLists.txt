add_subdirectory( data )

add_executable( agglo-dg EXCLUDE_FROM_ALL agglo-dg.cc )
target_link_libraries( agglo-dg dunevem )
target_link_dune_default_libraries( agglo-dg )
