add_subdirectory( data )
add_executable( agglo-heat EXCLUDE_FROM_ALL agglo-heat.cc )
target_link_libraries( agglo-heat )
