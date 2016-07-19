add_executable( fem-helmholtz EXCLUDE_FROM_ALL fem-helmholtz.cc )
target_link_libraries(fem-helmholtz )
add_executable( vem-helmholtz EXCLUDE_FROM_ALL vem-helmholtz.cc )
target_link_libraries( vem-helmholtz )
