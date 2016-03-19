add_subdirectory( constraints )

install(
  FILES mass.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/operator
)
