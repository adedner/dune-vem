install(
  FILES gmsh.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/io
)

dune_add_library( dunevem
  gmsh.cc
  ADD_LIBS ${DUNE_LIBS}
)
