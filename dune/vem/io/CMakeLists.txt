install(
  FILES gmsh.hh
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/io
)

dune_add_library(io OBJECT gmsh.cc)
