add_subdirectory( constraints )


set(HEADERS
  diffusionmodel.hh
  mass.hh
  stabmatrix.hh
  vemdirichletconstraints.hh
  vemelliptic.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/operator)
