add_subdirectory( constraints )


set(HEADERS
  elliptic.hh
  mass.hh
  vemdirichletconstraints.hh
  vemdirichletconstraints.old.hh
  vemelliptic.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/operator)
