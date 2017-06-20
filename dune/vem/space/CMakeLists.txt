set(HEADERS
  agglomeration.hh
  basisfunctionset.hh
  interpolate.hh
  interpolation.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/space)
