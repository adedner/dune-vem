set(HEADERS
  basisfunctionset.hh
  curlfree.hh
  default.hh
  divfree.hh
  indexset.hh
  interpolate.hh
  hk.hh
  test.hh
)

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/space)
