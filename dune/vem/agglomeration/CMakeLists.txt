set( HEADERS
  agglomeration.hh
  boundingbox.hh
  indexset.hh
)

install( FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/agglomeration )
