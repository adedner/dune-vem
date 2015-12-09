set( HEADERS
  agglomeration.hh
  basisfunctionset.hh
  boundingbox.hh
  dgmapper.hh
  dgspace.hh
  functor.hh
  indexset.hh
)

install( FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/agglomeration )
