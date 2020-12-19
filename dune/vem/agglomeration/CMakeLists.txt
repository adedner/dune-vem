set(HEADERS
  agglomeration.hh
  boundingbox.hh
  dgspace.hh
  functor.hh
  shapefunctionset.hh
  basisfunctionset.hh
  dgmapper.hh
  dofmapper.hh
  indexset.hh
)



install( FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/vem/agglomeration )
