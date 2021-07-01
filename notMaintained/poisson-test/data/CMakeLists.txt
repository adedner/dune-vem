set( DATAFILES
  mymesh.msh
  parameter
  trivial.param
  unitcube-2d.dgf
)

foreach( datafile ${DATAFILES} )
  configure_file( ${datafile} ${CMAKE_CURRENT_BINARY_DIR}/${datafile} COPYONLY )
endforeach()
