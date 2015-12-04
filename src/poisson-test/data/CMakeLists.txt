set( DATAFILES
  mymesh.msh
  parameter
)

foreach( datafile ${DATAFILES} )
  configure_file( ${datafile} ${CMAKE_CURRENT_BINARY_DIR}/${datafile} COPYONLY )
endforeach()
