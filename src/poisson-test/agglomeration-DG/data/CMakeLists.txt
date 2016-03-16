set( DATAFILES
  agglomeration-parameter
  partitioned-mesh.msh
)

foreach( datafile ${DATAFILES} )
  configure_file( ${datafile} ${CMAKE_CURRENT_BINARY_DIR} COPYONLY )
endforeach()
