set( GRIDFILES
  unit_sq_4x4_npart=4.msh
  unit_sq_8x8_npart=16.msh
  unit_sq_8x8_npart=2.msh
  unit_sq_8x8_npart=4.msh
  unitsq-tria-unstr_npart=16.msh
)

foreach( gridfile ${GRIDFILES} )
  configure_file( ${gridfile} ${CMAKE_CURRENT_BINARY_DIR}/${gridfile} COPYONLY )
endforeach()
