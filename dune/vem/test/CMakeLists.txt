set( SCRIPTS biharmonic  curlfree     elliptic     hk     divfree
             hktest      interpolate  perturbation script )
add_python_targets(testscripts ${SCRIPTS})

foreach( script ${SCRIPTS} )
  dune_python_add_test(NAME ${script}
                       SCRIPT ${script}.py
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       LABELS quick)
endforeach()
