set( SCRIPTS biharmonic  curlfree     elliptic     hk
             hktest      interpolate  perturbation script )
add_python_targets(testscripts ${SCRIPTS})

dune_python_add_test(NAME script
                     SCRIPT script.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     LABELS quick)

if( ${FEM_TORTURE_TESTS} )
  dune_python_add_test(NAME pyscheme
                       SCRIPT scheme_assemble.py
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       TIMEOUT 2000
                       LABELS quick)
endif()
