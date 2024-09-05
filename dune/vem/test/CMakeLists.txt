set( SCRIPTS curlfree          divfree
             hk_laplace        perturbation
             varyingcoeff      uzawa )
add_python_targets(testscripts ${SCRIPTS})
foreach( script ${SCRIPTS} )
  dune_python_add_test(NAME ${script}
                       SCRIPT ${script}.py
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       LABELS quick)
endforeach()

if( ${FEM_TORTURE_TESTS} )
  set( SCRIPTS_TORTURE hk uzawa )
  add_python_targets(testscripts ${SCRIPTS_TORTURE})
  foreach( script ${SCRIPTS_TORTURE} )
    dune_python_add_test(NAME ${script}
                         SCRIPT ${script}.py
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endforeach()
endif()
