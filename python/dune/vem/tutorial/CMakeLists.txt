# make sure the example works and run in during tests
if( ${FEM_TORTURE_TESTS} )
  dune_python_add_test(NAME pywillmore
                       SCRIPT willmore.py
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       LABELS quick)
  dune_python_add_test(NAME pyns
                       SCRIPT cylinderUzawa.py
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                       LABELS quick)
endif()
dune_python_add_test(NAME pymixed
                     SCRIPT laplace.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     LABELS quick)
dune_symlink_to_source_files(FILES willmore.py cylinderUzawa.py laplace.py mixedSolver.py uzawa.py)
