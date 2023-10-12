dune_python_add_test(NAME pyhktest
                     SCRIPT hktest.py
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     LABELS quick)
dune_symlink_to_source_files(FILES hktest.py concave.py)
