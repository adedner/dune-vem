add_subdirectory(dune)

dune_python_install_package(
  PATH "."
  CMAKE_METADATA_FILE dune/vem/metadata.cmake
  CMAKE_METADATA_FLAGS DUNE_OPTS_FILE
)
cmake_policy(SET CMP0087 NEW)
