cmake_minimum_required(VERSION 3.16)

project(dune-vem CXX C)

#find dune-common and set the module path
find_package(dune-common)
list(APPEND CMAKE_MODULE_PATH ${dune-common_MODULE_PATH}
     "${CMAKE_SOURCE_DIR}/cmake/modules")# make sure our own modules are found

#include the dune macros
include(DuneMacros)

# deactivate global include-directories
dune_policy(SET DP_DEFAULT_INCLUDE_DIRS NEW)

# do not make suggested dependencies required automatically
dune_policy(SET DP_SUGGESTED_MODULE_DEPENDENCIES_REQUIRED_DOWNSTREAM NEW)

# start a dune project with information from dune.module
dune_project()

# add all package flags to all executables in all subdirs
dune_enable_all_packages()

include(AddMPIFlags)

add_subdirectory("dune")
add_subdirectory("cmake/modules")
add_subdirectory("pydemo")
add_subdirectory("python")

# create library target and export it as Dune::Vem
dune_add_library(dunevem INTERFACE
  EXPORT_NAME Vem
  LINK_LIBRARIES ${DUNE_LIBS})

# finalize the dune project, e.g. generating config.h etc.
finalize_dune_project(GENERATE_CONFIG_H_CMAKE)
