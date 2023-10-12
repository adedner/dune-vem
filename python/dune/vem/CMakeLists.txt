add_subdirectory(tutorial)
add_python_targets(vem
  __init__
  __main__
  vem
  voronoi
  min_bounding_rect
  qhull_2d
  bbox
  patch
)
dune_symlink_to_source_files(FILES earcut.hpp)
install(FILES earcut.hpp)
