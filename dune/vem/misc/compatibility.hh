#ifndef DUNE_VEM_MISC_COMPATIBILITY_HH
#define DUNE_VEM_MISC_COMPATIBILITY_HH

#include <utility>

#include <dune/grid/common/entity.hh>
#if !DUNE_VERSION_NEWER(DUNE_GRID, 2, 6)
#include <dune/grid/common/entitypointer.hh>
#endif // #if !DUNE_VERSION_NEWER(DUNE_GRID, 2, 6)

namespace Dune
{

  namespace Vem
  {

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_MISC_COMPATIBILITY_HH
