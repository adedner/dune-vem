#ifndef DUNE_VEM_MISC_COMPATIBILITY_HH
#define DUNE_VEM_MISC_COMPATIBILITY_HH

#include <utility>

#include <dune/grid/common/entity.hh>
// #include <dune/grid/common/entitypointer.hh>

namespace Dune
{

  namespace Vem
  {

    // make_entity
    // -----------

#if 0
    template< class Grid, class Implementation >
    typename Dune::EntityPointer< Grid, Implementation >::Entity
    make_entity ( const Dune::EntityPointer< Grid, Implementation > &entityPointer )
    {
      return *entityPointer;
    }

    template< int codim, int dim, class Grid, template< int, int, class > class Implementation >
    typename Dune::Entity< codim, dim, Grid, Implementation >
    make_entity ( Dune::Entity< codim, dim, Grid, Implementation > entity )
    {
      return std::move( entity );
    }
#endif
  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_MISC_COMPATIBILITY_HH
