#ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationDofMapper
    // ----------------------

    template< class GridPart, class Layout >
    class AgglomerationDofMapper
    {
      typedef AgglomerationDofMapper< GridPart > ThisType;

    public:
      typedef GridPart GridPartType;

      static const int dimension = GridPartType::dimension;

      AgglomerationDofMapper ( const GridPartType &gridPart, std::vector< std::size_t > agglomerateIndex, Layout layout = Layout() );

    private:
      const GridPartType &gridPart_;
      Layout layout_;
      std::vector< std::size_t > agglomerateIndices_;
      std::vector< std::vector< std::size_t > > agglomerateConnectivity_;
    };



    // Implementation of AgglomerationDofMapper
    // ---------------------------------------

    template< class GridPart, class Layout >
    inline AgglomerationDofMapper< GridPart, Layout >
      ::AgglomerationDofMapper ( const GridPartType &gridPart, std::vector< std::size_t > agglomerateIndices, Layout layout )
      : gridPart_( gridPart ), layout_( layout ), agglomerateIndices_( std::move( agglomerateIndices ) )
    {
      typedef typename GridPart::IndexSetType::IndexType IndexType;

      const typename GridPart::IndexSetType &indexSet = gridPart_.indexSet();

      std::vector< std::vector< IndexType > subAgglomerates( GlobalGeometryTypeIndex::size( dimension-1 ) );

      // find subagglomerates

      for( const auto element : elements( gridPart, Partitions::interiorBorder ) )
      {
        const std::size_t agIndex = agglomerateIndices_[ indexSet.index( element ) ];
        const auto &refElement = ReferenceElements< typename GridPart::ctype, dimension >::general( element.type() );
        for( const auto intersection : intersections( gridPart, element ) )
        {
          assert( intersection.conforming() );
          if( !intersection.neighbor() || (agglomerateIndices_[ indexSet.index( intersection.outside() ) ] != agIndex) )
          {
            const int face = intersection.indexInInside();
            for( int codim = 1; codim <= dimension; ++codim )
            {
              const int numSubEntities = refElement.size( face, 1, codim );
              for( int i = 0; i < numSubEntities; ++i )
              {
                const int k = refElement.subentity( face, 1, i, codim );
                const std::size_t typeIndex = GlobalGeometryTypeIndex::index( refElement.type( k, codim ) );
                subAgglomerates[ typeIndex ].push_back( indexSet.subIndex( element, k, codim ) );
              }
            }
          }
        }
      }

      // make subagglomerates unique

      for( auto &list : subAgglomerates )
      {
        std::sort( list.begin(), list.end() );
        list.erase( std::unique( list.begin(), list.end() ), list.end() );
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
