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

    template< class GridPart >
    class AgglomerationDofMapper
    {
      typedef AgglomerationDofMapper< GridPart > ThisType;

      typedef typename GridPart::IndexSetType IndexSetType;

    public:
      typedef GridPart GridPartType;

      AgglomerationDofMapper ( const GridPartType &gridPart, std::vector< std::size_t > agglomerateIndex );

    private:
      const GridPartType &gridPart_;
      std::vector< std::size_t > agglomerateIndices_;
      std::vector< std::vector< std::size_t > > agglomerateConnectivity_;
    };



    // Implementation of AgglomerationDofMapper
    // ---------------------------------------

    template< class GridPart >
    inline AgglomerationDofMapper::AgglomerationDofMapper ( const GridPartType &gridPart, std::vector< std::size_t > agglomerateIndices )
      : gridPart_( gridPart ), agglomerateIndices_( std::move( agglomerateIndices ) )
    {
      const IndexSetType &indexSet = gridPart_.indexSet();

      // find faces of agglomerates

      std::vector< typename IndexSetType::IndexType > agFaces;
      for( const auto element : elements( gridPart, Partitions::interiorBorder ) )
      {
        const std::size_t agIndex = agglomerateIndices_[ indexSet.index( element ) ];
        for( const auto intersection : intersections( gridPart, element ) )
        {
          assert( intersection.conforming() );
          if( !intersection.neighbor() || (agglomerateIndices_[ indexSet.index( intersection.outside() ) ] != agIndex) )
            agFaces.push_back( indexSet.index( element.template subEntity< 1 >( intersection.indexInInside() ) ) );
        }
      }
      std::sort( agFaces.begin(), agFaces.end() );
      agFaces.erase( std::unique( agFaces.begin(), agFaces.end() ), agFaces.end() );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
