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
      std::vector< std::array< std::size_t, dimension+1 > > agglomerateOffset_;
      std::vector< std::size_t > connectivity_;
    };



    // Implementation of AgglomerationDofMapper
    // ----------------------------------------

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

      // build connectivity

      const std::size_t numAgglomerates = *std::max_element( agglomerateIndices_.begin(), agglomerateIndices_.end() ) + 1u;
      std::vector< std::array< std::vector< std::size_t >, dimension > > connectivity( numAgglomerates );

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
              std::vector< std::size_t > &list = connectivity[ agIndex ][ dimension - codim ];
              const int numSubEntities = refElement.size( face, 1, codim );
              for( int i = 0; i < numSubEntities; ++i )
              {
                const int k = refElement.subentity( face, 1, i, codim );
                const auto &subAgs = subAgglomerates[ GlobalGeometryTypeIndex::index( refElement.type( k, codim ) ) ];
                const auto pos = std::lower_bound( subAgs.begin(), subAgs.end(), indexSet.subIndex( element, k, codim ) );
                assert( (pos != subAgs.end()) && (*pos == indexSet.subIndex( element, k, codim )) );
                list.push_back( static_cast< std::size_t >( pos - subAgs.begin() ) );
              }
            }
          }
        }
      }

      // compress connectivity

      agglomerateOffset.resize( numAgglomerates );
      connectivity_.clear();
      for( std::size_t i = 0; i < numAgglomerates; ++i )
      {
        for( std::size_t dim = 0; dim < dimension; ++dim )
        {
          std::vector< std::size_t > &list = connectivity[ i ][ dim ];
          std::sort( list.begin(), list.end() );
          list.erase( std::unique( list.begin(), list.end() ), list.end() );

          offsets_[ i ][ dim ] = connectivity_.size();
          connectivity_.insert( connectivity_.end(), list.begin(), list.end() );
        }
        connectivity_[ i ][ dimension ] = connectivity_.size();
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
