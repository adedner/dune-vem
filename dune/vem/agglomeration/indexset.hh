#ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
#define DUNE_VEM_AGGLOMERATION_INDEXSET_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationIndexSet
    // ---------------------

    template< class GridPart, class Allocator = std::allocator< std::size_t > >
    class AgglomerationIndexSet
    {
      typedef AgglomerationIndexSet< GridPart > ThisType;

      struct Agglomerate;

    public:
      typedef GridPart GridPartType;

      typedef typename Allocator::rebind< std::size_t >::other AllocatorType;

      static const int dimension = GridPartType::dimension;

      AgglomerationIndexSet ( const GridPartType &gridPart, std::vector< std::size_t > agglomerateIndices, AllocatorType allocator = AllocatorType() );

    private:
      const GridPartType &gridPart_;
      AllocatorType allocator_;
      std::vector< std::size_t > agglomerateIndices_;
      std::vector< Agglomerate > agglomerates_;
    };



    // AgglomerationIndexSet::Agglomerate
    // ----------------------------------

    template< class GridPart >
    struct AgglomerationIndexSet::Agglomerate
      : private AllocatorType
    {
      Agglomerate ( AllocatorType allocator = Allocator() )
        : AllocatorType( std::move( allocator ) )
      {
        connectivity_.fill( nullptr );
      }

      template< class V >
      Agglomerate ( const std::array< V, dimension > &connectivity, AllocatorType allocator )
        : AllocatorType( std::move( allocator ) )
      {
        const std::size_t size = std::accumulate( connectivity.begin(), connectivity.end(), std::size_t( 0 ), [] ( const V &v ) { return v.size(); } );
        connectivity_[ 0 ] = allocator().allocate( size );
        for( int dim = 0; dim < dimension; ++dim )
        {
          connectivity_[ dim+1 ] = connectivity_[ dim ];
          for( std::size_t index : connectivity[ dim ] )
            allocator().construct( connectivity_[ dim+1 ]++, index );
        }
      }

      Agglomerate ( const Agglomerate & ) = delete;

      Agglomerate ( Agglomerate &&other )
        : AllocatorType( std::move( allocator() ) ), connectivity_( other.connectivity_ )
      {
        other.connectivity_.fill( nullptr );
      }

      ~Agglomerate ()
      {
        if( connectivity_[ 0 ] )
        {
          for( auto p = connectivity_[ 0 ]; p != connectivity_[ dimension ]; ++p )
            allocator().destroy( p );
          allocator().deallocate( connectivity_[ 0 ], connectivity_[ dimension ] - connectivity_[ 0 ] );
        }
      }

      Agglomerate &operator= ( const Agglomerate & ) = delete;

      Agglomerate &operator= ( Agglomerate && )
      {
        allocator() = std::move( other.allocator() );
        connectivity_ = other.connectivity_;
        other.connectivity_.fill( nullptr );
        return *this;
      }

      std::size_t subEntities ( int codim ) const
      {
        assert( (codim >= 0) && (codim <= dimension) );
        return (codim == 0 ? 1u : size( dimension - codim ));
      }

    private:
      std::size_t size ( int dim ) const { return (connectivity_[ dim+1 ] - connectivity[ dim ]); }

      AllocatorType &allocator () { return static_cast< AllocatorType & >( *this ); }

      std::array< typename AllocatorType::pointer, dimension+1 > connectivity_;
    };



    // Implementation of AgglomerationIndexSet
    // ---------------------------------------

    template< class GridPart, class Allocator >
    template< class I >
    inline AgglomerationIndexSet< GridPart, Allocator >
      ::AgglomerationIndexSet ( const GridPartType &gridPart, std::vector< I > agglomerateIndices, AllocatorType allocator )
      : gridPart_( gridPart ), allocator_( std::move( allocator ) ), agglomerateIndices_( std::move( agglomerateIndices ) )
    {
      const typename GridPart::IndexSetType &indexSet = gridPart_.indexSet();

      std::vector< std::vector< typename GridPart::IndexSetType::IndexType > subAgglomerates( GlobalGeometryTypeIndex::size( dimension-1 ) );

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

      // compute offsets

      std::vector< std::size_t > offset( GlobalGeometryTypeIndex::size( dimension ) );
      for( int dim = 0; dim < dimension; ++dim )
      {
        std::size_t size = 0;
        for( std::size_t typeIndex = GlobalGeometryTypeIndex::offset( dim ); typeIndex < GlobalGeometryTypeIndex::offset( dim+1 ); ++typeIndex )
        {
          offset[ i ] = size;
          size += subAgglomerates[ typeIndex ].size();
        }
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
                const std::size_t typeIndex = GlobalGeometryTypeIndex::index( refElement.type( k, codim ) );
                const auto &subAgs = subAgglomerates[ typeIndex ];
                const auto pos = std::lower_bound( subAgs.begin(), subAgs.end(), indexSet.subIndex( element, k, codim ) );
                assert( (pos != subAgs.end()) && (*pos == indexSet.subIndex( element, k, codim )) );
                list.push_back( offset[ typeIndex ] + static_cast< std::size_t >( pos - subAgs.begin() ) );
              }
            }
          }
        }
      }

      // compress connectivity

      agglomerates_.reserve( numAgglomerates );
      for( auto &c : connectivity )
      {
        for( int dim = 0; dim < dimension; ++dim )
        {
          std::sort( c[ dim ].begin(), c[ dim ].end() );
          c[ dim ].erase( std::unique( c[ dim ].begin(), c[ dim ].end() ), c[ dim ].end() );
        }
        agglomerate_.emplace_back( c, allocator_ );
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
