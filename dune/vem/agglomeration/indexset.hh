#ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
#define DUNE_VEM_AGGLOMERATION_INDEXSET_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include <dune/grid/common/rangegenerators.hh>

#include <dune/vem/agglomeration/agglomeration.hh>

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

    public:
      typedef GridPart GridPartType;

      typedef Agglomeration< GridPartType > AgglomerationType;

      typedef typename Allocator::template rebind< std::size_t >::other AllocatorType;

      static const int dimension = GridPartType::dimension;

      typedef FieldVector< typename GridPartType::ctype, GridPartType::dimensionworld > GlobalCoordinate;

      typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;

    private:
      struct Agglomerate;

    public:
      AgglomerationIndexSet ( const AgglomerationType &agglomeration, AllocatorType allocator = AllocatorType() );

      std::size_t index ( const EntityType &entity ) const { return agglomeration_.index( entity ); }

      std::size_t subIndex ( const EntityType &entity, int i, int codim ) const
      {
        assert( ( codim >= 0 ) && ( codim <= dimension ) );
        return ( codim == 0 ? index( entity ) : agglomerate( entity ).index( i, dimension - codim ));
      }

      int localIndex ( const EntityType &entity, int i, int codim ) const
      {
        assert( ( codim >= 0 ) && ( codim <= dimension ) );
        if( codim == 0 )
          return 0;
        assert( codim == dimension );
        const typename GridPartType::IndexSetType indexSet = agglomeration_.gridPart().indexSet();
        const int globalIndexInPolygonalGrid = corners_[ indexSet.subIndex( entity, i, codim ) ];
        if( globalIndexInPolygonalGrid == -1 )
          return -1;
        auto &localAgg = agglomerate( entity );
        for( std::size_t k = 0; k < localAgg.size( dimension-codim ); ++k )
          if( localAgg.index( k, dimension-codim ) == static_cast< std::size_t >( globalIndexInPolygonalGrid ) )
            return k;
        assert( false );
        return -1;
      }

      int localEdgeIndex ( const EntityType &entity, int i, int codim ) const
      {
        assert( ( codim >= 0 ) && ( codim <= dimension ) );
        if( codim == 0 )
          return 0;
        assert( codim == dimension-1 );     // for edges
        const typename GridPartType::IndexSetType indexSet = agglomeration_.gridPart().indexSet();
        int globalEdgeIndexInPolygonalGrid = edges_[ indexSet.subIndex( entity, i, codim ) ];
        if( globalEdgeIndexInPolygonalGrid == -1 )
          return -1;
        auto &localAgg = agglomerate( entity );
        for( std::size_t k = 0; k < localAgg.size( dimension-codim ); ++k )
          if( localAgg.index( k, dimension-codim ) == static_cast< std::size_t >( globalEdgeIndexInPolygonalGrid ) )
            return k;
        assert( false );
        return -1;
      }


      int numPolyVertices ( const EntityType &entity, int codim ) const
      {
        assert(( codim > 0 ) && ( codim == dimension ));
        return agglomerate( entity ).size( dimension-codim );
      }

      /**
       * obtain number of subagglomerates
       *
       * \param[in]  index  index of the agglomerate
       * \param[in]  codim  codimension of the subagglomerates
       */
      std::size_t subAgglomerates ( std::size_t index, int codim ) const
      {
        assert( ( codim >= 0 ) && ( codim <= dimension ) );
        return ( codim == 0 ? 1u : agglomerate( index ).size( dimension - codim ));
      }

      /**
       * obtain number of subagglomerates
       *
       * \param[in]  entity  any entity belonging to the agglomerate
       * \param[in]  codim   codimension of the subagglomerates
       */
      std::size_t subAgglomerates ( const EntityType &entity, int codim ) const
      {
        return subAgglomerates( index( entity ), codim );
      }

      std::size_t maxSubAgglomerates ( int codim ) const
      {
        assert( (codim >= 0) && (codim <= dimension) );
        return maxSubAgglomerates_[ dimension-codim ];
      }

      std::size_t size ( int codim ) const
      {
        assert( (codim >= 0) && (codim <= dimension) );
        return size_[ dimension-codim ];
      }

    private:
      const Agglomerate &agglomerate ( std::size_t agglomerateIndex ) const { return agglomerates_[ agglomerateIndex ]; }
      const Agglomerate &agglomerate ( const EntityType &entity ) const { return agglomerate( index( entity ) ); }

      const AgglomerationType &agglomeration_;
      AllocatorType allocator_;
      std::vector< Agglomerate > agglomerates_;
      std::array< std::size_t, dimension+1 > size_;
      std::array< std::size_t, dimension+1 > maxSubAgglomerates_;
      std::vector< int > corners_;
      std::vector< int > edges_;
    };



    // AgglomerationIndexSet::Agglomerate
    // ----------------------------------

    template< class GridPart, class Allocator >
    struct AgglomerationIndexSet< GridPart, Allocator >::Agglomerate
      : private AllocatorType
    {
      Agglomerate( AllocatorType allocator = Allocator() )
        : AllocatorType( std::move( allocator ) )
      {
        connectivity_.fill( nullptr );
      }

      template< class V >
      Agglomerate( const std::array< V, dimension > &connectivity, AllocatorType allocator )
        : AllocatorType( std::move( allocator ) )
      {
        build( connectivity );
      }

      Agglomerate( const Agglomerate & ) = delete;

      Agglomerate( Agglomerate &&other )
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

      Agglomerate &operator= ( Agglomerate &&other )
      {
        allocator() = std::move( other.allocator() );
        connectivity_ = other.connectivity_;
        other.connectivity_.fill( nullptr );
        return *this;
      }

      std::size_t index ( std::size_t i, int dim ) const
      {
        assert( i < size( dim ) );
        return connectivity_[ dim ][ i ];
      }

      std::size_t size ( int dim ) const { return ( connectivity_[ dim+1 ] - connectivity_[ dim ] ); }

    private:
      AllocatorType &allocator () { return static_cast< AllocatorType & >( *this ); }

      template< class V >
      void build ( const std::array< V, dimension > &connectivity )
      {
        const std::size_t size = std::accumulate( connectivity.begin(), connectivity.end(), std::size_t( 0 ), [] ( std::size_t i, const V &v ) {
            return i + v.size();
          } );
        connectivity_[ 0 ] = allocator().allocate( size );
        for( int dim = 0; dim < dimension; ++dim )
        {
          connectivity_[ dim+1 ] = connectivity_[ dim ];
          for( std::size_t index : connectivity[ dim ] )
            allocator().construct( connectivity_[ dim+1 ]++, index );
        }
      }

      std::array< typename AllocatorType::pointer, dimension+1 > connectivity_;
    };



    // Implementation of AgglomerationIndexSet
    // ---------------------------------------

    template< class GridPart, class Allocator >
    inline AgglomerationIndexSet< GridPart, Allocator >::AgglomerationIndexSet ( const AgglomerationType &agglomeration, AllocatorType allocator )
      : agglomeration_( agglomeration ),
      allocator_( std::move( allocator ) )
    {
      const typename GridPartType::IndexSetType indexSet = agglomeration_.gridPart().indexSet();
      std::vector< std::vector< typename GridPartType::IndexSetType::IndexType > > subAgglomerates( GlobalGeometryTypeIndex::size( dimension-1 ) );

      // find subagglomerates
      //
      // insert the index of each subentity belonging to more than one agglomerate into the
      // subAgglomerates vector for the corresponding geometry type

      for( const auto element : elements( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), Partitions::interiorBorder ) )
      {
        const std::size_t agIndex = index( element );
        const auto &refElement = ReferenceElements< typename GridPart::ctype, dimension >::general( element.type() );
        for( const auto intersection : intersections( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), element ) )
        {
          assert( intersection.conforming() );
          if( !intersection.neighbor() || ( index( intersection.outside() ) != agIndex ) )
          {
            const int face = intersection.indexInInside();
            for( int codim = 1; codim <= dimension; ++codim )
            {
              const int numSubEntities = refElement.size( face, 1, codim );
              for( int i = 0; i < numSubEntities; ++i )
              {
                const int k = refElement.subEntity( face, 1, i, codim );
                const std::size_t typeIndex = GlobalGeometryTypeIndex::index( refElement.type( k, codim ) );
                subAgglomerates[ typeIndex ].push_back( indexSet.subIndex( element, k, codim ) );
              }
            }
          }
        }
      }

      // make subagglomerates unique
      //
      // The position of a subentity's index in the subAgglometates vector will then give its subagglomerate index.

      for( auto &list : subAgglomerates )
      {
        std::sort( list.begin(), list.end() );
        list.erase( std::unique( list.begin(), list.end() ), list.end() );
      }

      // compute offsets

      std::vector< std::size_t > offset( GlobalGeometryTypeIndex::size( dimension ) );
      for( int dim = 0; dim < dimension; ++dim )
      {
        maxSubAgglomerates_[ dim ] = 0;
        size_[ dim ] = 0;
        for( std::size_t typeIndex = GlobalGeometryTypeIndex::offset( dim ); typeIndex < GlobalGeometryTypeIndex::offset( dim+1 ); ++typeIndex )
        {
          offset[ typeIndex ] = size_[ dim ];
          const std::size_t numSubAgglomerages = subAgglomerates[ typeIndex ].size();
          size_[ dim ] += numSubAgglomerates;
          maxSubAgglomerates_[ dim ] = std::max( maxSubAgglomerates_[ dim ], numSubAgglomerates );
        }
      }

      // build connectivity

      maxSubAgglomerates_[ dimension ] = 1;
      size_[ dimension ] = agglomeration.size();
      std::vector< std::array< std::vector< std::size_t >, dimension > > connectivity( size_[ dimension ] );

      for( const auto element : elements( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), Partitions::interiorBorder ) )
      {
        const std::size_t agIndex = agglomeration.index( element );
        const auto &refElement = ReferenceElements< typename GridPart::ctype, dimension >::general( element.type() );
        //for( const auto intersection : intersections( agglomeration_.gridPart(), element ) )
        for( const auto intersection : intersections( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), element ))
        {
          assert( intersection.conforming() );
          if( !intersection.neighbor() || ( agglomeration.index( intersection.outside() ) != agIndex ) )
          {
            const int face = intersection.indexInInside();
            for( int codim = 1; codim <= dimension; ++codim )
            {
              std::vector< std::size_t > &list = connectivity[ agIndex ][ dimension - codim ];
              const int numSubEntities = refElement.size( face, 1, codim );
              for( int i = 0; i < numSubEntities; ++i )
              {
                const int k = refElement.subEntity( face, 1, i, codim );
                const std::size_t typeIndex = GlobalGeometryTypeIndex::index( refElement.type( k, codim ) );
                const auto &subAgs = subAgglomerates[ typeIndex ];
                const auto pos = std::lower_bound( subAgs.begin(), subAgs.end(), indexSet.subIndex( element, k, codim ) );
                assert( ( pos != subAgs.end()) && ( *pos == indexSet.subIndex( element, k, codim )) );
                list.push_back( offset[ typeIndex ] + static_cast< std::size_t >( pos - subAgs.begin() ) );
              }
            }
          }
        }
      }

      // compress connectivity

      agglomerates_.reserve( size_[ dimension ] );
      for( auto &c : connectivity )
      {
        for( int dim = 0; dim < dimension; ++dim )
        {
          std::sort( c[ dim ].begin(), c[ dim ].end() );
          c[ dim ].erase( std::unique( c[ dim ].begin(), c[ dim ].end() ), c[ dim ].end() );
        }
        agglomerates_.emplace_back( c, allocator_ );
      }

      // copy corners

      corners_.resize( indexSet.size( dimension ), -1 );
      for( const auto vertex : vertices( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), Partitions::interiorBorder ) )
      {
        const std::size_t typeIndex = GlobalGeometryTypeIndex::index( vertex.type() );
        const auto &subAgs = subAgglomerates[ typeIndex ];
        const auto vertexIndex = indexSet.index( vertex );
        const auto pos = std::lower_bound( subAgs.begin(), subAgs.end(), vertexIndex );
        if( ( pos != subAgs.end()) && ( *pos == vertexIndex ) )
          corners_[ vertexIndex ]
            = offset[ typeIndex ] + static_cast< std::size_t >( pos - subAgs.begin() );
      }

      // copy edges
      edges_.resize( indexSet.size( dimension-1 ), -1 );
      for( const auto edge : edges( static_cast< typename GridPart::GridViewType >( agglomeration_.gridPart() ), Partitions::interiorBorder ) )
      {
        const std::size_t typeIndex = GlobalGeometryTypeIndex::index( edge.type() );
        const auto &subAgs = subAgglomerates[ typeIndex ];
        const auto edgeIndex = indexSet.index( edge );
        const auto pos = std::lower_bound( subAgs.begin(), subAgs.end(), edgeIndex );
        if( ( pos != subAgs.end()) && ( *pos == edgeIndex ) )
          edges_[ edgeIndex ]
            = offset[ typeIndex ] + static_cast< std::size_t >( pos - subAgs.begin() );
      }
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_INDEXSET_HH
