#ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH

#include <cassert>

#include <array>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include <dune/geometry/type.hh>

#include <dune/fem/gridpart/common/gridpart.hh>

#include <dune/vem/agglomeration/indexset.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationDofMapper
    // ----------------------

    template< class GridPart, class IndexSet=AgglomerationIndexSet< GridPart > >
    class AgglomerationDofMapper
    {
      typedef AgglomerationDofMapper< GridPart > This;

    public:
      typedef std::size_t SizeType;

      typedef Agglomeration< GridPart > AgglomerationType;
      typedef IndexSet IndexSetType;

    protected:
      struct SubEntityInfo
      {
        unsigned int codim;
        unsigned int numDofs;
        SizeType offset;
      };

      static const int dimension = GridPart::dimension;

    public:
      typedef SizeType GlobalKeyType;

      typedef GridPart GridPartType;

      typedef typename GridPartType::template Codim< 0 >::EntityType ElementType;

      template< class Iterator >
      AgglomerationDofMapper ( const IndexSetType &indexSet, Iterator begin, Iterator end );

      AgglomerationDofMapper ( const IndexSetType &indexSet, const std::array<std::pair<int,unsigned int>,dimension+1> &dofsPerCodim )
        : AgglomerationDofMapper( indexSet, dofsPerCodim.begin(), dofsPerCodim.end() )
      {}

      template< class Functor >
      void mapEach ( const ElementType &element, Functor f ) const;

      void map ( const ElementType &element, std::vector< GlobalKeyType > &indices ) const
      {
        indices.resize( numDofs( element ) );
        mapEach( element, [ &indices ] ( std::size_t i, GlobalKeyType k ) { indices[ i ] = k; } );
      }

      void onSubEntity ( const ElementType &element, int i, int c, std::vector< bool > &filter ) const;

      unsigned int maxNumDofs () const { return maxNumDofs_; }

      unsigned int numDofs ( std::size_t agglomerate ) const;
      unsigned int numDofs ( const ElementType &element ) const;

      // assignment of DoFs to entities
      template< class Entity, class Functor >
      void mapEachEntityDof ( const Entity &entity, Functor f ) const;

      template< class Entity >
      void mapEntityDofs ( const Entity &entity, std::vector< GlobalKeyType > &indices ) const
      {
        indices.resize( numEntityDofs( entity ) );
        mapEachEntityDof( entity, [ &indices ] ( std::size_t i, GlobalKeyType k ) { indices[ i ] = k; } );
      }

      template< class Entity >
      unsigned int numEntityDofs ( const Entity &entity ) const
      {
        const int codimIndex = codimIndex_[ Entity::codimension ];
        return ((codimIndex >= 0) && indexSet().globalIndex( entity ).second ? subEntityInfo_[ codimIndex ].numDofs : 0u);
      }

      // global information

      bool contains ( unsigned int codim ) const
      {
        assert( codim <= static_cast< unsigned int >( dimension ) );
        return (codimIndex_[ codim ] >= 0);
      }

      bool fixedDataSize ( int codim ) const { return false; }

      SizeType size () const { return size_; }

      void update ();

      /* Compatibility methods; users expect an AdaptiveDiscreteFunction to
       * compile over spaces built on top of a LeafGridPart or LevelGridPart.
       *
       * The AdaptiveDiscreteFunction requires the block mapper (i.e. this
       * type) to be adaptive. The CodimensionMapper however is truly
       * adaptive if and only if the underlying index set is adaptive. We
       * don't want to wrap the index set as 1) it hides the actual problem
       * (don't use the AdaptiveDiscreteFunction with non-adaptive index
       * sets), and 2) other dune-fem classes may make correct use of the
       * index set's capabilities.
       */

      static constexpr bool consecutive () noexcept { return false; }

      SizeType numBlocks () const { DUNE_THROW( NotImplemented, "Method numBlocks() called on non-adaptive block mapper" ); }
      SizeType numberOfHoles ( int ) const { DUNE_THROW( NotImplemented, "Method numberOfHoles() called on non-adaptive block mapper" ); }
      GlobalKeyType oldIndex ( int hole, int ) const { DUNE_THROW( NotImplemented, "Method oldIndex() called on non-adaptive block mapper" ); }
      GlobalKeyType newIndex ( int hole, int ) const { DUNE_THROW( NotImplemented, "Method newIndex() called on non-adaptive block mapper" ); }
      SizeType oldOffSet ( int ) const { DUNE_THROW( NotImplemented, "Method oldOffSet() called on non-adaptive block mapper" ); }
      SizeType offSet ( int ) const { DUNE_THROW( NotImplemented, "Method offSet() called on non-adaptive block mapper" ); }

      const IndexSetType &indexSet () const { return indexSet_; }
      const AgglomerationType &agglomeration () const { return indexSet().agglomeration(); }

    protected:
      const IndexSetType &indexSet_;
      unsigned int maxNumDofs_ = 0;
      SizeType size_;
      std::vector< SubEntityInfo > subEntityInfo_;
      std::array< int, dimension+1 > codimIndex_;
      std::vector< bool > edgeTwist_;
    };



    // Implementation of AgglomerationDofMapper
    // ----------------------------------------

    template< class GridPart, class IndexSet >
    const int AgglomerationDofMapper< GridPart, IndexSet >::dimension;


    template< class GridPart, class IndexSet >
    template< class Iterator >
    inline AgglomerationDofMapper< GridPart, IndexSet >::AgglomerationDofMapper ( const IndexSetType &indexSet, Iterator begin, Iterator end )
      : indexSet_( indexSet ), subEntityInfo_( std::distance( begin, end ) )
    {
      std::transform( begin, end, subEntityInfo_.begin(), [] ( std::pair< int, unsigned int > codimDofs ) {
          SubEntityInfo info;
          info.codim = codimDofs.first;
          info.numDofs = codimDofs.second;
          return info;
        } );

      std::fill( codimIndex_.begin(),codimIndex_.end(), -1 );
      const int numCodims = subEntityInfo_.size();
      for( int i = 0; i < numCodims; ++i )
      {
        const int codim = subEntityInfo_[ i ].codim;
        if( codimIndex_[ codim ] >= 0 )
          DUNE_THROW( InvalidStateException, "Codimension inserted twice into AgglomerationDofMapper" );
        codimIndex_[ codim ] = i;
      }

      update();
    }


    template< class GridPart, class IndexSet >
    inline unsigned int AgglomerationDofMapper< GridPart, IndexSet >::numDofs ( std::size_t agglomerate ) const
    {
      unsigned int numDofs = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
        numDofs += info.numDofs * indexSet().subAgglomerates( agglomerate, info.codim );
      return numDofs;
    }


    template< class GridPart, class IndexSet >
    inline unsigned int AgglomerationDofMapper< GridPart, IndexSet >::numDofs ( const ElementType &element ) const
    {
      unsigned int numDofs = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
        numDofs += info.numDofs * indexSet().subAgglomerates( element, info.codim );
      return numDofs;
    }


    template< class GridPart, class IndexSet >
    inline void AgglomerationDofMapper< GridPart, IndexSet >::onSubEntity ( const ElementType &element, int i, int c, std::vector< bool > &filter ) const
    {
      filter.resize( numDofs( element ) );
      std::fill( filter.begin(), filter.end(), false );
      const auto &refElement = Dune::ReferenceElements< typename GridPart::ctype, dimension >::general( element.type() );
      unsigned int localOfs = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
      {
        const int size = refElement.size( i, c, info.codim );
        for( int k = 0; k < size; ++k )
        {
          int idx = indexSet().localIndex( element, refElement.subEntity( i, c, k, info.codim ), info.codim );
          if( idx >= 0 )
            for (int l=0;l<info.numDofs;++l)
              filter[ localOfs + idx*info.numDofs + l ] = true;
        }
        localOfs += info.numDofs * indexSet().subAgglomerates( element, info.codim );
      }
    }


    template< class GridPart, class IndexSet >
    template< class Functor >
    inline void AgglomerationDofMapper< GridPart, IndexSet >::mapEach ( const ElementType &element, Functor f ) const
    {
      unsigned int local = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
      {
        const std::size_t numSubAgglomerates = indexSet().subAgglomerates( element, info.codim );
        for( std::size_t subAgglomerate = 0; subAgglomerate < numSubAgglomerates; ++subAgglomerate )
        {
          const SizeType subIndex = indexSet().subIndex( element, subAgglomerate, info.codim );
          SizeType index = info.offset + SizeType( info.numDofs ) * subIndex;

          if( (info.codim == 1) && (edgeTwist_[ subIndex ] == 1) )
          {
            const SizeType begin = index;
            for( index += info.numDofs; index > begin; )
              f( local++, --index );
          }
          else
          {
            const SizeType end = index + info.numDofs;
            for( ; index < end; )
              f( local++, index++ );
          }
        }
      }
    }


    template< class GridPart, class IndexSet >
    template< class Entity, class Functor >
    inline void AgglomerationDofMapper< GridPart, IndexSet >::mapEachEntityDof ( const Entity &entity, Functor f ) const
    {
      const SubEntityInfo &info = subEntityInfo_[ codimIndex_[ Entity::codimension ] ];
      if( info.numDofs == 0u )
        return;

      const auto result = indexSet().globalIndex( entity );
      if( !result.second )
        return;

      SizeType index = info.offset + SizeType( info.numDofs ) * result.first;
      if( (Entity::codimension == 1) && (edgeTwist_[ result.first ]) )
      {
        for( unsigned int i = info.numDofs; i > 0; )
          f( --i, index++ );
      }
      else
      {
        for( unsigned int i = 0; i < info.numDofs; )
          f( i++, index++ );
      }
    }


    template< class GridPart, class IndexSet >
    inline void AgglomerationDofMapper< GridPart, IndexSet >::update ()
    {
      size_ = 0;
      maxNumDofs_ = 0;
      for( SubEntityInfo &info : subEntityInfo_ )
      {
        info.offset = size_;
        size_ += SizeType( info.numDofs ) * SizeType( indexSet().size( info.codim ) );
        maxNumDofs_ += SizeType( info.numDofs ) * SizeType( indexSet().maxSubAgglomerates( info.codim ) );
      }

      if( dimension > 1 )
      {
        const auto &idSet = agglomeration().gridPart().grid().globalIdSet();

        edgeTwist_.resize( indexSet().size( dimension-1 ) );
        for( const auto element : elements( agglomeration().gridPart(), Partitions::interiorBorder ) )
        {
          const auto &refElement = ReferenceElements< typename GridPart::ctype, dimension >::general( element.type() );

          const int numEdges = refElement.size( dimension-1 );
          for( int i = 0; i < numEdges; ++i )
          {
            const auto left = idSet.subId( Dune::Fem::gridEntity(element), refElement.subEntity( i, dimension-1, 0, dimension ), dimension );
            const auto right = idSet.subId( Dune::Fem::gridEntity(element), refElement.subEntity( i, dimension-1, 1, dimension ), dimension );
            edgeTwist_[ indexSet().subIndex( element, i, dimension-1 ) ] = (right < left);
          }
        }
      }
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
