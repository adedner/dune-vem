#ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH

#include <cassert>

#include <array>
#include <type_traits>
#include <vector>

#include <dune/geometry/type.hh>

#include <dune/vem/agglomeration/indexset.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationDofMapper
    // ----------------------

    template< class GridPart >
    class AgglomerationDofMapper
    {
      typedef AgglomerationDofMapper< GridPart > This;

    public:
      typedef std::size_t SizeType;

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
      AgglomerationDofMapper ( const AgglomerationIndexSet< GridPart > &agIndexSet, Iterator begin, Iterator end );

      template< class Functor >
      void mapEach ( const ElementType &element, Functor f ) const;

      void map ( const ElementType &element, std::vector< GlobalKeyType > &indices ) const
      {
        indices.resize( numDofs( element ) );
        mapEach( element, [ &indices ] ( std::size_t i, GlobalKeyType k ) { indices[ i ] = k; } );
      }

      void onSubEntity ( const ElementType &element, int i, int c, std::vector< bool > &indices ) const
      {
        indices.resize( numDofs( element ) );
        DUNE_THROW( NotImplemented, "AgglomerateDofMapepr::onSubEntity not implemented, yet" );
      }

      unsigned int maxNumDofs () const { return maxNumDofs_; }

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
        return subEntityInfo( entity ).numDofs;
      }

      // global information

      bool contains ( int codim ) const
      {
        const auto isCodim = [] ( const SubEntityInfo &info ) { return (info.codim == codim); };
        return (std::find( subEntityInfo_.begin(), subEntityInfo_.end(), isCodim ) != subEntityInfo_.end());
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

    protected:
      const AgglomerationIndexSet< GridPart > &agIndexSet_;
      unsigned int maxNumDofs_ = 0;
      SizeType size_;
      std::vector< SubEntityInfo > subEntityInfo_;
    };



    // Implementation of AgglomerationDofMapper
    // ----------------------------------------

    template< class GridPart >
    const int AgglomerationDofMapper< GridPart >::dimension;


    template< class GridPart >
    template< class Iterator >
    inline AgglomerationDofMapper< GridPart >::AgglomerationDofMapper ( const AgglomerationIndexSet< GridPart > &agIndexSet, Iterator begin, Iterator end )
      : agIndexSet_( agIndexSet ),
        subEntityInfo_( end - begin )
    {
      std::transform( begin, end, subEntityInfo_.begin(), [] ( std::pair< int, unsigned int > codimDofs, SubEntityInfo &info ) {
          info.codim = codimDofs.first;
          info.numDofs = codimDofs.second;
        } );

      update();
    }


    template< class GridPart >
    inline unsigned int AgglomerationDofMapper< GridPart >::numDofs ( const ElementType &element ) const
    {
      unsigned int numDofs = 0;
      for( const subEntityInfo &info : subEntityInfo_ )
        numDofs += info.numDofs * agIndexSet_.subAgglomerates( element, info.codim );
      return numDofs;
    }


    template< class GridPart >
    template< class Functor >
    inline void AgglomerationDofMapper< GridPart >::mapEach ( const ElementType &element, Functor f ) const
    {
      unsigned int local = 0;
      for( const subEntityInfo &info : subEntityInfo_ )
      {
        const std::size_t numSubAgglomerates = agIndexSet_.subAgglomerates( element, info.codim );
        for( std::size_t subAgglomerate = 0; subAgglomerate < numSubAgglomerates; ++subAgglomerate )
        {
          const SizeType subIndex = agIndexSet_.subIndex( element, subAgglomerate, codim );
          SizeType index = info.offset + SizeType( info.numDofs ) * subIndex;

          const SizeType end = index + info.numDofs;
          while( index < end )
            functor( local++, index++ );
        }
      }
    }


    template< class GridPart >
    template< class Entity, class Functor >
    inline void AgglomerationDofMapper< GridPart >::mapEachEntityDof ( const Entity &entity, Functor f ) const
    {
      DUNE_THROW( NotImplemented, "mapEachEntityDof not implemented, yet" );
    }


    template< class GridPart >
    inline void AgglomerationDofMapper< GridPart >::update ()
    {
      size_ = 0;
      maxNumDofs_ = 0;
      for( const subEntityInfo &info : subEntityInfo_ )
      {
        info.offset = size_;
        size_ += SizeType( info.numDofs ) * SizeType( agIndexSet_.size( info.codim ) );
        maxNumDofs_ += SizeType( info.numDofs ) * SizeType( agIndexSet_.maxSubAgglomerates( info.codim ) );
      }
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
