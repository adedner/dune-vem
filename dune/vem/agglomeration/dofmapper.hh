#ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH

#include <cassert>

#include <array>
#include <initializer_list>
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

      typedef Agglomeration< GridPart > AgglomerationType;
      typedef AgglomerationIndexSet< GridPart > IndexSetType;

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

      AgglomerationDofMapper ( const IndexSetType &indexSet, std::initializer_list< std::pair< int, unsigned int > > dofsPerCodim )
        : AgglomerationDofMapper( indexSet, dofsPerCodim.begin(), dofsPerCodim.end() )
      {}

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
        DUNE_THROW( NotImplemented, "numEntityDofs not implemented, yet" );
      }

      // global information

      bool contains ( unsigned int codim ) const
      {
        const auto isCodim = [ codim ] ( const SubEntityInfo &info ) { return (info.codim == codim); };
        return (std::find_if( subEntityInfo_.begin(), subEntityInfo_.end(), isCodim ) != subEntityInfo_.end());
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
    };



    // Implementation of AgglomerationDofMapper
    // ----------------------------------------

    template< class GridPart >
    const int AgglomerationDofMapper< GridPart >::dimension;


    template< class GridPart >
    template< class Iterator >
    inline AgglomerationDofMapper< GridPart >::AgglomerationDofMapper ( const IndexSetType &indexSet, Iterator begin, Iterator end )
      : indexSet_( indexSet ), subEntityInfo_( std::distance( begin, end ) )
    {
      std::transform( begin, end, subEntityInfo_.begin(), [] ( std::pair< int, unsigned int > codimDofs ) {
          SubEntityInfo info;
          info.codim = codimDofs.first;
          info.numDofs = codimDofs.second;
          return info;
        } );

      update();
    }


    template< class GridPart >
    inline unsigned int AgglomerationDofMapper< GridPart >::numDofs ( const ElementType &element ) const
    {
      unsigned int numDofs = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
        numDofs += info.numDofs * indexSet().subAgglomerates( element, info.codim );
      return numDofs;
    }


    template< class GridPart >
    template< class Functor >
    inline void AgglomerationDofMapper< GridPart >::mapEach ( const ElementType &element, Functor f ) const
    {
      unsigned int local = 0;
      for( const SubEntityInfo &info : subEntityInfo_ )
      {
        const std::size_t numSubAgglomerates = indexSet().subAgglomerates( element, info.codim );
        for( std::size_t subAgglomerate = 0; subAgglomerate < numSubAgglomerates; ++subAgglomerate )
        {
          const SizeType subIndex = indexSet().subIndex( element, subAgglomerate, info.codim );
          SizeType index = info.offset + SizeType( info.numDofs ) * subIndex;

          const SizeType end = index + info.numDofs;
          while( index < end )
            f( local++, index++ );
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
      for( SubEntityInfo &info : subEntityInfo_ )
      {
        info.offset = size_;
        size_ += SizeType( info.numDofs ) * SizeType( indexSet().size( info.codim ) );
        maxNumDofs_ += SizeType( info.numDofs ) * SizeType( indexSet().maxSubAgglomerates( info.codim ) );
      }
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH