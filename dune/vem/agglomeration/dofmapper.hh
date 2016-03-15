#ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
#define DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH

#include <cassert>

#include <array>
#include <type_traits>
#include <vector>

#include <dune/common/iteratorrange.hh>

#include <dune/geometry/referenceelements.hh>
#include <dune/geometry/type.hh>
#include <dune/geometry/typeindex.hh>

#include <dune/fem/gridpart/common/gridpart.hh>
#include <dune/fem/gridpart/common/indexset.hh>
#include <dune/fem/misc/functor.hh>
#include <dune/fem/space/common/dofmanager.hh>
#include <dune/fem/space/mapper/code.hh>
#include <dune/fem/space/mapper/dofmapper.hh>
#include <dune/fem/space/mapper/exceptions.hh>

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
        unsigned int numDofs = 0;
        SizeType offset, oldOffset;
      };

      enum class CodimType { Empty, FixedSize, VariableSize };

      static const int dimension = GridPart::dimension;
      typedef Dune::ReferenceElement< typename GridPart::ctype, dimension > RefElementType;
      typedef Dune::ReferenceElements< typename GridPart::ctype, dimension > RefElementsType;

      struct BuildFunctor;

      struct SubEntityFilter;

      template< class Functor >
      struct MapFunctor;

      struct SubEntityFilterFunctor;

    public:
      typedef SizeType GlobalKeyType;

      typedef GridPart GridPartType;

      typedef typename GridPartType::template Codim< 0 >::EntityType ElementType;

      template< class CodeFactory >
      AgglomerationDofMapper ( const GridPartType &gridPart, const CodeFactory &codeFactory );

      template< class Functor >
      void mapEach ( const ElementType &element, Functor f ) const
      {
        code( element )( MapFunctor< Functor >( gridPart_, subEntityInfo_, element, f ) );
      }

      void map ( const ElementType &element, std::vector< GlobalKeyType > &indices ) const
      {
        indices.resize( numDofs( element ) );
        mapEach( element, [ &indices ] ( std::size_t i, GlobalKeyType k ) {
          indices[ i ] = k;
        } );
      }

      /** \brief fills a vector of bools with true indicating that the corresponding
       *  local degree of freedom is attached to the subentity specified by the (c,i)
       *  pair.
       *  A local dof is attached to a subentity S if it is attached either to that
       *  subentity or to a subentity S'<S i.e. S' has codimension greater than c
       *  which and lies within S. For example all dofs are attached to the element itself
       *  and dofs are attached to an edge also in the case where they are attached to
       *  the vertices of that edge.
       **/
      void onSubEntity ( const ElementType &element, int i, int c, std::vector< bool > &indices ) const
      {
        indices.resize( numEntityDofs( element ) );
        code( element )( SubEntityFilterFunctor( gridPart_, subEntityInfo_, element, i, c, indices ) );
      }

      unsigned int maxNumDofs () const { return maxNumDofs_; }
      unsigned int numDofs ( const ElementType &element ) const { return code( element ).numDofs(); }

      // assignment of DoFs to entities
      template< class Entity, class Functor >
      void mapEachEntityDof ( const Entity &entity, Functor f ) const;

      template< class Entity >
      void mapEntityDofs ( const Entity &entity, std::vector< GlobalKeyType > &indices ) const
      {
        indices.resize( numEntityDofs( entity ) );
        mapEachEntityDof( entity, [ &indices ] ( std::size_t i, GlobalKeyType k ) {
          indices[ i ] = k;
        } );
      }

      template< class Entity >
      unsigned int numEntityDofs ( const Entity &entity ) const
      {
        return subEntityInfo( entity ).numDofs;
      }

      // global information

      bool contains ( int codim ) const { return ( codimType_[ codim ] != CodimType::Empty ); }

      bool fixedDataSize ( int codim ) const { return ( codimType_[ codim ] != CodimType::VariableSize ); }

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
      typedef typename GridPartType::IndexSetType IndexSetType;
      typedef std::vector< GeometryType > BlockMapType;

      const DofMapperCode &code ( const GeometryType &gt ) const { return code_[ LocalGeometryTypeIndex::index( gt ) ]; }
      const DofMapperCode &code ( const ElementType &element ) const { return code( element.type() ); }

      template< class Entity >
      const SubEntityInfo &subEntityInfo ( const Entity &entity ) const
      {
        return subEntityInfo_[ GlobalGeometryTypeIndex::index( entity.type() ) ];
      }

      const IndexSetType &indexSet () const { return gridPart_.indexSet(); }

      const GridPartType &gridPart_;
      std::vector< DofMapperCode > code_;
      unsigned int maxNumDofs_ = 0;
      SizeType size_;
      std::vector< SubEntityInfo > subEntityInfo_;
      BlockMapType blockMap_;
      stsd::array< CodimType, dimension+1 > codimType_;
    };



    // AgglomerationDofMapper::BuildFunctor
    // ------------------------------------

    template< class GridPart >
    struct AgglomerationDofMapper< GridPart >::BuildFunctor
    {
      explicit BuildFunctor ( std::vector< SubEntityInfo > &subEntityInfo )
        : subEntityInfo_( subEntityInfo )
      {}

      template< class Iterator >
      void operator() ( unsigned int gtIndex, unsigned int subEntity, Iterator it, Iterator end )
      {
        SubEntityInfo &info = subEntityInfo_[ gtIndex ];
        const unsigned int numDofs = end - it;
        if( info.numDofs == 0 )
          info.numDofs = numDofs;
        else if( info.numDofs != numDofs )
          DUNE_THROW( DofMapperError, "Inconsistent number of DoFs on subEntity (codim = " << info.codim << ")." );
      }

    private:
      std::vector< SubEntityInfo > &subEntityInfo_;
    };



    // AgglomerationDofMapper::MapFunctor
    // ----------------------------------

    // The functor maps all DoFs for a given entity. Intentially, it
    // is passed as argument to DofMapperCode::operator() which then
    // calls the apply()-method for each sub-entity with DoFs in turn.
    template< class GridPart >
    template< class Functor >
    struct AgglomerationDofMapper< GridPart >::MapFunctor
    {
      static const bool isCartesian = Dune::Capabilities::isCartesian< typename GridPart::GridType >::v;

      MapFunctor( const GridPart &gridPart, const std::vector< SubEntityInfo > &subEntityInfo,
                  const ElementType &element, Functor functor )
        : gridPart_( gridPart ),
          indexSet_( gridPart_.indexSet() ),
          subEntityInfo_( subEntityInfo ),
          element_( element ),
          functor_( functor )
      {}

      // subEntity is the sub-entity number, given codim, as returned
      // by refElem.subEntity(). The iterators iterate over all DoFs
      // attached to the given sub-entity.
      template< class Iterator >
      void operator() ( unsigned int gtIndex, unsigned int subEntity, Iterator it, Iterator end )
      {
        const int dimension = GridPart::dimension;

        const SubEntityInfo &info = subEntityInfo_[ gtIndex ];
        const SizeType subIndex = indexSet_.subIndex( element_, subEntity, info.codim );
        SizeType index = info.offset + SizeType( info.numDofs ) * subIndex;

        const unsigned int codim = info.codim;

        const unsigned int numDofs = info.numDofs;
        // for non-Cartesian grids check twist if on edges with noDofs > 1
        // this should be the case for polOrder > 2.
        // Note that in 3d this only solves the twist problem up to polOrder = 3
        if( !isCartesian && ( codim == dimension-1 ) && ( numDofs > 1 ) )
        {
          typedef typename GridPart::ctype FieldType;
          const auto &refElem = Dune::ReferenceElements< FieldType, dimension >::general( element_.type() );

          const int vx[ 2 ] = { refElem.subEntity( subEntity, codim, 0, dimension ), refElem.subEntity( subEntity, codim, 1, dimension ) };

          // flip index if face is twisted
          if( gridPart_.grid().localIdSet().subId( gridEntity( element_ ), vx[ 1 ], dimension )
              < gridPart_.grid().localIdSet().subId( gridEntity( element_ ), vx[ 0 ], dimension ) )
          {
            std::vector< unsigned int > global( numDofs );
            std::vector< unsigned int > local( numDofs );

            for( unsigned int count = 0; it != end; ++count )
            {
              global[ count ] = index++;
              local[ count ] = *( it++ );
            }

            for( unsigned int i = 0, reverse = numDofs; i < numDofs; ++i )
              functor_( local[ i ], global[ --reverse ] );

            return;
          }
        }

        while( it != end )
          functor_( *( it++ ), index++ );
      }

    private:
      const GridPart &gridPart_;
      const IndexSetType &indexSet_;
      const std::vector< SubEntityInfo > &subEntityInfo_;
      const ElementType &element_;
      Functor functor_;
    };



    // SubEntityFilter
    // ---------------

    template< class GridPart >
    struct AgglomerationDofMapper< GridPart >::SubEntityFilter
    {
      SubEntityFilter( const RefElementType &refElement, int subEntity, int codim )
        : active_( dimension+1 )
      {
        for( int c = 0; c <= dimension; ++c )
        {
          std::vector< bool > &a = active_[ c ];
          a.resize( refElement.size( c ), false );
          if( c < codim )
            continue;

          if( c == codim )
          {
            a[ subEntity ] = true;
            ++size_;
            continue;
          }

          for( int i = 0; i < refElement.size( subEntity, codim, c ); ++i )
          {
            a[ refElement.subEntity( subEntity, codim, i, c ) ] = true;
            ++size_;
          }
        }
      }

      bool operator() ( int i, int c ) const { return active_[ c ][ i ]; }

    private:
      std::vector< std::vector< bool > > active_;
      int size_ = 0;
    };



    // SubEntityFilterFunctor
    // ----------------------

    template< class GridPart >
    struct AgglomerationDofMapper< GridPart >::SubEntityFilterFunctor
    {
      static const bool isCartesian = Dune::Capabilities::isCartesian< typename GridPart::GridType >::v;

      SubEntityFilterFunctor( const GridPart &gridPart, const std::vector< SubEntityInfo > &subEntityInfo,
                              const ElementType &element, int i, int c, std::vector< bool > &vec )
        : gridPart_( gridPart ),
          subEntityInfo_( subEntityInfo ),
          element_( element ),
          vec_( vec ),
          filter_( RefElementsType::general( element.type() ), i, c )
      {}

      template< class Iterator >
      void operator() ( unsigned int gtIndex, unsigned int subEntity, Iterator it, Iterator end )
      {
        const SubEntityInfo &info = subEntityInfo_[ gtIndex ];
        bool active = filter_( subEntity, info.codim );
        while( it != end )
          vec_[ *( it++ ) ] = active;
      }

    private:
      const GridPart &gridPart_;
      const std::vector< SubEntityInfo > &subEntityInfo_;
      const ElementType &element_;
      std::vector< bool > &vec_;
      SubEntityFilter filter_;
    };



    // Implementation of AgglomerationDofMapper
    // ----------------------------------------

    template< class GridPart >
    const int AgglomerationDofMapper< GridPart >::dimension;

    template< class GridPart >
    template< class CodeFactory >
    inline AgglomerationDofMapper< GridPart >::AgglomerationDofMapper ( const GridPartType &gridPart, const CodeFactory &codeFactory )
      : gridPart_( gridPart ),
      code_( LocalGeometryTypeIndex::size( dimension ) ),
      subEntityInfo_( GlobalGeometryTypeIndex::size( dimension ) )
    {
      std::vector< GeometryType > gt( GlobalGeometryTypeIndex::size( dimension ) );

      IteratorRange< typename RefElementsType::Iterator > refElements( RefElementsType::begin(), RefElementsType::end() );
      for( const auto &refElement : refElements )
      {
        for( int codim = 0; codim <= dimension; ++codim )
          for( int i = 0; i < refElement.size( codim ); ++i )
          {
            const unsigned int gtIdx = GlobalGeometryTypeIndex::index( refElement.type( i, codim ) );
            gt[ gtIdx ] = refElement.type( i, codim );
            subEntityInfo_[ gtIdx ].codim = codim;
          }

        DofMapperCode &code = code_[ LocalGeometryTypeIndex::index( refElement.type() ) ];
        code = codeFactory( refElement );
        maxNumDofs_ = std::max( code.numDofs(), maxNumDofs_ );
        code( BuildFunctor( subEntityInfo_ ) );
      }

      std::fill( codimType_.begin(), codimType_.end(), CodimType::Empty );
      unsigned int codimDofs[ dimension+1 ];
      for( unsigned int i = 0; i < subEntityInfo_.size(); ++i )
      {
        const SubEntityInfo &info = subEntityInfo_[ i ];
        if( info.numDofs == 0 )
          continue;

        if( codimType_[ info.codim ] == CodimType::Empty )
          codimType_[ info.codim ] = CodimType::FixedSize;
        else if( codimDofs[ info.codim ] != info.numDofs )
          codimType_[ info.codim ] = CodimType::VariableSize;

        codimDofs[ info.codim ] = info.numDofs;
        blockMap_.push_back( gt[ i ] );
      }

      update();
    }


    template< class GridPart >
    template< class Entity, class Functor >
    inline void AgglomerationDofMapper< GridPart >::mapEachEntityDof ( const Entity &entity, Functor f ) const
    {
      const SubEntityInfo &info = subEntityInfo( entity );
      const unsigned int numDofs = info.numDofs;
      SizeType index = info.offset + numDofs * SizeType( indexSet().index( entity ) );
      for( unsigned int i = 0; i < info.numDofs; ++i )
        f( i, index++ );
    }


    template< class GridPart >
    inline void AgglomerationDofMapper< GridPart >::update ()
    {
      size_ = 0;
      for( typename BlockMapType::const_iterator it = blockMap_.begin(); it != blockMap_.end(); ++it )
      {
        SubEntityInfo &info = subEntityInfo_[ GlobalGeometryTypeIndex::index( *it ) ];
        info.oldOffset = info.offset;
        info.offset = size_;
        size_ += SizeType( info.numDofs ) * SizeType( indexSet().size( *it ) );
      }
    }

  } // namespace Fem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_DOFMAPPER_HH
