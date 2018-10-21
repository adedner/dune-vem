#ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
#define DUNE_VEM_SPACE_INTERPOLATION_HH

#include <cstddef>

#include <initializer_list>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/vem/agglomeration/shapefunctionset.hh>

namespace Dune
{

  namespace Vem
  {

    // AgglomerationVEMInterpolation
    // -----------------------------

    template< class AgglomerationIndexSet >
    class AgglomerationVEMInterpolation
    {
      typedef AgglomerationVEMInterpolation< AgglomerationIndexSet > ThisType;

    public:
      static const int dimension = AgglomerationIndexSet::dimension;

      typedef typename AgglomerationIndexSet::ElementType ElementType;

    private:
      typedef typename ElementType::Geometry::ctype ctype;

    public:
      explicit AgglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
        : indexSet_( indexSet )
      {}

      template< class LocalFunction, class LocalDofVector >
      void operator() ( const ElementType &element, const LocalFunction &localFunction, LocalDofVector &localDofVector ) const
      {
      // void operator() ( const LocalFunction &localFunction, LocalDofVector &localDofVector ) const
      // {
        // const ElementType &element = localFunction.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        for( int i = 0; i < refElement.size( dimension ); ++i )
        {
          const int k = indexSet_.localIndex( element, i, dimension );
          if( k == -1 )
            continue;

          const auto &x = refElement.position( i, dimension );

          typename LocalFunction::RangeType value;
          assert( value.dimension == 1 );
          localFunction.evaluate( x, value );
          localDofVector[ k ] = value[ 0 ];
        }
      }
      void operator() ( const ElementType &element, std::vector<bool> &mask) const
      {
        std::fill(mask.begin(),mask.end(),false);
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        for( int i = 0; i < refElement.size( dimension ); ++i )
        {
          const int k = indexSet_.localIndex( element, i, dimension );
          if ( k!=-1 ) mask[k] = true;
          // mask[k] = !( k == -1 );
        }
      }

      template< class ShapeFunctionSet, class LocalDofMatrix >
      void operator() ( const BoundingBoxShapeFunctionSet< ElementType, ShapeFunctionSet > &shapeFunctionSet, LocalDofMatrix &localDofMatrix ) const
      {
        const ElementType &element = shapeFunctionSet.entity();
        const auto &refElement = ReferenceElements< ctype, dimension >::general( element.type() );

        for( int i = 0; i < refElement.size( dimension ); ++i )
        {
          const int k = indexSet_.localIndex( element, i, dimension );
          if( k == -1 )
            continue;

          const auto &x = refElement.position( i, dimension );
          shapeFunctionSet.evaluateEach( x, [ &localDofMatrix, k ] ( std::size_t alpha, typename ShapeFunctionSet::RangeType phi ) {
              assert( phi.dimension == 1 );
              localDofMatrix[ k ][ alpha ] = phi[ 0 ];
            } );
        }
      }

      template <int polOrder>
      static std::initializer_list< std::pair< int, unsigned int > > dofsPerCodim ()
      {
        switch (polOrder)
        {
        case 1:
          return { std::make_pair( dimension, 1u ) };
        case 2:
          return { std::make_pair( dimension, 1u ),
                   std::make_pair( dimension-1, 1u ) };
        case 3:
          return { std::make_pair( dimension, 1u ),
                   std::make_pair( dimension-1, 2u ),
                   std::make_pair( dimension-2, 1u ) };
        otherwise:
          DUNE_THROW( NotImplemented, "dofsPerCodim only implemented for order 1,2, and 3" );
        }
      }

    private:
      const AgglomerationIndexSet &indexSet_;
    };



    // agglomerationVEMInterpolation
    // -----------------------------

    template< class AgglomerationIndexSet >
    inline static AgglomerationVEMInterpolation< AgglomerationIndexSet >
    agglomerationVEMInterpolation ( const AgglomerationIndexSet &indexSet ) noexcept
    {
      return AgglomerationVEMInterpolation< AgglomerationIndexSet >( indexSet );
    }

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_INTERPOLATION_HH
