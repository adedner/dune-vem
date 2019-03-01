#ifndef DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH
#define DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <utility>

#include <dune/fem/quadrature/quadrature.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#error SHOULD NOT BE USED
#if 0
namespace Dune
{

  namespace Vem
  {

    // BoundingBoxShapeFunctionSet
    // ---------------------------

    template< class Entity, class ShapeFunctionSet >
    class BoundingBoxShapeFunctionSet
    {
      typedef BoundingBoxShapeFunctionSet< Entity, ShapeFunctionSet > ThisType;

    public:
      typedef typename ShapeFunctionSet::FunctionSpaceType FunctionSpaceType;

      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      static constexpr int dimDomain = DomainType::dimension;

      BoundingBoxShapeFunctionSet () = default;

      BoundingBoxShapeFunctionSet ( const Entity &entity, std::pair< DomainType, DomainType > bbox,
                                    ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ), shapeFunctionSet_( std::move( shapeFunctionSet ) ), bbox_( std::move( bbox ) )
      {}

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return shapeFunctionSet_.size(); }

      template< class Point, class Functor >
      void evaluateEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), functor );
      }

      template< class Point, class Functor >
      void jacobianEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.jacobianEach( position( x ), functor );
      }

      template< class Point, class Functor >
      void hessianEach ( const Point &x, Functor functor ) const
      {
        shapeFunctionSet_.hessianEach( position( x ), functor );
      }

      // implementation-defined methods

      const Entity &entity () const { assert( entity_ ); return *entity_; }

    private:
      template< class Point >
      DomainType position ( const Point &x ) const
      {
        DomainType y = entity().geometry().global( Fem::coordinate( x ) ) - bbox_.first;
        for( int k = 0; k < dimDomain; ++k )
          y[ k ] /= (bbox_.second[ k ] - bbox_.first[ k ]);
        return y;
      }

      const Entity *entity_ = nullptr;
      ShapeFunctionSet shapeFunctionSet_;
      std::pair< DomainType, DomainType > bbox_;
    };

  } // namespace Vem

} // namespace Dune
#endif

#endif // #ifndef DUNE_VEM_AGGLOMERATION_SHAPEFUNCTIONSET_HH
