#ifndef DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH
#define DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <type_traits>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/common/coordinate.hh>
#include <dune/fem/space/basisfunctionset/functor.hh>

namespace Dune
{

  namespace Vem
  {

    // BoundingBoxBasisFunctionSet
    // ---------------------------

    template< class Entity, class ShapeFunctionSet >
    class BoundingBoxBasisFunctionSet
    {
      typedef BoundingBoxBasisFunctionSet< Entity, ShapeFunctionSet > ThisType;

    public:
      typedef Entity EntityType;

      typedef typename ShapeFunctionSet::FunctionSpaceType FunctionSpaceType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      typedef ReferenceElement< typename DomainType::field_type, DomainType::dimension > ReferenceElementType;

      BoundingBoxBasisFunctionSet () = default;

      BoundingBoxBasisFunctionSet ( const EntityType &entity, std::pair< DomainType, DomainType > bbox,
                                    ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ), shapeFunctionSet_( std::move( shapeFunctionSet ) ), bbox_( std::move( bbox ) )
      {
        assert( shapeFunctionSet_.type().isCube() );
      }

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return shapeFunctionSet_.size(); }

      const ReferenceElementType &referenceElement () const
      {
        return ReferenceElements< DomainFieldType, DomainType::dimension >::general( entity().type() );
      }

      template< class Quadrature, class Vector, class DofVector >
      void axpy ( const Quadrature &quadrature, const Vector &values, DofVector &dofs ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          axpy( quadrature[ qp ], values[ qp ], dofs );
      }

      template< class Quadrature, class VectorA, class VectorB, class DofVector >
      void axpy ( const Quadrature &quadrature, const VectorA &valuesA, const VectorB &valuesB, DofVector &dofs ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
        {
          axpy( quadrature[ qp ], valuesA[ qp ], dofs );
          axpy( quadrature[ qp ], valuesB[ qp ], dofs );
        }
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const RangeType &valueFactor, DofVector &dofs ) const
      {
        Fem::FunctionalAxpyFunctor< RangeType, DofVector > f( valueFactor, dofs );
        shapeFunctionSet_.evaluateEach( position( x ), f );
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const JacobianRangeType &jacobianFactory, DofVector &dofs ) const
      {
        Fem::FunctionalAxpyFunctor< JacobianRangeType, DofVector > f( jacobianFactor, dofs );
        shapeFunctionSet_.jacobianEach( position( x ), f );
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const RangeType &valueFactor, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        axpy( x, valueFactor, dofs );
        axpy( x, jacobianFactor, dofs );
      }

      template< class Quadrature, class DofVector, class Values >
      void evaluateAll ( const Quadrature &quadrature, const DofVector &dofs, Values &values ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          evaluateAll( quadrature[ qp ], dofs, values[ qp ] );
      }

      template< class Point, class DofVector >
      void evaluateAll ( const Point &x, const DofVector &dofs, RangeType &value ) const
      {
        value = RangeType( 0 );
        Fem::AxpyFunctor< DofVector, RangeType > f( dofs, value );
        shapeFunctionSet_.evaluateEach( position( x ), f );
      }

      template< class Point, class Values > const
      void evaluateAll ( const Point &x, Values &values ) const
      {
        assert( values.size() >= size() );
        Fem::AssignFunctor< Values > f( values );
        shapeFunctionSet().evaluateEach( position( x ), f );
      }

      template< class Point, class DofVector, class Jacobians >
      void jacobianAll ( const Quadrature &quadrature, const DofVector &dofs, Jacobians &jacobians ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          jacobianAll( quadrature[ qp ], dofs, jacobians[ qp ] );
      }

      template< class Point, class DofVector >
      void jacobianAll ( const Point &x, const DofVector &dofs, JacobianRangeType &jacobian ) const
      {
        jacobian = JacobianRangeType( 0 );
        Fem::AxpyFunctor< DofVector, JacobianRangeType > f( dofs, jacobian );
        shapeFunctionSet_.jacobianEach( position( x ), f );
      }

      template< class Point, class Jacobians > const
      void jacobianAll ( const Point &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        Fem::AssignFunctor< Jacobians > f( jacobians );
        shapeFunctionSet().jacobianEach( position( x ), f );
      }

      template< class Point, class DofVector, class Hessians >
      void hessianAll ( const Quadrature &quadrature, const DofVector &dofs, Hessians &hessians ) const
      {
        const std::size_t nop = quadrature.nop();
        for( std::size_t qp = 0; qp < nop; ++qp )
          hessianAll( quadrature[ qp ], dofs, hessians[ qp ] );
      }

      template< class Point, class DofVector >
      void hessianAll ( const Point &x, const DofVector &dofs, HessianRangeType &hessian ) const
      {
        hessian = HessianRangeType( RangeFieldType( 0 ) );
        Fem::AxpyFunctor< DofVector, HessianRangeType > f( dofs, hessian );
        shapeFunctionSet_.hessianEach( position( x ), f );
      }

      template< class Point, class Hessians > const
      void hessianAll ( const Point &x, Hessians &hessians ) const
      {
        assert( hessians.size() >= size() );
        Fem::AssignFunctor< Hessians > f( hessians );
        shapeFunctionSet().hessianEach( position( x ), f );
      }

      const EntityType &entity () const { assert( entity_ ); return *entity_; }

    private:
      template< class Point >
      DomainType position ( const Point &x )
      {
        DomainType y = entity().geometry().global( Fem::coordinate( x ) ) - bbox_.first;
        for( int k = 0; k < DomainType::dimension; ++k )
          y[ k ] /= (bbox_[ k ].second - bbox_[ k ].first);
        return y;
      }

      const EntityType *entity_ = nullptr;
      ShapeFunctionSet shapeFunctionSet_;
      std::pair< DomainType, DomainType > bbox_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH
