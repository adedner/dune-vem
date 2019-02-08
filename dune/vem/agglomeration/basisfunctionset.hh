#ifndef DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH
#define DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <type_traits>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/quadrature/quadrature.hh>
#include <dune/fem/space/basisfunctionset/functor.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/functor.hh>

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

      typedef typename FunctionSpaceType::DomainFieldType DomainFieldType;
      typedef typename FunctionSpaceType::RangeFieldType RangeFieldType;

      typedef typename FunctionSpaceType::DomainType DomainType;
      typedef typename FunctionSpaceType::RangeType RangeType;
      typedef typename FunctionSpaceType::JacobianRangeType JacobianRangeType;
      typedef typename FunctionSpaceType::HessianRangeType HessianRangeType;

      static constexpr int dimDomain = DomainType::dimension;

      typedef ReferenceElement< typename DomainType::field_type, dimDomain > ReferenceElementType;

    private:
      struct Transformation
      {
        Transformation() {}
        explicit Transformation ( const std::pair< DomainType, DomainType > &bbox ) : extentInv_( bbox.second - bbox.first )
        {
          std::transform(extentInv_.begin(),extentInv_.end(),extentInv_.begin(),
              [](const double &x){return 1./x;});
        }

        JacobianRangeType operator() ( JacobianRangeType jacobian ) const
        {
          for( int i = 0; i < RangeType::dimension; ++i )
            applyScalar( jacobian[ i ] );
          return jacobian;
        }

        template< class ScalarJacobian >
        Fem::MakeVectorialExpression< ScalarJacobian, JacobianRangeType > operator() ( Fem::MakeVectorialExpression< ScalarJacobian, JacobianRangeType > jacobian ) const
        {
          applyScalar( jacobian.scalar()[ 0 ] );
          return jacobian;
        }

        HessianRangeType operator() ( HessianRangeType hessian ) const
        {
          for( int i = 0; i < RangeType::dimension; ++i )
            applyScalar( hessian[ i ] );
          return hessian;
        }

        template< class ScalarHessian >
        Fem::MakeVectorialExpression< ScalarHessian, HessianRangeType > operator() ( Fem::MakeVectorialExpression< ScalarHessian, HessianRangeType > hessian ) const
        {
          applyScalar( hessian.scalar()[ 0 ] );
          return hessian;
        }

        void applyScalar ( FieldVector< RangeFieldType, dimDomain > &jacobian ) const
        {
          for( int j = 0; j < dimDomain; ++j )
            jacobian[ j ] *= extentInv_[ j ];
        }

        void applyScalar ( FieldMatrix< RangeFieldType, dimDomain, dimDomain > &hessian ) const
        {
          for( int j = 0; j < dimDomain; ++j )
            for( int k = 0; k < dimDomain; ++k )
              hessian[ j ][ k ] *= (extentInv_[ j ] * extentInv_[ k ]);
        }

        DomainType extentInv_;
      };

    public:
      BoundingBoxBasisFunctionSet ()
      : entity_(nullptr)
      { }

      BoundingBoxBasisFunctionSet ( const EntityType &entity, std::pair< DomainType, DomainType > bbox,
                                    ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ), shapeFunctionSet_( std::move( shapeFunctionSet ) ), bbox_( std::move( bbox ) ),
          transformation_(bbox_)
      {}

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return shapeFunctionSet_.size(); }

      const ReferenceElementType &referenceElement () const
      {
        return ReferenceElements< DomainFieldType, dimDomain >::general( entity().type() );
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
      void axpy ( const Point &x, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        const JacobianRangeType transformedFactor = transformation_( jacobianFactor );
        Fem::FunctionalAxpyFunctor< JacobianRangeType, DofVector > f( transformedFactor, dofs );
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
        shapeFunctionSet_.evaluateEach( position( x ), f );
      }

      template< class Quadrature, class DofVector, class Jacobians >
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
        jacobian = transformation_( jacobian );
      }

      template< class Point, class Jacobians > const
      void jacobianAll ( const Point &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        Fem::AssignFunctor< Jacobians, TransformedAssign< Transformation > > f( jacobians, transformation_ );
        shapeFunctionSet_.jacobianEach( x , f );
      }

      template< class Quadrature, class DofVector, class Hessians >
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
        hessian = transformation_( hessian );
      }

      template< class Point, class Hessians > const
      void hessianAll ( const Point &x, Hessians &hessians ) const
      {
        assert( hessians.size() >= size() );
        Fem::AssignFunctor< Hessians, TransformedAssign< Transformation > > f( hessians, transformation_ );
        shapeFunctionSet_.hessianEach( position( x ), f );
      }

      const EntityType &entity () const { assert( entity_ ); return *entity_; }

    private:
      template< class Point >
      DomainType position ( const Point &x ) const
      {
        DomainType y = entity().geometry().global( Fem::coordinate( x ) ) - bbox_.first;
        for( int k = 0; k < dimDomain; ++k )
          y[ k ] /= (bbox_.second[ k ] - bbox_.first[ k ]);
        return y;
      }

      const EntityType *entity_ = nullptr;
      ShapeFunctionSet shapeFunctionSet_;
      std::pair< DomainType, DomainType > bbox_;
      Transformation transformation_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_AGGLOMERATION_BASISFUNCTIONSET_HH
