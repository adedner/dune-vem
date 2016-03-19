#ifndef DUNE_VEM_SPACE_BASISFUNCTIONSET_HH
#define DUNE_VEM_SPACE_BASISFUNCTIONSET_HH

#include <cassert>
#include <cstddef>

#include <algorithm>
#include <type_traits>
#include <utility>

#include <dune/geometry/referenceelements.hh>

#include <dune/fem/common/fmatrixcol.hh>
#include <dune/fem/quadrature/quadrature.hh>
#include <dune/fem/space/basisfunctionset/functor.hh>
#include <dune/fem/space/shapefunctionset/vectorial.hh>

#include <dune/vem/agglomeration/functor.hh>

namespace Dune
{

  namespace Vem
  {

    // VEMBasisFunctionSet
    // -------------------

    // TODO: add template arguments for ValueProjection and JacobianProjection

    template< class Entity, class ShapeFunctionSet >
    class VEMBasisFunctionSet
    {
      typedef VEMBasisFunctionSet< Entity, ShapeFunctionSet > ThisType;

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

      typedef std::vector< std::vector< DomainFieldType > > ValueProjection;
      typedef std::vector< std::vector< DomainType > > JacobianProjection;

      VEMBasisFunctionSet () = default;

      VEMBasisFunctionSet ( const EntityType &entity, std::pair< DomainType, DomainType > bbox,
                            ValueProjection valueProjection, JacobianProjection jacobianProjection,
                            ShapeFunctionSet shapeFunctionSet = ShapeFunctionSet() )
        : entity_( &entity ),
          shapeFunctionSet_( std::move( shapeFunctionSet ) ),
          valueProjection_( std::move( valueProjection ) ),
          jacobianProjection_( std::move( jacobianProjection ) ),
          bbox_( std::move( bbox ) )
      {}

      int order () const { return shapeFunctionSet_.order(); }

      std::size_t size () const { return valueProjection_[0].size(); }

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
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &valueFactor, &dofs ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              dofs[ j ] += valueProjection_[ alpha ][ j ] * (valueFactor * phi_alpha);
          } );
      }

      template< class Point, class DofVector >
      void axpy ( const Point &x, const JacobianRangeType &jacobianFactor, DofVector &dofs ) const
      {
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &jacobianFactor, &dofs ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              for( int k = 0; k < dimDomain; ++k )
              {
                FieldMatrixColumn< const JacobianRangeType > jacobianFactor_k( jacobianFactor, k );
                dofs[ j ] += jacobianProjection_[ alpha ][ j ][ k ] * (jacobianFactor_k * phi_alpha);
              }
          } );
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
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &dofs, &value ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              value.axpy( valueProjection_[ alpha ][ j ]*dofs[ j ], phi_alpha );
          } );
      }

      template< class Point, class Values > const
      void evaluateAll ( const Point &x, Values &values ) const
      {
        assert( values.size() >= size() );
        std::fill( values.begin(), values.end(), RangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &values ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              values[ j ].axpy( valueProjection_[ alpha ][ j ], phi_alpha );
          } );
      }

      // TODO: use lower order shape function set for Jacobian

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
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &dofs, &jacobian ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              for( int k = 0; k < dimDomain; ++k )
              {
                FieldMatrixColumn< JacobianRangeType > jacobian_k( jacobian, k );
                jacobian_k.axpy( jacobianProjection_[ alpha ][ j ][ k ]*dofs[ j ], phi_alpha );
              }
          } );
      }

      template< class Point, class Jacobians > const
      void jacobianAll ( const Point &x, Jacobians &jacobians ) const
      {
        assert( jacobians.size() >= size() );
        std::fill( jacobians.begin(), jacobians.end(), JacobianRangeType( 0 ) );
        shapeFunctionSet_.evaluateEach( position( x ), [ this, &jacobians ] ( std::size_t alpha, RangeType phi_alpha ) {
            for( std::size_t j = 0; j < size(); ++j )
              for( int k = 0; k < dimDomain; ++k )
              {
                FieldMatrixColumn< JacobianRangeType > jacobian_jk( jacobians[ j ], k );
                jacobian_jk.axpy( jacobianProjection_[ alpha ][ j ][ k ], phi_alpha );
              }
        } );
      }

      template< class Quadrature, class DofVector, class Hessians >
      void hessianAll ( const Quadrature &quadrature, const DofVector &dofs, Hessians &hessians ) const
      {
        DUNE_THROW( NotImplemented, "hessians not implemented for VEMBasisFunctionSet" );
      }

      template< class Point, class DofVector >
      void hessianAll ( const Point &x, const DofVector &dofs, HessianRangeType &hessian ) const
      {
        DUNE_THROW( NotImplemented, "hessians not implemented for VEMBasisFunctionSet" );
      }

      template< class Point, class Hessians > const
      void hessianAll ( const Point &x, Hessians &hessians ) const
      {
        DUNE_THROW( NotImplemented, "hessians not implemented for VEMBasisFunctionSet" );
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
      ValueProjection valueProjection_;
      JacobianProjection jacobianProjection_;
      std::pair< DomainType, DomainType > bbox_;
    };

  } // namespace Vem

} // namespace Dune

#endif // #ifndef DUNE_VEM_SPACE_BASISFUNCTIONSET_HH
